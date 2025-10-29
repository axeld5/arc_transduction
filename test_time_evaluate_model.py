"""
Test-time adaptation evaluation for ARC transduction problems.

This module implements test-time adaptation using SFT or RL:
1. Loads a level 3 problem (3x3 zeros placeholder) from evaluation set
2. Creates augmented training data with ALL placeholder levels (0-4) using leave-one-out:
   - Parses the problem text back to structured format
   - Uses leave-one-out on training examples (holds out one, trains on rest)
   - Generates multiple augmentations with placeholders at all difficulty levels
   - We don't use the test case since we don't have access to its output
3. Performs test-time adaptation:
   - SFT: Supervised finetuning on the diverse training data (faster, simpler)
   - RL: Reinforcement learning with dense + sparse reward (more complex)
4. Evaluates the adapted model on the ORIGINAL level 3 problem using vLLM

Training: Leave-one-out with ALL levels (0-4) and augmentations
Evaluation: ONLY the original level 3 problem with test case
Methods: SFT (supervised finetuning) or RL (reinforcement learning)
"""

import json
from typing import *
import os
import random
import gc
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

from create_train_data import (
    create_training_examples_for_problem_eval,
    augment_problem,
    from_data_to_problem,
)
from placeholder_creation import create_placeholder
from reward_functions import reward_function_diff
from evaluate_model import evaluate_model_vllm
from transduction_rl_unigpu import convert_to_rl_format

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass


def load_eval_problem(
    eval_data_path: str, 
    problem_idx: int, 
    level_filter: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load a specific problem from evaluation data, optionally filtered by level.
    
    Args:
        eval_data_path: Path to evaluation data JSON
        problem_idx: Index of problem to load (within filtered set if level_filter is used)
        level_filter: If provided, only consider problems of this level
        
    Returns:
        Problem data dictionary
    """
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Filter by level if requested
    if level_filter is not None:
        eval_data = [p for p in eval_data if p.get('metadata', {}).get('level') == level_filter]
        if not eval_data:
            raise ValueError(f"No problems found for level {level_filter}")
        print(f"Filtered to {len(eval_data)} problems at level {level_filter}")
    
    if problem_idx >= len(eval_data):
        raise ValueError(f"Problem index {problem_idx} out of range (max: {len(eval_data)-1})")
    
    return eval_data[problem_idx]


def parse_problem_text(problem_text: str, answer_text: str) -> Dict[str, Any]:
    """
    Parse formatted problem text back into structured problem data.
    
    Args:
        problem_text: Formatted problem text with training examples and test input
        answer_text: The ground truth answer
        
    Returns:
        Structured problem data with train/test examples
    """
    lines = problem_text.strip().split('\n')
    problem_data = {'train': [], 'test': []}
    
    current_section = None
    current_grid = []
    current_io = None  # 'input' or 'output'
    current_example = {}
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Identify'):
            continue
            
        if line == "Training Examples:":
            current_section = 'train'
            continue
        elif line == "Test Case:":
            # Save any pending training example before switching to test
            if current_section == 'train' and current_example and current_io == 'output' and current_grid:
                current_example['output'] = current_grid
                problem_data['train'].append(current_example)
                current_example = {}
                current_grid = []
            current_section = 'test'
            current_io = None
            continue
        elif line.startswith("Example "):
            # Save previous example if exists
            if current_example and 'input' in current_example and current_io == 'output' and current_grid:
                current_example['output'] = current_grid
                problem_data['train'].append(current_example)
            current_example = {}
            current_grid = []
            current_io = None
            continue
        elif line == "Input:":
            # Save output if we were collecting one
            if current_io == 'output' and current_grid and 'input' in current_example:
                current_example['output'] = current_grid
                problem_data['train'].append(current_example)
                current_example = {}
            current_io = 'input'
            current_grid = []
            continue
        elif line == "Output:" or line == "Output Placeholder:":
            if current_io == 'input' and current_grid:
                current_example['input'] = current_grid
            current_io = 'output'
            current_grid = []
            continue
        
        # Parse grid row
        if current_io and line:
            try:
                row = [int(x) for x in line.split()]
                if row:  # Only add non-empty rows
                    current_grid.append(row)
            except ValueError:
                pass
    
    # Finalize last training example if exists
    if current_section == 'train' and current_io == 'output' and current_grid and 'input' in current_example:
        current_example['output'] = current_grid
        problem_data['train'].append(current_example)
    
    # Parse test case
    if current_section == 'test' and current_grid:
        # current_grid should be the test input
        # Parse the answer for test output
        answer_lines = answer_text.strip().split('\n')
        answer_grid = []
        for line in answer_lines:
            try:
                row = [int(x) for x in line.split()]
                if row:
                    answer_grid.append(row)
            except ValueError:
                pass
        
        problem_data['test'].append({
            'input': current_grid,
            'output': answer_grid
        })
    
    return problem_data


def create_test_time_training_data(
    problem: Dict[str, Any],
    num_augmentations: int = 10,
    placeholders_per_level: int = 5,
) -> List[Dict[str, Any]]:
    """
    Create training data from a single problem for test-time RL using leave-one-out.
    
    Since we don't have access to the test output during test-time, we use leave-one-out
    on the training examples: hold out one training example as "test", use the rest as training.
    Generates ALL placeholder levels (0-4) with augmentations to give the model diverse training signal.
    
    Args:
        problem: Problem data with 'problem', 'answer', and 'metadata' keys
        num_augmentations: Number of augmented versions to create
        placeholders_per_level: Number of placeholders per level (1-4)
        
    Returns:
        List of training examples with all placeholder levels
    """
    # Extract problem metadata
    metadata = problem.get('metadata', {})
    problem_name = metadata.get('problem_name', 'test_problem')
    concept = metadata.get('concept', 'unknown')
    
    # Parse the formatted text back into structured data
    problem_data = parse_problem_text(problem['problem'], problem['answer'])
    
    # Check if there are enough training examples for leave-one-out
    train_examples = problem_data.get('train', [])
    if len(train_examples) < 2:
        print(f"Warning: Problem has fewer than 2 training examples, cannot do leave-one-out")
        return []
    
    training_examples = []
    
    # Use leave-one-out: randomly select one training example to hold out
    held_out_idx = random.randint(0, len(train_examples) - 1)
    
    # Create modified problem data with leave-one-out
    modified_data = {
        'train': [ex for i, ex in enumerate(train_examples) if i != held_out_idx],
        'test': [train_examples[held_out_idx]]
    }
    
    # Create augmented versions with all placeholder levels
    for aug_idx in range(num_augmentations):
        seed = hash(f"{problem_name}_testtime_{held_out_idx}_{aug_idx}") % (2**32)
        random.seed(seed)
        augmented_data = augment_problem(modified_data, seed=seed)
        augmented_ground_truth = augmented_data['test'][0]['output']
        
        # Level 0: Unmodified (ground truth in test output)
        unmodified_problem = {
            'train': augmented_data['train'],
            'test': [{
                'input': augmented_data['test'][0]['input'],
                'output': augmented_ground_truth
            }]
        }
        
        formatted_unmodified = from_data_to_problem(unmodified_problem, include_test_output=True)
        answer = ""
        for row in augmented_ground_truth:
            answer += " ".join(map(str, row)) + "\n"
        answer = answer.strip()
        
        training_examples.append({
            'problem': formatted_unmodified['problem'],
            'answer': answer,
            'metadata': {
                'concept': concept,
                'problem_name': problem_name,
                'method': 'test_time_leave_one_out',
                'held_out_idx': held_out_idx,
                'augmentation_idx': aug_idx,
                'augmentation_seed': seed,
                'level': 0,
                'placeholder_idx': -1,
                'placeholder_seed': -1
            }
        })
        
        # Levels 1-4: Create placeholders
        for level in range(1, 5):
            for placeholder_idx in range(placeholders_per_level):
                placeholder_seed = hash(f"{problem_name}_testtime_{held_out_idx}_{aug_idx}_{level}_{placeholder_idx}") % (2**32)
                random.seed(placeholder_seed)
                
                placeholder = create_placeholder(augmented_data, augmented_ground_truth, level=level)
                
                problem_with_placeholder = {
                    'train': augmented_data['train'],
                    'test': [{
                        'input': augmented_data['test'][0]['input'],
                        'output': placeholder
                    }]
                }
                
                formatted = from_data_to_problem(problem_with_placeholder, include_test_output=True)
                
                training_examples.append({
                    'problem': formatted['problem'],
                    'answer': answer,  # Always ground truth
                    'metadata': {
                        'concept': concept,
                        'problem_name': problem_name,
                        'method': 'test_time_leave_one_out',
                        'held_out_idx': held_out_idx,
                        'augmentation_idx': aug_idx,
                        'augmentation_seed': seed,
                        'level': level,
                        'placeholder_idx': placeholder_idx,
                        'placeholder_seed': placeholder_seed
                    }
                })
    
    return training_examples


def run_test_time_sft(
    base_model: str,
    training_data: List[Dict[str, Any]],
    output_dir: str,
    use_system_prompt: bool = True,
    num_steps: int = 100,
) -> str:
    """
    Perform test-time SFT (supervised finetuning) on a single problem.
    
    Args:
        base_model: Path to base model
        training_data: Training examples created from the problem
        output_dir: Directory to save adapter
        use_system_prompt: Whether to use system prompt
        num_steps: Training steps
        
    Returns:
        Path to the trained adapter
    """
    print(f"\n{'='*60}")
    print(f"Test-Time SFT Training")
    print(f"Training samples: {len(training_data)}")
    print(f"Training steps: {num_steps}")
    print(f"Using system prompt: {use_system_prompt}")
    print(f"{'='*60}")
    
    max_seq_length = 16000
    lora_rank = 256
    
    # Load model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            max_lora_rank=lora_rank,
            fast_inference=False,
        )
    except Exception as e:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen2.5_3b_transduction_sft/final",
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            max_lora_rank=lora_rank,
            fast_inference=False,
        )
        model.save_pretrained_merged(
            "qwen2.5_3b_transduction_sft/merged",
            tokenizer,
            save_method="merged_16bit",
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            max_lora_rank=lora_rank,
            fast_inference=False,
        )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    model.get_input_embeddings().requires_grad_(True)
    
    # Set up chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )
    
    # Convert to SFT format
    def formatting_prompts_func(examples):
        texts = []
        for problem, answer in zip(examples['problem'], examples['answer']):
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": "You are an expert at solving ARC (Abstraction and Reasoning Corpus) problems. Given input-output grid examples, identify the transformation pattern and apply it to solve new test cases."},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": answer}
                ]
            else:
                messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": answer}
                ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}
    
    # Prepare dataset
    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Training
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=num_steps,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=num_steps,  # Save at end
        optim="paged_adamw_8bit",
        warmup_steps=10,
        report_to="tensorboard",
        dataset_text_field="text",
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    trainer.train()
    
    # Save adapter
    final_adapter_path = os.path.join(output_dir, "final")
    trainer.save_model(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    # Cleanup
    del trainer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"SFT Complete. Adapter saved to: {final_adapter_path}")
    
    return final_adapter_path


def run_test_time_rl(
    base_model: str,
    training_data: List[Dict[str, Any]],
    output_dir: str,
    use_system_prompt: bool = True,
    dense_steps: int = 50,
    sparse_steps: int = 150,
) -> str:
    """
    Perform test-time RL training on a single problem.
    
    Args:
        base_model: Path to base model
        training_data: Training examples created from the problem
        output_dir: Directory to save adapter
        use_system_prompt: Whether to use system prompt
        dense_steps: Steps for dense reward phase
        sparse_steps: Steps for sparse reward phase
        
    Returns:
        Path to the trained adapter
    """
    print(f"\n{'='*60}")
    print(f"Test-Time RL Training")
    print(f"Training samples: {len(training_data)}")
    print(f"Dense steps: {dense_steps}")
    print(f"Sparse steps: {sparse_steps}")
    print(f"Using system prompt: {use_system_prompt}")
    print(f"{'='*60}")
    
    max_seq_length = 16000
    lora_rank = 256
    
    # Load model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            max_lora_rank=lora_rank,
            fast_inference=False,
        )
    except Exception as e:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen2.5_3b_transduction_sft/final",
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            max_lora_rank=lora_rank,
            fast_inference=False,
        )
        model.save_pretrained_merged(
            "qwen2.5_3b_transduction_sft/merged",
            tokenizer,
            save_method="merged_16bit",
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            max_lora_rank=lora_rank,
            fast_inference=False,
        )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    model.get_input_embeddings().requires_grad_(True)
    
    # Convert to RL format
    converted = convert_to_rl_format(training_data, use_system_prompt=use_system_prompt)
    dataset = Dataset.from_list(converted)
    print(f"Dataset size: {len(dataset)}")
    
    # Phase 1: Dense reward training
    print(f"\n[Phase 1] Dense reward training ({dense_steps} steps)...")
    
    def dense_reward_func(completions, expected_output, **kwargs):
        return reward_function_diff(completions, expected_output, use_dense_reward=True, **kwargs)
    dense_reward_func.__name__ = "reward_function_dense"
    
    training_args_dense = GRPOConfig(
        use_vllm=False,
        importance_sampling_level="sequence",
        loss_type="grpo",
        output_dir=f"{output_dir}_dense",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        beta=0.04,
        epsilon=3e-4,
        max_steps=dense_steps,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=dense_steps,  # Save at end
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        num_generations=4,
        max_prompt_length=8192,
        max_completion_length=4096,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer_dense = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[dense_reward_func],
        args=training_args_dense,
        train_dataset=dataset,
    )
    
    trainer_dense.train()
    
    # Save dense adapter
    dense_adapter_path = os.path.join(f"{output_dir}_dense", "final")
    trainer_dense.save_model(dense_adapter_path)
    tokenizer.save_pretrained(dense_adapter_path)
    
    # Cleanup
    del trainer_dense
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"[Phase 1] Complete. Adapter saved to: {dense_adapter_path}")
    
    # Phase 2: Load dense adapter and continue with sparse reward
    print(f"\n[Phase 2] Sparse reward training ({sparse_steps} steps)...")
    
    # Reload model with dense adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=dense_adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        max_lora_rank=lora_rank,
        fast_inference=False,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    model.get_input_embeddings().requires_grad_(True)
    
    def sparse_reward_func(completions, expected_output, **kwargs):
        return reward_function_diff(completions, expected_output, use_dense_reward=False, **kwargs)
    sparse_reward_func.__name__ = "reward_function_sparse"
    
    training_args_sparse = GRPOConfig(
        use_vllm=False,
        importance_sampling_level="sequence",
        loss_type="grpo",
        output_dir=f"{output_dir}_sparse",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        beta=0.04,
        epsilon=3e-4,
        max_steps=sparse_steps,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=sparse_steps,  # Save at end
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        num_generations=4,
        max_prompt_length=8192,
        max_completion_length=4096,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer_sparse = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[sparse_reward_func],
        args=training_args_sparse,
        train_dataset=dataset,
    )
    
    trainer_sparse.train()
    
    # Save final adapter
    final_adapter_path = os.path.join(f"{output_dir}_sparse", "final")
    trainer_sparse.save_model(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    # Cleanup
    del trainer_sparse
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"[Phase 2] Complete. Final adapter saved to: {final_adapter_path}")
    
    return final_adapter_path


def test_time_evaluate(
    base_model: str,
    eval_data_path: str,
    output_dir: str = "test_time_adaptation",
    problem_idx: int = 0,
    level_filter: int = 3,
    num_augmentations: int = 10,
    placeholders_per_level: int = 5,
    use_system_prompt: bool = True,
    method: str = "sft",
    sft_steps: int = 100,
    dense_steps: int = 50,
    sparse_steps: int = 150,
    inference_mode: str = "standard",
    temperature: float = 0.7,
    attempts_per_problem: int = 5,
) -> Dict[str, Any]:
    """
    Perform test-time adaptation evaluation on a single problem.
    
    Args:
        base_model: Path to base model
        eval_data_path: Path to evaluation data JSON
        output_dir: Base directory for saving outputs
        problem_idx: Index of problem to evaluate (within filtered level)
        level_filter: Only evaluate problems of this level (default: 3 for 3x3 zeros)
        num_augmentations: Number of augmented versions to create for training
        placeholders_per_level: Number of placeholders per level (1-4) for training
        use_system_prompt: Whether to use system prompt
        method: "sft" for supervised finetuning or "rl" for reinforcement learning
        sft_steps: Steps for SFT training (if method="sft")
        dense_steps: Steps for dense reward phase (if method="rl")
        sparse_steps: Steps for sparse reward phase (if method="rl")
        inference_mode: Evaluation strategy ("standard", "deep_dive", "augmented_voting", "augmented_deep_dive")
        temperature: Sampling temperature
        attempts_per_problem: Number of generation attempts
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*80}")
    print(f"TEST-TIME {method.upper()} EVALUATION")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Eval data: {eval_data_path}")
    print(f"Level filter: {level_filter} (3x3 zeros placeholder)")
    print(f"Problem index: {problem_idx} (within level {level_filter})")
    print(f"Method: {method.upper()}")
    print(f"Inference mode: {inference_mode}")
    print(f"Augmentations: {num_augmentations}")
    print(f"Placeholders per level: {placeholders_per_level}")
    print(f"Total training examples: {num_augmentations * (1 + 4 * placeholders_per_level)}")
    if method == "sft":
        print(f"SFT steps: {sft_steps}")
    elif method == "rl":
        print(f"Dense steps: {dense_steps}")
        print(f"Sparse steps: {sparse_steps}")
    print(f"{'='*80}\n")
    
    # Step 1: Load the problem (filtered by level 3)
    print(f"[Step 1] Loading level {level_filter} problem {problem_idx}...")
    problem = load_eval_problem(eval_data_path, problem_idx, level_filter=level_filter)
    print(f"Problem metadata: {problem.get('metadata', {})}")
    
    # Step 2: Create training data with ALL placeholder levels (0-4) using leave-one-out
    print(f"\n[Step 2] Creating training data with ALL placeholder levels (0-4) using leave-one-out...")
    print(f"  (Holding out one training example, training on the rest)")
    training_data = create_test_time_training_data(
        problem=problem,
        num_augmentations=num_augmentations,
        placeholders_per_level=placeholders_per_level,
    )
    print(f"Created {len(training_data)} training examples")
    level_counts = {}
    for ex in training_data:
        lvl = ex['metadata']['level']
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    print(f"Level distribution: {level_counts}")
    
    # Step 3: Perform test-time adaptation (SFT or RL) on ALL placeholder levels
    print(f"\n[Step 3] Performing test-time {method.upper()} on all placeholder levels...")
    problem_output_dir = os.path.join(output_dir, f"problem_{problem_idx}")
    
    if method == "sft":
        adapter_path = run_test_time_sft(
            base_model=base_model,
            training_data=training_data,
            output_dir=problem_output_dir,
            use_system_prompt=use_system_prompt,
            num_steps=sft_steps,
        )
    elif method == "rl":
        adapter_path = run_test_time_rl(
            base_model=base_model,
            training_data=training_data,
            output_dir=problem_output_dir,
            use_system_prompt=use_system_prompt,
            dense_steps=dense_steps,
            sparse_steps=sparse_steps,
        )
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'sft' or 'rl'")
    
    # Step 4: Create temporary eval file with ORIGINAL level 3 problem
    print(f"\n[Step 4] Evaluating adapted model on ORIGINAL level {level_filter} problem...")
    temp_eval_path = os.path.join(output_dir, f"temp_eval_problem_{problem_idx}.json")
    with open(temp_eval_path, 'w') as f:
        json.dump([problem], f, indent=2)  # Use original level 3 problem for evaluation
    
    # Step 5: Evaluate using vLLM with the adapter
    results = evaluate_model_vllm(
        model_path=base_model,
        eval_data_path=temp_eval_path,
        max_samples_per_level=1,
        attempts_per_problem=attempts_per_problem,
        temperature=temperature,
        use_system_prompt=use_system_prompt,
        use_lora=True,
        lora_path=adapter_path,
        print_examples=True,
        inference_mode=inference_mode,
    )
    
    # Cleanup temp file
    os.remove(temp_eval_path)
    
    print(f"\n{'='*80}")
    print(f"TEST-TIME {method.upper()} EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Method: {method.upper()}")
    print(f"Adapter path: {adapter_path}")
    print(f"Results: {results}")
    print(f"{'='*80}\n")
    
    return {
        'problem_idx': problem_idx,
        'method': method,
        'adapter_path': adapter_path,
        'results': results,
        'problem_metadata': problem.get('metadata', {}),
    }


def batch_test_time_evaluate(
    base_model: str,
    eval_data_path: str,
    output_dir: str = "test_time_batch",
    level_filter: int = 3,
    num_problems: int = 10,
    start_idx: int = 0,
    method: str = "sft",
    **kwargs
) -> Dict[str, Any]:
    """
    Perform test-time adaptation evaluation on multiple problems.
    
    Args:
        base_model: Path to base model
        eval_data_path: Path to evaluation data JSON
        output_dir: Base directory for saving outputs
        level_filter: Only evaluate problems of this level (default: 3)
        num_problems: Number of problems to evaluate
        start_idx: Starting problem index (within filtered level)
        method: "sft" for supervised finetuning or "rl" for reinforcement learning
        **kwargs: Additional arguments passed to test_time_evaluate
        
    Returns:
        Dictionary with results for all problems
    """
    print(f"\n{'='*80}")
    print(f"BATCH TEST-TIME {method.upper()} EVALUATION")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Method: {method.upper()}")
    print(f"Level filter: {level_filter}")
    print(f"Evaluating {num_problems} problems starting from index {start_idx}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for i in range(num_problems):
        problem_idx = start_idx + i
        print(f"\n{'#'*80}")
        print(f"Processing problem {i+1}/{num_problems} (index {problem_idx} in level {level_filter})")
        print(f"{'#'*80}\n")
        
        try:
            result = test_time_evaluate(
                base_model=base_model,
                eval_data_path=eval_data_path,
                output_dir=output_dir,
                problem_idx=problem_idx,
                level_filter=level_filter,
                method=method,
                **kwargs
            )
            all_results[f"problem_{problem_idx}"] = result
        except Exception as e:
            print(f"ERROR processing problem {problem_idx}: {e}")
            all_results[f"problem_{problem_idx}"] = {
                'error': str(e),
                'problem_idx': problem_idx,
            }
        
        # Save intermediate results
        results_path = os.path.join(output_dir, "batch_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BATCH EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    # Example: Test-time adaptation on a single problem (Level 3 = 3x3 zeros placeholder)
    # Training: Uses leave-one-out on training examples with all placeholder levels (0-4)
    # Evaluation: Tests on the original level 3 problem with test case
    
    # Method 1: Supervised Finetuning (SFT) - Faster and simpler
    result = test_time_evaluate(
        base_model="qwen2.5_3b_transduction_sft/merged",
        eval_data_path="generated_data/eval_data.json",
        output_dir="test_time_sft_output",
        problem_idx=0,
        level_filter=3,  # Only evaluate Level 3 (3x3 zeros placeholder)
        num_augmentations=30,  # Number of augmented versions for leave-one-out training
        placeholders_per_level=5,  # Placeholders per level (1-4) for training
        # Total training examples: 30 * (1 + 5*4) = 630 examples across all levels
        # These are generated from leave-one-out on training examples
        use_system_prompt=True,
        method="sft",  # Use supervised finetuning
        sft_steps=100,  # Number of training steps
        inference_mode="standard",  # Options: "standard", "deep_dive", "augmented_voting", "augmented_deep_dive"
        temperature=1,
        attempts_per_problem=5,
    )
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))
    
    # Method 2: Reinforcement Learning (RL) - More complex but potentially better
    # Uncomment to use RL instead:
    # result = test_time_evaluate(
    #     base_model="qwen2.5_3b_transduction_sft/merged",
    #     eval_data_path="generated_data/eval_data.json",
    #     output_dir="test_time_rl_output",
    #     problem_idx=0,
    #     level_filter=3,
    #     num_augmentations=30,
    #     placeholders_per_level=5,
    #     use_system_prompt=True,
    #     method="rl",  # Use reinforcement learning
    #     dense_steps=50,  # Phase 1: Dense reward
    #     sparse_steps=150,  # Phase 2: Sparse reward
    #     inference_mode="standard",
    #     temperature=1,
    #     attempts_per_problem=5,
    # )

