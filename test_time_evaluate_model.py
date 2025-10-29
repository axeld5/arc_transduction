"""
Test-time RL evaluation for ARC transduction problems.

This module implements test-time adaptation:
1. Takes a problem from evaluation set
2. Creates augmented training data with placeholders (all 4 levels)
3. Performs RL training with dense + sparse reward
4. Evaluates the adapted model using vLLM with various strategies
"""

import json
from typing import *
import os
import random
import gc
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from datasets import Dataset

from create_train_data import (
    create_training_examples_for_problem_eval,
)
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


def create_test_time_training_data(
    problem: Dict[str, Any],
    num_repetitions: int = 50,
    include_all_levels: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create training data from a single problem for test-time RL.
    
    For test-time RL, we use the problem as-is (already formatted text) and create
    multiple repetitions to give the model sufficient training signal. We include
    examples from all difficulty levels (1-4) to provide comprehensive training.
    
    Args:
        problem: Problem data with 'problem' and 'answer' keys
        num_repetitions: Number of times to repeat the problem
        include_all_levels: If True, tag examples with different levels for curriculum
        
    Returns:
        List of training examples
    """
    # Extract problem metadata if available
    metadata = problem.get('metadata', {})
    problem_name = metadata.get('problem_name', 'test_problem')
    concept = metadata.get('concept', 'unknown')
    original_level = metadata.get('level', 0)
    
    training_examples = []
    
    if include_all_levels:
        # Create examples for each level (1-4) to simulate curriculum
        # This helps the model learn progressively
        levels = [1, 2, 3, 4]
        repetitions_per_level = num_repetitions // len(levels)
        
        for level in levels:
            for rep_idx in range(repetitions_per_level):
                training_examples.append({
                    'problem': problem['problem'],
                    'answer': problem['answer'],
                    'metadata': {
                        'concept': concept,
                        'problem_name': problem_name,
                        'method': 'test_time',
                        'original_level': original_level,
                        'level': level,  # Assigned level for training
                        'repetition_idx': rep_idx,
                    }
                })
    else:
        # Simple repetition without level assignment
        for rep_idx in range(num_repetitions):
            training_examples.append({
                'problem': problem['problem'],
                'answer': problem['answer'],
                'metadata': {
                    'concept': concept,
                    'problem_name': problem_name,
                    'method': 'test_time',
                    'original_level': original_level,
                    'level': original_level,
                    'repetition_idx': rep_idx,
                }
            })
    
    return training_examples


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
        learning_rate=1e-7,
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
        learning_rate=1e-7,
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
    output_dir: str = "test_time_rl",
    problem_idx: int = 0,
    level_filter: int = 3,
    num_repetitions: int = 200,
    include_all_levels: bool = True,
    use_system_prompt: bool = True,
    dense_steps: int = 50,
    sparse_steps: int = 150,
    inference_mode: str = "standard",
    temperature: float = 0.7,
    attempts_per_problem: int = 5,
) -> Dict[str, Any]:
    """
    Perform test-time RL evaluation on a single problem.
    
    Args:
        base_model: Path to base model
        eval_data_path: Path to evaluation data JSON
        output_dir: Base directory for saving outputs
        problem_idx: Index of problem to evaluate (within filtered level)
        level_filter: Only evaluate problems of this level (default: 3 for 3x3 zeros)
        num_repetitions: Number of repetitions of the problem for training
        include_all_levels: Whether to distribute repetitions across levels 1-4
        use_system_prompt: Whether to use system prompt
        dense_steps: Steps for dense reward phase
        sparse_steps: Steps for sparse reward phase
        inference_mode: Evaluation strategy ("standard", "deep_dive", "augmented_voting", "augmented_deep_dive")
        temperature: Sampling temperature
        attempts_per_problem: Number of generation attempts
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*80}")
    print(f"TEST-TIME RL EVALUATION")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Eval data: {eval_data_path}")
    print(f"Level filter: {level_filter} (3x3 zeros placeholder)")
    print(f"Problem index: {problem_idx} (within level {level_filter})")
    print(f"Inference mode: {inference_mode}")
    print(f"Repetitions: {num_repetitions}")
    print(f"Include all levels: {include_all_levels}")
    print(f"Dense steps: {dense_steps}")
    print(f"Sparse steps: {sparse_steps}")
    print(f"{'='*80}\n")
    
    # Step 1: Load the problem (filtered by level)
    print(f"[Step 1] Loading problem {problem_idx} from level {level_filter}...")
    problem = load_eval_problem(eval_data_path, problem_idx, level_filter=level_filter)
    print(f"Problem metadata: {problem.get('metadata', {})}")
    
    # Step 2: Create training data
    print(f"\n[Step 2] Creating training data...")
    training_data = create_test_time_training_data(
        problem=problem,
        num_repetitions=num_repetitions,
        include_all_levels=include_all_levels,
    )
    print(f"Created {len(training_data)} training examples")
    
    # Step 3: Perform test-time RL
    print(f"\n[Step 3] Performing test-time RL...")
    problem_output_dir = os.path.join(output_dir, f"problem_{problem_idx}")
    adapter_path = run_test_time_rl(
        base_model=base_model,
        training_data=training_data,
        output_dir=problem_output_dir,
        use_system_prompt=use_system_prompt,
        dense_steps=dense_steps,
        sparse_steps=sparse_steps,
    )
    
    # Step 4: Create temporary eval file with just this problem
    print(f"\n[Step 4] Evaluating adapted model...")
    temp_eval_path = os.path.join(output_dir, f"temp_eval_problem_{problem_idx}.json")
    with open(temp_eval_path, 'w') as f:
        json.dump([problem], f, indent=2)
    
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
    print(f"TEST-TIME RL EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Adapter path: {adapter_path}")
    print(f"Results: {results}")
    print(f"{'='*80}\n")
    
    return {
        'problem_idx': problem_idx,
        'adapter_path': adapter_path,
        'results': results,
        'problem_metadata': problem.get('metadata', {}),
    }


def batch_test_time_evaluate(
    base_model: str,
    eval_data_path: str,
    output_dir: str = "test_time_rl_batch",
    level_filter: int = 3,
    num_problems: int = 10,
    start_idx: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform test-time RL evaluation on multiple problems.
    
    Args:
        base_model: Path to base model
        eval_data_path: Path to evaluation data JSON
        output_dir: Base directory for saving outputs
        level_filter: Only evaluate problems of this level (default: 3)
        num_problems: Number of problems to evaluate
        start_idx: Starting problem index (within filtered level)
        **kwargs: Additional arguments passed to test_time_evaluate
        
    Returns:
        Dictionary with results for all problems
    """
    print(f"\n{'='*80}")
    print(f"BATCH TEST-TIME RL EVALUATION")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
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
    # Example: Test-time RL on a single problem (Level 3 = 3x3 zeros placeholder)
    result = test_time_evaluate(
        base_model="qwen2.5_3b_transduction_sft/merged",
        eval_data_path="generated_data/eval_data.json",
        output_dir="test_time_rl_output",
        problem_idx=0,
        level_filter=3,  # Only evaluate Level 3 (3x3 zeros placeholder)
        num_repetitions=200,  # Total repetitions of the problem
        include_all_levels=True,  # Distribute across levels 1-4
        use_system_prompt=True,
        dense_steps=50,  # Phase 1: Dense reward
        sparse_steps=150,  # Phase 2: Sparse reward
        inference_mode="standard",  # Options: "standard", "deep_dive", "augmented_voting", "augmented_deep_dive"
        temperature=1,
        attempts_per_problem=5,
    )
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))

