import json
from typing import *
import os
import platform
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import re
from datasets import Dataset
from vllm import SamplingParams

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

# ========== UTILITY FUNCTIONS ==========

def check_array(output_string: str) -> bool:
    if not output_string or not isinstance(output_string, str):
        return False
    response = output_string.strip()
    if not response:
        return False
    if '\n' in response:
        grid_match = re.search(r'[0-9\n\s]+', response)
        if not grid_match:
            return False
        grid_str = grid_match.group()
        try:
            rows = grid_str.split('\n')
            if not rows:
                return False
            grid = []
            expected_width = None
            for row in rows:
                if not row.strip():
                    return False
                parts = row.strip().split()
                if len(parts) > 1:
                    try:
                        grid_row = [int(p) for p in parts if p.strip()]
                    except ValueError:
                        return False
                else:
                    if not row.strip().isdigit():
                        return False
                    grid_row = [int(char) for char in row.strip()]
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return False
                if expected_width is None:
                    expected_width = len(grid_row)
                elif len(grid_row) != expected_width:
                    return False
                grid.append(grid_row)
            return len(grid) > 0 and len(grid[0]) > 0
        except (ValueError, IndexError):
            return False
    return False


def parse_grid_from_string(output_string: str) -> Optional[List[List[int]]]:
    if not output_string or not isinstance(output_string, str):
        return None
    response = output_string.strip()
    if not response:
        return None
    if '\n' in response:
        grid_match = re.search(r'[0-9\n\s]+', response)
        if not grid_match:
            return None
        grid_str = grid_match.group()
        try:
            rows = grid_str.split('\n')
            grid = []
            for row in rows:
                if not row.strip():
                    continue
                parts = row.strip().split()
                if len(parts) > 1:
                    try:
                        grid_row = [int(p) for p in parts if p.strip()]
                    except ValueError:
                        return None
                else:
                    if not row.strip().isdigit():
                        return None
                    grid_row = [int(char) for char in row.strip()]
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return None
                grid.append(grid_row)
            return grid if grid else None
        except (ValueError, IndexError):
            return None
    return None


def check_value(output_string: str, expected_value: List[List[int]]) -> bool:
    if not isinstance(expected_value, list) or not expected_value:
        return False
    if not check_array(output_string):
        return False
    parsed_grid = parse_grid_from_string(output_string)
    if parsed_grid is None:
        return False
    return parsed_grid == expected_value


def same_shape(a: List[List[int]], b: List[List[int]]) -> bool:
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) for ra, rb in zip(a, b))


# ========== REWARD FUNCTIONS ==========

def reward_function_diff(
    completions: List[str],
    expected_output: List[str],
    **kwargs: Any
) -> List[float]:
    """
    Reward function based on cell-wise accuracy.
    Returns 1.0 for perfect match, proportional reward for partial match.
    """
    rewards: List[float] = []
    for completion, expected in zip(completions, expected_output, strict=False):
        value = completion[0]["content"] if isinstance(completion, list) else completion
        if not check_array(value):
            rewards.append(-1.0)
            continue
        comp_grid = parse_grid_from_string(value)
        exp_grid = parse_grid_from_string(expected)
        if comp_grid is None or exp_grid is None or not same_shape(comp_grid, exp_grid):
            rewards.append(-0.5)
            continue
        rows = len(exp_grid)
        cols = len(exp_grid[0]) if rows else 0
        diffs = 0
        for r in range(rows):
            for c in range(cols):
                if comp_grid[r][c] != exp_grid[r][c]:
                    diffs += 1
        if diffs != 0:
            rewards.append(0.5 * (1 - diffs / (rows * cols)))
        else:
            rewards.append(1.0)
    return rewards


# ========== DATA LOADING ==========

def load_data_by_level(data_path: str, level: int) -> List[Dict[str, Any]]:
    """Load training data filtered by level."""
    print(f"Loading data from {data_path} for level {level}...")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    
    filtered_data = [ex for ex in all_data if ex['metadata']['level'] == level]
    print(f"Found {len(filtered_data)} examples for level {level}")
    return filtered_data


def convert_to_rl_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert training data to RL format with prompt and expected_output."""
    result = []
    for ex in data:
        result.append({
            "prompt": [
                {"role": "user", "content": ex['problem']}
            ],
            "expected_output": ex['answer']
        })
    return result


def convert_to_sft_format(data: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    """Convert training data to SFT format (conversations)."""
    result = []
    for ex in data:
        result.append([
            {"role": "user", "content": ex['problem']},
            {"role": "assistant", "content": ex['answer']}
        ])
    return result


# ========== EVALUATION ==========

def evaluate_model(model, tokenizer, eval_data_path: str, max_samples: int = 100) -> Dict[str, float]:
    """Evaluate model on eval data."""
    print(f"\nEvaluating model on {eval_data_path}...")
    
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Sample some examples from different levels
    level_samples = {}
    for level in range(1, 7):
        level_data = [ex for ex in eval_data if ex['metadata']['level'] == level]
        if level_data:
            import random
            level_samples[level] = random.sample(level_data, min(max_samples // 6, len(level_data)))
    
    results = {f"level_{level}": {"correct": 0, "total": 0} for level in range(1, 7)}
    results["overall"] = {"correct": 0, "total": 0}
    
    FastLanguageModel.for_inference(model)
    
    for level, samples in level_samples.items():
        for sample in samples:
            messages = [{"role": "user", "content": sample['problem']}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            
            outputs = model.generate(input_ids=inputs, max_new_tokens=4096, use_cache=True)
            generated_tokens = outputs[:, inputs.shape[-1]:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            expected_grid = parse_grid_from_string(sample['answer'])
            is_correct = check_value(decoded, expected_grid)
            
            results[f"level_{level}"]["total"] += 1
            results["overall"]["total"] += 1
            if is_correct:
                results[f"level_{level}"]["correct"] += 1
                results["overall"]["correct"] += 1
    
    # Calculate accuracies
    for key in results:
        total = results[key]["total"]
        correct = results[key]["correct"]
        results[key]["accuracy"] = correct / total if total > 0 else 0.0
    
    print("\nEvaluation Results:")
    print(f"Overall: {results['overall']['correct']}/{results['overall']['total']} = {results['overall']['accuracy']:.2%}")
    for level in range(1, 7):
        level_key = f"level_{level}"
        if results[level_key]["total"] > 0:
            print(f"  Level {level}: {results[level_key]['correct']}/{results[level_key]['total']} = {results[level_key]['accuracy']:.2%}")
    
    return results


# ========== TRAINING FUNCTIONS ==========

def pick_attn_impl() -> str:
    if platform.system() == "Linux":
        try:
            import importlib
            importlib.import_module("flash_attn")
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"


def run_initial_sft(
    train_data: List[Dict[str, Any]],
    output_dir: str = "qwen3_4b_curriculum_sft",
    base_model: str = "unsloth/Qwen2.5-3B-Instruct",
    learning_rate: float = 8e-5,
    num_train_epochs: int = 300,
):
    """Run initial SFT on a single problem to bootstrap the model."""
    print(f"\n{'='*60}")
    print("Starting initial SFT training...")
    print(f"{'='*60}")
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=8192,
        dtype=compute_dtype,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Convert to SFT format
    conversations = convert_to_sft_format(train_data)
    data_text = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
    )
    
    import pandas as pd
    data_series = pd.Series(data_text)
    data_series.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(data_series))
    dataset = dataset.shuffle(seed=3407)
    
    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        max_grad_norm=None,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=8192,
    )
    
    print("[SFT] Starting training...")
    trainer.train()
    print("[SFT] Saving final adapter...")
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    try:
        tokenizer.save_pretrained(final_path)
    except Exception:
        pass
    
    return final_path


def run_rl_for_level(
    level: int,
    train_data: List[Dict[str, Any]],
    model_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    max_steps: int = 500,
):
    """Run RL training for a specific level."""
    print(f"\n{'='*60}")
    print(f"Starting RL training for Level {level}")
    print(f"Training samples: {len(train_data)}")
    print(f"{'='*60}")
    
    max_seq_length = 8192
    lora_rank = 128
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.2,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Convert to RL format
    converted = convert_to_rl_format(train_data)
    dataset = Dataset.from_list(converted)
    print(f"Dataset size: {len(dataset)}")
    
    vllm_sampling_params = SamplingParams(
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    training_args = GRPOConfig(
        use_vllm=True,
        importance_sampling_level="sequence",
        loss_type="grpo",
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        beta=0.04,
        epsilon=3e-4,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        report_to="none",
        num_generations=4,
        max_prompt_length=4096,
        max_completion_length=2048,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function_diff],
        args=training_args,
        train_dataset=dataset,
    )
    
    print(f"[RL Level {level}] Starting training...")
    trainer.train()
    
    final_path = os.path.join(output_dir, "final")
    print(f"[RL Level {level}] Saving model to {final_path}...")
    trainer.save_model(final_path)
    try:
        tokenizer.save_pretrained(final_path)
    except Exception:
        pass
    
    return final_path


# ========== MAIN CURRICULUM TRAINING ==========

def run_curriculum_training(
    train_data_path: str = "generated_data/train_data.json",
    eval_data_path: str = "generated_data/eval_data.json",
    base_output_dir: str = "qwen3_4b_curriculum",
    initial_sft_samples: int = 1000,
    rl_samples_per_level: int = 5000,
):
    """
    Run curriculum training: Initial SFT + Iterative RL by level (1-6).
    """
    print(f"\n{'='*80}")
    print("CURRICULUM TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Train data: {train_data_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}\n")
    
    # Load all training data
    print("Loading training data...")
    with open(train_data_path, 'r') as f:
        all_train_data = json.load(f)
    print(f"Total training examples: {len(all_train_data)}")
    
    # Step 1: Initial SFT on a small sample
    print("\n" + "="*80)
    print("STEP 1: Initial SFT Training")
    print("="*80)
    
    # Get level 1 samples for initial SFT
    level_1_data = [ex for ex in all_train_data if ex['metadata']['level'] == 1]
    import random
    random.seed(42)
    sft_data = random.sample(level_1_data, min(initial_sft_samples, len(level_1_data)))
    
    sft_output_dir = f"{base_output_dir}_sft"
    current_model_path = run_initial_sft(
        train_data=sft_data,
        output_dir=sft_output_dir,
        num_train_epochs=300,
    )
    
    # Evaluate after SFT
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=current_model_path,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    sft_results = evaluate_model(model, tokenizer, eval_data_path)
    
    # Save results
    with open(f"{base_output_dir}_results.json", 'w') as f:
        json.dump({"sft": sft_results}, f, indent=2)
    
    # Step 2: Iterative RL training by level (1-6, skipping 0)
    all_results = {"sft": sft_results}
    
    for level in range(1, 7):
        print(f"\n{'='*80}")
        print(f"STEP {level + 1}: RL Training for Level {level}")
        print(f"{'='*80}")
        
        # Load data for this level
        level_data = [ex for ex in all_train_data if ex['metadata']['level'] == level]
        
        # Sample data if too large
        if len(level_data) > rl_samples_per_level:
            random.seed(42 + level)
            level_data = random.sample(level_data, rl_samples_per_level)
        
        # Run RL training
        rl_output_dir = f"{base_output_dir}_rl_level{level}"
        current_model_path = run_rl_for_level(
            level=level,
            train_data=level_data,
            model_path=current_model_path,
            output_dir=rl_output_dir,
            max_steps=500,
        )
        
        # Evaluate after this level
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=current_model_path,
            max_seq_length=8192,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        level_results = evaluate_model(model, tokenizer, eval_data_path)
        all_results[f"rl_level_{level}"] = level_results
        
        # Save cumulative results
        with open(f"{base_output_dir}_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n[Level {level} Complete] Model saved to: {current_model_path}")
    
    print(f"\n{'='*80}")
    print("CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Final model: {current_model_path}")
    print(f"Results saved to: {base_output_dir}_results.json")
    print(f"{'='*80}\n")
    
    return current_model_path, all_results


if __name__ == "__main__":
    # Run the curriculum training pipeline
    final_model, results = run_curriculum_training(
        train_data_path="generated_data/train_data.json",
        eval_data_path="generated_data/eval_data.json",
        base_output_dir="qwen3_4b_curriculum",
        initial_sft_samples=1000,
        rl_samples_per_level=5000,
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    for stage, stage_results in results.items():
        print(f"\n{stage.upper()}:")
        print(f"  Overall Accuracy: {stage_results['overall']['accuracy']:.2%}")
