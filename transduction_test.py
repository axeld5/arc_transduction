import json
from typing import *
import os
import platform
import random
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
from vllm import SamplingParams
from reward_functions import (
    parse_grid_from_string,
    check_value,
    DynamicRewardFunction
)

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

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

def run_rl_for_level(
    level: int,
    train_data: List[Dict[str, Any]],
    model_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    max_steps: int = 500,
):
    """Run RL training for a specific level with dynamic reward (dense -> discrete)."""
    print(f"\n{'='*60}")
    print(f"Starting RL training for Level {level}")
    print(f"Training samples: {len(train_data)}")
    print(f"Max steps: {max_steps}")
    print(f"Dense reward phase: steps 0-{max_steps // 4} (1/4)")
    print(f"Discrete reward phase: steps {max_steps // 4 + 1}-{max_steps} (3/4)")
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
    
    # Create dynamic reward function that switches from dense to discrete
    dynamic_reward = DynamicRewardFunction(max_steps=max_steps)
    
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
        reward_funcs=[dynamic_reward],
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
    base_model: str = "unsloth/Qwen2.5-3B-Instruct",
    rl_samples_per_level: int = 5000,
):
    """
    Run curriculum training: Direct RL training by level (1-6).
    Uses dynamic reward: dense for first 1/4 of steps, discrete for remaining 3/4.
    """
    print(f"\n{'='*80}")
    print("CURRICULUM RL TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Train data: {train_data_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Output directory: {base_output_dir}")
    print(f"Reward strategy: Dense (1/4) -> Discrete (3/4)")
    print(f"{'='*80}\n")
    
    # Load all training data
    print("Loading training data...")
    with open(train_data_path, 'r') as f:
        all_train_data = json.load(f)
    print(f"Total training examples: {len(all_train_data)}")
    
    # Start with base model (no SFT step)
    current_model_path = base_model
    
    # Iterative RL training by level (1-6)
    all_results = {}
    
    for level in range(1, 7):
        print(f"\n{'='*80}")
        print(f"STEP {level}: RL Training for Level {level}")
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
    # Run the curriculum training pipeline (no SFT, direct RL with dynamic rewards)
    final_model, results = run_curriculum_training(
        train_data_path="generated_data/train_data.json",
        eval_data_path="generated_data/eval_data.json",
        base_output_dir="qwen3_4b_curriculum",
        base_model="unsloth/Qwen2.5-3B-Instruct",
        rl_samples_per_level=5000,
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    for stage, stage_results in results.items():
        print(f"\n{stage.upper()}:")
        print(f"  Overall Accuracy: {stage_results['overall']['accuracy']:.2%}")
