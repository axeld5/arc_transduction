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
from vllm import SamplingParams
from reward_functions import (
    reward_function_diff
)
from evaluate_model import evaluate_model_vllm

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

# ========== DATA LOADING ==========

def load_optimized_prompt(prompt_path: str = "optimized_prompt.txt") -> str:
    """Load the optimized system prompt from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {prompt_path} not found, proceeding without system prompt")
        return ""


def load_data_by_level(data_path: str, level: int) -> List[Dict[str, Any]]:
    """Load training data filtered by level."""
    print(f"Loading data from {data_path} for level {level}...")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    
    filtered_data = [ex for ex in all_data if ex['metadata']['level'] == level]
    print(f"Found {len(filtered_data)} examples for level {level}")
    return filtered_data


def convert_to_rl_format(
    data: List[Dict[str, Any]],
    use_system_prompt: bool = True,
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convert training data to RL format with prompt and expected_output.
    
    Args:
        data: List of training examples
        use_system_prompt: Whether to prepend the system prompt to each problem
        system_prompt: The system prompt to use (loaded from file if None)
        
    Returns:
        List of formatted examples for RL training
    """
    if use_system_prompt and system_prompt is None:
        system_prompt = load_optimized_prompt()
    
    result = []
    for ex in data:
        if use_system_prompt and system_prompt:
            # Prepend system prompt to the problem
            content = f"{system_prompt}\n\n{ex['problem']}"
        else:
            content = ex['problem']
        
        result.append({
            "prompt": [
                {"role": "user", "content": content}
            ],
            "expected_output": ex['answer']
        })
    return result


# ========== TRAINING FUNCTIONS ==========

def run_rl_for_level(
    level: int,
    train_data: List[Dict[str, Any]],
    model_path: str,
    output_dir: str,
    learning_rate: float = 1e-7,
    max_steps: int = 500,
    use_system_prompt: bool = True,
    use_dense_reward: bool = True,
    phase: str = "",
):
    """Run RL training for a specific level with specified reward type."""
    reward_type = "DENSE" if use_dense_reward else "DISCRETE"
    print(f"\n{'='*60}")
    print(f"Starting RL training for Level {level} {phase}")
    print(f"Training samples: {len(train_data)}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Reward type: {reward_type}")
    print(f"Using system prompt: {use_system_prompt}")
    print(f"{'='*60}")
    
    max_seq_length = 16000
    lora_rank = 128
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        max_lora_rank=lora_rank,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    model.get_input_embeddings().requires_grad_(True)
    
    # Convert to RL format with optional system prompt
    converted = convert_to_rl_format(train_data, use_system_prompt=use_system_prompt)
    dataset = Dataset.from_list(converted)
    print(f"Dataset size: {len(dataset)}")
    
    vllm_sampling_params = SamplingParams(
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    # Create reward function wrapper with appropriate dense/discrete setting
    def reward_func(completions, expected_output, **kwargs):
        """Wrapper for reward_function_diff with fixed use_dense_reward parameter."""
        return reward_function_diff(completions, expected_output, use_dense_reward=use_dense_reward, **kwargs)
    
    # Set the __name__ attribute so TRL can identify it
    reward_func.__name__ = f"reward_function_{'dense' if use_dense_reward else 'discrete'}"
    
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
        report_to="tensorboard",
        num_generations=4,
        max_prompt_length=8192,
        max_completion_length=4096,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    print(f"[RL Level {level} {phase}] Starting training...")
    trainer.train()
    
    final_path = os.path.join(output_dir, "final")
    merged_path = os.path.join(output_dir, "merged")
    
    # Save LoRA adapter
    print(f"[RL Level {level} {phase}] Saving LoRA adapter to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save merged 16bit model (required for next training stage)
    print(f"[RL Level {level} {phase}] Saving merged 16bit model to {merged_path}...")
    try:
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        print(f"[RL Level {level} {phase}] Successfully saved merged 16bit model")
    except Exception as e:
        print(f"[RL Level {level} {phase}] ERROR: Failed to save merged model: {e}")
        raise
    
    # Clean up trainer and vLLM communicator to free resources
    print(f"[RL Level {level} {phase}] Cleaning up trainer and vLLM resources...")
    try:
        # Close vLLM communicator if using vLLM
        if hasattr(trainer, 'vllm_client') and trainer.vllm_client is not None:
            trainer.vllm_client.close_communicator()
            print("  - Closed vLLM communicator")
    except Exception as e:
        print(f"  - Warning: Could not close vLLM communicator: {e}")
    
    # Delete trainer to free memory
    del trainer
    del model
    del tokenizer
    
    # Clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"[RL Level {level} {phase}] Cleanup complete")
    
    return merged_path


# ========== MAIN CURRICULUM TRAINING ==========

def run_curriculum_training(
    train_data_path: str = "generated_data/train_data.json",
    eval_data_path: str = "generated_data/eval_data.json",
    base_output_dir: str = "qwen3_4b_curriculum",
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    rl_samples_per_level: int = 5000,
    use_system_prompt: bool = True,
):
    """
    Run curriculum training: Direct RL training by level (1-6).
    
    Training is cumulative: Level N includes data from levels 1 through N.
    - Level 1: trains on level 1 data only
    - Level 2: trains on levels 1 + 2 data
    - Level 3: trains on levels 1 + 2 + 3 data
    - ... and so on
    
    Each level has two training phases:
      - Phase 1: Dense reward (125 steps) - gives partial credit
      - Phase 2: Discrete reward (375 steps) - binary success/fail
    
    Args:
        train_data_path: Path to training data JSON
        eval_data_path: Path to evaluation data JSON
        base_output_dir: Base directory for saving models
        base_model: Base model to start from
        rl_samples_per_level: Max samples to use (cumulative across all levels)
        use_system_prompt: Whether to prepend optimized prompt to each sample
    """
    print(f"\n{'='*80}")
    print("CURRICULUM RL TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Train data: {train_data_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Output directory: {base_output_dir}")
    print(f"Training strategy: 2 phases per level")
    print(f"  - Phase 1: Dense reward (125 steps)")
    print(f"  - Phase 2: Discrete reward (375 steps)")
    print(f"Use system prompt: {use_system_prompt}")
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
        
        # Load data for this level AND all previous levels (cumulative)
        level_data = [ex for ex in all_train_data if ex['metadata']['level'] <= level]
        
        print(f"Cumulative data: including levels 1-{level}")
        print(f"Total samples before sampling: {len(level_data)}")
        
        # Sample data if too large
        if len(level_data) > rl_samples_per_level:
            random.seed(42 + level)
            level_data = random.sample(level_data, rl_samples_per_level)
        
        print(f"Training on {len(level_data)} samples")
        
        # Phase 1: Dense reward training (1/4 of steps)
        dense_steps = 125  # 1/4 of 500
        dense_output_dir = f"{base_output_dir}_rl_level{level}_dense"
        current_model_path = run_rl_for_level(
            level=level,
            train_data=level_data,
            model_path=current_model_path,
            output_dir=dense_output_dir,
            max_steps=dense_steps,
            use_system_prompt=use_system_prompt,
            use_dense_reward=True,
            phase="(Phase 1: Dense)",
        )
        
        # Phase 2: Discrete reward training (3/4 of steps)
        discrete_steps = 375  # 3/4 of 500
        discrete_output_dir = f"{base_output_dir}_rl_level{level}_discrete"
        current_model_path = run_rl_for_level(
            level=level,
            train_data=level_data,
            model_path=current_model_path,
            output_dir=discrete_output_dir,
            max_steps=discrete_steps,
            use_system_prompt=use_system_prompt,
            use_dense_reward=False,
            phase="(Phase 2: Discrete)",
        )
        
        # Evaluate after this level using vLLM
        level_results = evaluate_model_vllm(
            model_path=current_model_path,
            eval_data_path=eval_data_path,
            max_samples_per_level=20,
            attempts_per_problem=1,
            temperature=0.7,
            use_system_prompt=use_system_prompt,
        )
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
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        rl_samples_per_level=5000,
        use_system_prompt=True,  # Set to False to disable system prompt
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    for stage, stage_results in results.items():
        print(f"\n{stage.upper()}:")
        print(f"  Overall Accuracy: {stage_results['overall']['accuracy']:.2%}")
