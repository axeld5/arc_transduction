import json
import pandas as pd
import os
from typing import List, Dict, Any, Optional

os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_LOSS"] = "1"
os.environ["UNSLOTH_DISABLE_LOSS_PATCHING"] = "1"

import unsloth
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
from accelerate import Accelerator
from evaluate_model import evaluate_model_vllm

accel = Accelerator(device_placement=False)

local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)

device_map = {"": accel.local_process_index}  # one GPU per rank

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass


class UnslothFixedTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, '_get_name') and 'unsloth' in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, 'clone'):
              loss = loss.clone()  # Converts view tensor to independent tensor
        if self.accelerator.num_processes > 1:
              loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


def load_optimized_prompt(prompt_path: str = "optimized_prompt.txt") -> str:
    """Load the optimized system prompt from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {prompt_path} not found, proceeding without system prompt")
        return ""


def load_all_training_data(
    data_path: str,
    use_system_prompt: bool = True,
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load all training data from JSON file and convert to conversation format.
    
    Args:
        data_path: Path to the training data JSON file
        use_system_prompt: Whether to prepend the system prompt
        system_prompt: The system prompt to use (loaded from file if None)
        
    Returns:
        List of conversation dictionaries ready for SFT
    """
    print(f"Loading training data from {data_path}...")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    
    print(f"Total training examples: {len(all_data)}")
    
    # Load system prompt if needed
    if use_system_prompt and system_prompt is None:
        system_prompt = load_optimized_prompt()
    
    # Convert to conversation format
    conversations = []
    for ex in all_data:
        if use_system_prompt and system_prompt:
            # Prepend system prompt to the problem
            user_content = f"{system_prompt}\n\n{ex['problem']}"
        else:
            user_content = ex['problem']
        
        conversations.append([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ex['answer']}
        ])
    
    print(f"Converted {len(conversations)} examples to conversation format")
    
    # Count examples by level for reporting
    level_counts = {}
    for ex in all_data:
        level = ex['metadata']['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("\nData distribution by level:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} examples")
    
    return conversations


def config_data_for_sft(conversations: List[List[Dict[str, str]]], tokenizer):
    """
    Format conversation data for SFT training.
    
    Args:
        conversations: List of conversation lists
        tokenizer: The tokenizer to use for formatting
        
    Returns:
        Dataset ready for SFT training
    """
    formatted_data = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
    )
    formatted_data = pd.Series(formatted_data)
    formatted_data.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    return dataset


def run_sft(
    train_data_path: str = "generated_data/train_data.json",
    eval_data_path: str = "generated_data/eval_data.json",
    output_dir: str = "qwen2.5_3b_transduction_sft",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    use_system_prompt: bool = True,
    max_seq_length: int = 20000,
    lora_rank: int = 256,
    eval_after_training: bool = True,
    eval_samples_per_level: int = 20,
    eval_temperature: float = 0.7,
):
    """
    Run SFT training on all levels of transduction data.
    
    Args:
        train_data_path: Path to training data JSON
        eval_data_path: Path to evaluation data JSON
        output_dir: Directory to save the trained model
        base_model: Base model to fine-tune
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        use_system_prompt: Whether to use the optimized system prompt
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank
        eval_after_training: Whether to run evaluation after training
        eval_samples_per_level: Max samples per level for evaluation
        eval_temperature: Sampling temperature for evaluation
    """
    print(f"\n{'='*80}")
    print("TRANSDUCTION SFT TRAINING (MULTI-GPU)")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Train data: {train_data_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Batch size per device: {per_device_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Use system prompt: {use_system_prompt}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Run evaluation: {eval_after_training}")
    print(f"{'='*80}\n")
    
    # Load and prepare data
    conversations = load_all_training_data(
        train_data_path,
        use_system_prompt=use_system_prompt
    )
    
    # Setup model
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    print(f"\nLoading model: {base_model}")
    print(f"Compute dtype: {compute_dtype}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=compute_dtype,
        load_in_4bit=False,
        device_map=device_map,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    )
    
    # Prepare dataset
    print("\nFormatting dataset...")
    dataset = config_data_for_sft(conversations, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Training arguments
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #num_train_epochs=num_train_epochs,
        max_steps=1000,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=100,
        save_steps=200,
        save_total_limit=2,
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        max_grad_norm=None,
    )
    
    trainer = UnslothFixedTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("\n[SFT] Starting training...")
    trainer.train()
    
    print("\n[SFT] Saving final adapter...")
    model_save_path = os.path.join(output_dir, "final")
    merged_save_path = os.path.join(output_dir, "merged")
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print("[SFT] Saving merged 16bit model...")
    try:
        model.save_pretrained_merged(
            merged_save_path,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"[SFT] Successfully saved merged model to {merged_save_path}")
        
        # Optional: Push to hub if token is available
        if os.getenv("HF_TOKEN"):
            hub_name = f"axel-darmouni/{output_dir.replace('/', '-')}"
            print(f"[SFT] Pushing to hub: {hub_name}")
            model.push_to_hub_merged(
                hub_name,
                tokenizer,
                save_method="merged_16bit",
                token=os.getenv("HF_TOKEN")
            )
    except Exception as e:
        print(f"[SFT] Warning: Could not save/push merged model: {e}")
    
    # Run evaluation if requested and on main process
    eval_results = None
    if eval_after_training and local_rank == 0:
        print("\n[SFT] Starting evaluation with LoRA adapter...")
        try:
            eval_results = evaluate_model_vllm(
                model_path=base_model,
                eval_data_path=eval_data_path,
                max_samples_per_level=eval_samples_per_level,
                attempts_per_problem=1,
                temperature=eval_temperature,
                use_system_prompt=use_system_prompt,
                use_lora=True,
                lora_path=model_save_path,
                print_examples=True,
            )
            
            # Save evaluation results
            results_file = os.path.join(output_dir, "eval_results.json")
            with open(results_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"[SFT] Evaluation results saved to: {results_file}")
            
        except Exception as e:
            print(f"[SFT] Warning: Evaluation failed: {e}")
    
    print(f"\n{'='*80}")
    print("SFT TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Adapter saved to: {model_save_path}")
    print(f"Merged model saved to: {merged_save_path}")
    if eval_results:
        print(f"Overall Accuracy: {eval_results['overall']['accuracy']:.2%}")
    print(f"{'='*80}\n")
    
    return model_save_path, merged_save_path, eval_results


if __name__ == "__main__":
    sft_model_save_path, sft_merged_save_path, eval_results = run_sft(
        train_data_path="generated_data/train_data.json",
        eval_data_path="generated_data/eval_data.json",
        output_dir="qwen2.5_3b_transduction_sft",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        use_system_prompt=True,
        max_seq_length=20000,
        lora_rank=256,
        eval_after_training=True,
        eval_samples_per_level=20,
        eval_temperature=0.7,
    )
    
    print(f"\nSFT adapter saved to: {sft_model_save_path}")
    print(f"SFT merged model saved to: {sft_merged_save_path}")
    if eval_results:
        print(f"\nEVALUATION SUMMARY:")
        print(f"  Overall: {eval_results['overall']['accuracy']:.2%}")
        for level in range(1, 7):
            level_key = f"level_{level}"
            if level_key in eval_results and eval_results[level_key]["total"] > 0:
                print(f"  Level {level}: {eval_results[level_key]['accuracy']:.2%}")

