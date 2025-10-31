import json
import pandas as pd
import os
from typing import List, Dict, Any, Optional
from unsloth.trainer import UnslothVisionDataCollator
from unsloth.chat_templates import get_chat_template


os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_LOSS"] = "1"
os.environ["UNSLOTH_DISABLE_LOSS_PATCHING"] = "1"

import unsloth
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from accelerate import Accelerator

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

def load_all_training_data(
    data_path: str,
    use_system_prompt: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load all training data from JSON file and convert to GPT-OSS conversation format.
    
    Args:
        data_path: Path to the training data JSON file
        use_system_prompt: Whether to prepend the system prompt
        
    Returns:
        List of conversation dictionaries ready for SFT with GPT-OSS format
    """
    print(f"Loading training data from {data_path}...")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    print(f"Total training examples: {len(all_data)}")
    conversations = []
    for ex in all_data:
        conversation = [
            {"role": "user", "content": ex['problem'],},
            {"role": "assistant", "content": ex['answer']},
        ]        
        conversations.append({
            "messages": conversation
        })
    print(f"Converted {len(conversations)} examples to Qwen3 VL conversation format")    
    return conversations

def config_data_for_sft(conversations: List[Dict[str, Any]], tokenizer):
    """
    Format conversation data for SFT training with Qwen3 VL.
    
    Args:
        conversations: List of conversation dictionaries with messages and reasoning_effort
        tokenizer: The tokenizer to use for formatting
        
    Returns:
        Dataset ready for SFT training
    """
    formatted_texts = []
    for conv in conversations:
        text = tokenizer.apply_chat_template(
            conv["messages"],
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix('<bos>')
        formatted_texts.append(text)
    formatted_data = pd.Series(formatted_texts)
    formatted_data.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    return dataset


def run_sft(
    train_data_path: str = "generated_data/train_data.json",
    eval_data_path: str = "generated_data/eval_data.json",
    output_dir: str = "qwen3_8b_conceptarc_sft",
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_seq_length: int = 8192,
    lora_rank: int = 256,
):
    """
    Run SFT training on GPT-OSS model for transduction data.
    
    Args:
        train_data_path: Path to training data JSON
        eval_data_path: Path to evaluation data JSON
        output_dir: Directory to save the trained model
        base_model: Base GPT-OSS model to fine-tune
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank (smaller for GPT-OSS due to model size)
    """    
    # Load and prepare data
    conversations = load_all_training_data(train_data_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map=device_map,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=64,
        bias="none",
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
        #               "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        use_gradient_checkpointing="unsloth",
    )
    
    # Prepare dataset
    print("\nFormatting dataset for GPT-OSS...")
    dataset = config_data_for_sft(conversations, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Print a sample for verification
    if local_rank == 0:
        print("\n" + "="*80)
        print("SAMPLE FORMATTED TEXT (first 500 chars):")
        print("="*80)
        print(dataset[0]['text'])
        print("="*80 + "\n")
    
    # Training arguments
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_steps=5,
        lr_scheduler_type="linear",
        fp16=False,  # GPT-OSS uses float32
        bf16=False,
        logging_steps=10,
        max_steps=100,
        save_total_limit=2,
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="adamw_8bit",
        weight_decay=0.001,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
    )
    
    trainer = UnslothFixedTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    )
    
    print("\n[SFT] Starting training...")
    trainer.train()
    
    print("\n[SFT] Saving final adapter...")
    model_save_path = os.path.join(output_dir, "final")
    merged_save_path = os.path.join(output_dir, "merged")
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save merged model (on main process only)
    if False: #local_rank == 0:
        print("[SFT] Saving merged model...")
        try:
            # Save as merged 16bit
            model.save_pretrained_merged(
                merged_save_path,
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"[SFT] Successfully saved merged model to {merged_save_path}")
            
            # Optional: Also save as mxfp4 for efficient inference
            mxfp4_save_path = os.path.join(output_dir, "merged_mxfp4")
            try:
                model.save_pretrained_merged(
                    mxfp4_save_path,
                    tokenizer,
                    save_method="mxfp4",
                )
                print(f"[SFT] Successfully saved mxfp4 model to {mxfp4_save_path}")
            except Exception as e:
                print(f"[SFT] Warning: Could not save mxfp4 model: {e}")
            
            # Optional: Push to hub if token is available
            if os.getenv("HF_TOKEN"):
                hub_name = f"axel-darmouni/{output_dir.replace('/', '-')}"
                print(f"[SFT] Pushing to hub: {hub_name}")
                try:
                    model.push_to_hub_merged(
                        hub_name,
                        tokenizer,
                        save_method="merged_16bit",
                        token=os.getenv("HF_TOKEN")
                    )
                    print(f"[SFT] Successfully pushed to hub: {hub_name}")
                except Exception as e:
                    print(f"[SFT] Warning: Could not push to hub: {e}")
        except Exception as e:
            print(f"[SFT] Warning: Could not save merged model: {e}")
    
    print(f"\n{'='*80}")
    print("GPT-OSS SFT TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Adapter saved to: {model_save_path}")
    if local_rank == 0:
        print(f"Merged model saved to: {merged_save_path}")
    print(f"{'='*80}\n")
    
    return model_save_path, merged_save_path


if __name__ == "__main__":
    sft_model_save_path, sft_merged_save_path, eval_results = run_sft(
        train_data_path="generated_data/train_conceptarc_data.json",
        eval_data_path="generated_data/eval_conceptarc_data.json",
        output_dir="gemma_3_27b_conceptarc_sft",
        base_model="unsloth/gemma-3-27b-it",
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=8192,
        lora_rank=32,
    )
    
    print(f"\nGPT-OSS SFT adapter saved to: {sft_model_save_path}")
    if sft_merged_save_path:
        print(f"GPT-OSS SFT merged model saved to: {sft_merged_save_path}")