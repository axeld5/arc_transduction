import json
import pandas as pd
import os

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

def config_data_for_sft(dataset_path: str, tokenizer):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    formatted_data = tokenizer.apply_chat_template(
        data["conversations"],
        tokenize = False,
    )
    formatted_data = pd.Series(formatted_data)
    formatted_data.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    return dataset

def run_sft(
    dataset_path: str,
    output_dir: str = "qwen2.5_7b_singled_out_sft",
    base_model: str = "julien31/Soar-qwen-7b",
    learning_rate: float = 5e-5,
    num_train_epochs: int = 10,
):      
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,                 # use the arg instead of hardcoding
        max_seq_length = 20000,
        dtype = compute_dtype,
        load_in_4bit = False,
        device_map = device_map,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 256,
        lora_alpha = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    )
    dataset = config_data_for_sft(dataset_path, tokenizer)
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
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
    print("[sft] Starting training...")
    trainer.train()
    print("[sft] Saving final adapter...")
    model_save_path = os.path.join(output_dir, "final")
    merged_save_path = os.path.join(output_dir, "merged")
    model.save_pretrained(model_save_path)
    try:
        tokenizer.save_pretrained(model_save_path)
        model.save_pretrained_merged(merged_save_path, tokenizer, save_method = "merged_16bit",)
        if os.getenv("HF_TOKEN"):
            model.push_to_hub_merged("axel-darmouni/qwen2.5-7b-soar-induction-sft", tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))
    except Exception:
        pass    
    return model_save_path, merged_save_path


if __name__ == "__main__":
    sft_model_save_path, sft_merged_save_path = run_sft("data.json")
    print(f"SFT model saved to: {sft_model_save_path}")
    print(f"SFT merged model saved to: {sft_merged_save_path}")

