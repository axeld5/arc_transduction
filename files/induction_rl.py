import json
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import unsloth
import torch
from typing import List, Any
from dotenv import load_dotenv
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL
from datasets import Dataset
from accelerate import Accelerator
accel = Accelerator(device_placement=False)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)                 # <- critical
device_map = {"": local_rank}                     # <- one GPU per rank

class PatchedGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure DDP-wrapped models expose .config
        try:
            _ = model.config  # will raise on DDP
        except AttributeError:
            base = self.accelerator.unwrap_model(model)
            # Attach the underlying config to the wrapper so downstream code works
            object.__setattr__(model, "config", getattr(base, "config", None))
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

def evaluate_code_validity(
    completions,
    arrays,
    is_partial_rl: bool = False,
    timeout_seconds: int = 10,
    **kwargs
):
    """
    Cross-platform, hard timeout using subprocess. Safely executes each completion's Python
    block in an isolated interpreter, so infinite loops get killed cleanly.

    Args:
        completions: list of model outputs (strings or {content: str} etc.)
        arrays: list of example lists; each example is a dict with keys "input" and "output"
        is_partial_rl: if True, reward = solved/total else +1.0 only if all solved (else -0.5)
        timeout_seconds: per-completion wall-clock timeout

    Returns:
        List[float]: rewards per completion
    """
    import json, os, sys, subprocess, textwrap

    def extract_text(c):
        if c is None: return ""
        if isinstance(c, str): return c
        if isinstance(c, dict): return c.get("content", "") or c.get("text", "")
        if isinstance(c, list):
            # handle [{"type":"text","text":"..."}, {"content":"..."}]
            for item in c:
                if isinstance(item, dict):
                    if "content" in item and item["content"]:
                        return item["content"]
                    if "text" in item and item["text"]:
                        return item["text"]
        return ""
    WORKER = textwrap.dedent(r"""
        import json, sys, os, io, contextlib, gc

        def main():
            payload = json.loads(sys.stdin.read())
            code = payload["code"]
            array_list = payload.get("array_list") or []
            is_partial = payload.get("is_partial", False)

            devnull = open(os.devnull, "w")

            local_namespace = {}
            try:
                # Silence user prints/errors
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    exec(code, local_namespace)
                fn = local_namespace.get("transform")
                if not callable(fn):
                    reward = -1.0
                else:
                    n_examples = len(array_list)
                    solved = 0
                    for ex in array_list:
                        try:
                            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                pred = fn(ex.get("input"))
                            if pred == ex.get("output"):
                                solved += 1
                        except Exception:
                            pass
                    if is_partial:
                        reward = (solved / n_examples) if n_examples > 0 else 0.0
                    else:
                        reward = 1.0 if (n_examples > 0 and solved == n_examples) else -0.5
            except Exception:
                reward = -1.0
            finally:
                try:
                    devnull.close()
                except Exception:
                    pass
                local_namespace.clear()
                gc.collect()

            sys.stdout.write(json.dumps({"reward": reward}))

        if __name__ == "__main__":
            main()
    """)
    start_marker, end_marker = "```python", "```"
    rewards = []
    for completion, array_list in zip(completions, arrays, strict=False):
        value = extract_text(completion)
        if not value:
            rewards.append(-1.0)
            continue
        start_idx = value.find(start_marker)
        if start_idx == -1:
            rewards.append(-1.0)
            continue
        start_idx += len(start_marker)
        end_idx = value.find(end_marker, start_idx)
        if end_idx == -1:
            rewards.append(-1.0)
            continue
        code = value[start_idx:end_idx].strip()
        payload = json.dumps({
            "code": code,
            "array_list": array_list or [],
            "is_partial": is_partial_rl,
        }).encode("utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-c", WORKER],
                input=payload,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            if proc.returncode != 0:
                rewards.append(-1.0)
                continue

            try:
                out = json.loads(proc.stdout.decode("utf-8") or "{}")
                rewards.append(float(out.get("reward", -1.0)))
            except Exception:
                rewards.append(-1.0)

        except subprocess.TimeoutExpired:
            rewards.append(-1.0)
        except Exception:
            rewards.append(-1.0)

    return rewards

def convert_conversations(raw_json):
    result = []
    for convo, array_list in zip(raw_json["conversations"], raw_json["arrays"]):
        user_msg = convo[0]["content"]
        result.append({
            "prompt": [
                {"role": "user", "content": user_msg}
            ],
            "arrays": array_list
        })
    return result

def run_rl(
    sft_merged_save_path: str,
    output_dir: str = "qwen2.5_7b_singled_out_rl",
    learning_rate: float = 5e-5,
    num_steps: int = 100,
    batch_size: int = 2,
    grad_accum: int = 2,
    num_generations: int = 2,
    data_dir: str = "data.json",
    is_partial: bool = False,
):
    max_seq_length = 15000
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = sft_merged_save_path,
        max_seq_length = max_seq_length,
        dtype = compute_dtype,
        load_in_4bit = False,        # <- be explicit
        fast_inference = False,      # <- ensure no embedded vLLM
        device_map = device_map,     # <- pin to the per-rank GPU
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 1,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth"
    )
    with open(data_dir) as f:
        raw = json.load(f)
    converted = convert_conversations(raw)
    dataset = Dataset.from_list(converted)
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )
    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="127.0.0.1",
        vllm_server_port=8000,
        #importance_sampling_level="sequence",
        #loss_type="grpo",
        vllm_sampling_params=vllm_sampling_params,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,  # Reduced from 8 to help with memory
        gradient_accumulation_steps=grad_accum,
        beta=0.04,
        epsilon=3e-4,
        max_steps=num_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        report_to="none",
        num_generations=num_generations,
        max_prompt_length=10000,  # Reduced from 20000
        max_completion_length=2048,  # Reduced from 8192 (total ~18k < 20k model limit)
        remove_unused_columns=False,
        ddp_find_unused_parameters=False
    )
    
    # Create reward function with is_partial_rl parameter
    def reward_func_with_partial(completions, arrays, **kwargs):
        return evaluate_code_validity(completions, arrays, is_partial_rl=is_partial, **kwargs)
    
    trainer = PatchedGRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [reward_func_with_partial],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    model_save_path = os.path.join(output_dir, "final")
    merged_save_path = os.path.join(output_dir, "merged")
    model.save_pretrained(model_save_path)
    try:
        tokenizer.save_pretrained(model_save_path)
        model.save_pretrained_merged(merged_save_path, tokenizer, save_method = "merged_16bit",)
        if os.getenv("HF_TOKEN") and not is_partial:
            model.push_to_hub_merged("axel-darmouni/qwen2.5-7b-soar-induction-rl", tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))
    except Exception:
        pass
    return model_save_path


if __name__ == "__main__":    
    #sft_merged_save_path = "qwen2.5_7b_singled_out_sft/merged"
    sft_merged_save_path = "julien31/Soar-qwen-7b"
    print("=" * 80)
    print("Stage 1: Training with PARTIAL reward function")
    print("=" * 80)
    
    """# First stage: Train with partial reward (proportional to examples solved)
    stage1_path = run_rl(
        sft_merged_save_path=sft_merged_save_path,
        output_dir="qwen2.5_7b_induction_rl_partial",
        learning_rate=5e-5,
        num_steps=400,
        batch_size=2,
        grad_accum=2,
        num_generations=2,
        data_dir="full_set.json",
        is_partial=True,
    )
    """
    print("\n" + "=" * 80)
    print("Stage 2: Training with FULL reward function (binary)")
    print("=" * 80)
    stage1_path = "qwen2.5_7b_induction_rl_full/merged"
    
    # Second stage: Train with full binary reward on the model from stage 1
    stage2_path = run_rl(
        sft_merged_save_path=stage1_path,
        output_dir="qwen2.5_7b_induction_rl_full",
        learning_rate=5e-5,
        num_steps=2000,
        batch_size=2,
        grad_accum=4,
        num_generations=2,
        data_dir="test_problems.json",
        is_partial=False,
    )
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Stage 1 (partial) model saved at: {stage1_path}")
    print(f"Stage 2 (full) model saved at: {stage2_path}")
    print("=" * 80)
