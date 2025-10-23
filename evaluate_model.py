"""
Model evaluation using vLLM for efficient batch generation.
Validates outputs using reward functions.
"""

import json
import random
import gc
from typing import Dict, Any, Optional
import torch


def load_optimized_prompt(prompt_path: str = "optimized_prompt.txt") -> str:
    """Load the optimized system prompt from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {prompt_path} not found, proceeding without system prompt")
        return ""


def evaluate_model_vllm(
    model_path: str,
    eval_data_path: str,
    max_samples_per_level: int = 20,
    attempts_per_problem: int = 1,
    temperature: float = 0.0,
    use_system_prompt: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model using vLLM for efficient batch generation.
    Uses reward functions to check validity of outputs.
    
    Args:
        model_path: Path to the model to evaluate
        eval_data_path: Path to evaluation data JSON file
        max_samples_per_level: Maximum samples to test per level (1-6)
        attempts_per_problem: Number of generation attempts per problem
        temperature: Sampling temperature (0.0 = greedy)
        use_system_prompt: Whether to prepend optimized prompt to each problem
        
    Returns:
        Dictionary with results per level and overall accuracy
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from reward_functions import parse_grid_from_string, check_value
    
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Using system prompt: {use_system_prompt}")
    print(f"{'='*60}")
    
    # Load system prompt if requested
    system_prompt = ""
    if use_system_prompt:
        system_prompt = load_optimized_prompt()
        if system_prompt:
            print(f"Loaded system prompt ({len(system_prompt)} characters)")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.5, max_model_len=16000)
    
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    
    # Load eval data
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Sample examples from different levels
    level_samples = {}
    for level in range(1, 7):
        level_data = [ex for ex in eval_data if ex['metadata']['level'] == level]
        if level_data:
            sampled = random.sample(level_data, min(max_samples_per_level, len(level_data)))
            level_samples[level] = sampled
    
    # Prepare ALL prompts upfront for batch generation
    all_prompts = []
    prompt_metadata = []  # Track which level/sample each prompt belongs to
    
    for level, samples in level_samples.items():
        for sample_idx, sample in enumerate(samples):
            # Prepend system prompt if requested
            if use_system_prompt and system_prompt:
                content = f"{system_prompt}\n\n{sample['problem']}"
            else:
                content = sample['problem']
            
            messages = [{"role": "user", "content": content}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Create multiple attempts per problem
            for attempt in range(attempts_per_problem):
                all_prompts.append(formatted_prompt)
                prompt_metadata.append({
                    'level': level,
                    'sample_idx': sample_idx,
                    'sample': sample,
                    'attempt': attempt
                })
    
    print(f"\nGenerating {len(all_prompts)} outputs in batch...")
    print(f"  {sum(len(s) for s in level_samples.values())} problems × {attempts_per_problem} attempts")
    
    # Generate ALL outputs in ONE batch
    outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    
    print("Batch generation complete! Evaluating results...\n")
    
    # Initialize results tracking
    results = {f"level_{level}": {"correct": 0, "total": 0} for level in range(1, 7)}
    results["overall"] = {"correct": 0, "total": 0}
    
    # Track which problems have been solved (for attempts_per_problem > 1)
    solved_problems = set()
    
    # Process outputs
    for i, (output, metadata) in enumerate(zip(outputs, prompt_metadata)):
        level = metadata['level']
        sample_idx = metadata['sample_idx']
        sample = metadata['sample']
        attempt = metadata['attempt']
        
        problem_key = (level, sample_idx)
        
        # Skip if already solved this problem
        if problem_key in solved_problems:
            continue
        
        # Only count each problem once (on first attempt)
        if attempt == 0:
            results[f"level_{level}"]["total"] += 1
            results["overall"]["total"] += 1
        
        # Extract generated text
        generated_text = output.outputs[0].text
        
        # Parse expected and generated grids
        expected_grid = parse_grid_from_string(sample['answer'])
        is_correct = check_value(generated_text, expected_grid)
        
        if is_correct:
            if problem_key not in solved_problems:
                results[f"level_{level}"]["correct"] += 1
                results["overall"]["correct"] += 1
                solved_problems.add(problem_key)
                print(f"✓ Level {level}, Problem {sample_idx} (attempt {attempt + 1}/{attempts_per_problem})")
        elif attempt == attempts_per_problem - 1:
            # Last attempt failed
            print(f"✗ Level {level}, Problem {sample_idx} (failed all {attempts_per_problem} attempts)")
    
    # Calculate accuracies
    for key in results:
        total = results[key]["total"]
        correct = results[key]["correct"]
        results[key]["accuracy"] = correct / total if total > 0 else 0.0
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall: {results['overall']['correct']}/{results['overall']['total']} = {results['overall']['accuracy']:.2%}")
    for level in range(1, 7):
        level_key = f"level_{level}"
        if results[level_key]["total"] > 0:
            print(f"  Level {level}: {results[level_key]['correct']}/{results[level_key]['total']} = {results[level_key]['accuracy']:.2%}")
    print(f"{'='*60}\n")
    
    # Cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model using vLLM")
    parser.add_argument("model_path", help="Path to the model to evaluate")
    parser.add_argument("--eval-data", default="generated_data/eval_data.json",
                        help="Path to evaluation data (default: generated_data/eval_data.json)")
    parser.add_argument("--samples-per-level", type=int, default=20,
                        help="Max samples per level (default: 20)")
    parser.add_argument("--attempts", type=int, default=1,
                        help="Attempts per problem (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Disable system prompt")
    
    args = parser.parse_args()
    
    results = evaluate_model_vllm(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        max_samples_per_level=args.samples_per_level,
        attempts_per_problem=args.attempts,
        temperature=args.temperature,
        use_system_prompt=not args.no_system_prompt,
    )
    
    # Save results
    output_file = f"{args.model_path.replace('/', '_')}_eval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

