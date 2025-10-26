"""
Model evaluation using vLLM for efficient batch generation.
Validates outputs using reward functions.

Supports three inference modes:
1. Standard: Direct batch generation (default)
2. Deep Dive: Batched iterative refinement (all problems batched at each iteration)
3. Augmented Voting: Apply augmentations, infer, reverse, and vote (efficient batching)

All methods leverage vLLM's batching capabilities for maximum efficiency.
See INFERENCE_METHODS.md for detailed documentation.
"""

import json
import random
import gc
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import Counter
import torch
from augment import (
    rotate_90, rotate_180, rotate_270,
    flip_vertical, flip_horizontal, double_flip,
    apply_color_permutation, assign_random_augmentations,
    apply_augmentations_to_grids
)


def load_optimized_prompt(prompt_path: str = "optimized_prompt.txt") -> str:
    """Load the optimized system prompt from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {prompt_path} not found, proceeding without system prompt")
        return ""


def get_reverse_augmentation(aug_func: Callable) -> Callable:
    """Get the reverse transformation for a given augmentation function."""
    # Rotation reverses
    if aug_func == rotate_90:
        return rotate_270
    elif aug_func == rotate_270:
        return rotate_90
    elif aug_func == rotate_180:
        return rotate_180  # 180 is its own inverse
    # Flip reverses (all are their own inverse)
    elif aug_func in [flip_vertical, flip_horizontal, double_flip]:
        return aug_func
    # Color permutation needs special handling
    elif aug_func == apply_color_permutation:
        return apply_color_permutation
    else:
        raise ValueError(f"Unknown augmentation function: {aug_func}")


def reverse_augmentations_on_grid(grid: List[List[int]], 
                                  augmentations: List[Tuple[Callable, Dict]]) -> List[List[int]]:
    """
    Reverse a sequence of augmentations on a grid.
    
    Args:
        grid: The augmented grid
        augmentations: List of (augmentation_func, kwargs) tuples applied in forward order
    
    Returns:
        Original grid with augmentations reversed
    """
    result = grid
    # Reverse in opposite order
    for aug_func, kwargs in reversed(augmentations):
        if aug_func == apply_color_permutation:
            # Reverse the color map
            color_map = kwargs.get('color_map', {})
            reverse_map = {v: k for k, v in color_map.items()}
            result = apply_color_permutation(result, reverse_map)
        else:
            reverse_func = get_reverse_augmentation(aug_func)
            result = reverse_func(result)
    
    return result


def grid_to_string(grid: List[List[int]]) -> str:
    """Convert a grid to string format for prompting."""
    return '\n'.join([' '.join(map(str, row)) for row in grid])


def deep_dive_inference_batched(
    llm,
    tokenizer,
    samples: List[Dict[str, Any]],
    system_prompt: str,
    sampling_params,
    lora_request=None,
    max_iterations: int = 16,
) -> List[str]:
    """
    Batched iterative deep dive inference: feed model outputs back as inputs with placeholders.
    Processes all samples together at each iteration for efficiency.
    
    Args:
        llm: vLLM model instance
        tokenizer: Tokenizer instance
        samples: List of problem samples with 'problem' field
        system_prompt: System prompt to prepend
        sampling_params: Sampling parameters for generation
        lora_request: LoRA request if using LoRA
        max_iterations: Maximum number of iterations (default: 16)
    
    Returns:
        List of final generated answers (one per sample)
    """
    from reward_functions import parse_grid_from_string
    
    # Track state for each sample
    num_samples = len(samples)
    current_problems = [sample['problem'] for sample in samples]
    final_outputs = [None] * num_samples
    active_indices = list(range(num_samples))  # Samples still being processed
    
    for iteration in range(max_iterations):
        if not active_indices:
            break  # All samples have finished
        
        # Prepare prompts for active samples
        prompts = []
        for idx in active_indices:
            problem = current_problems[idx]
            if system_prompt:
                content = f"{system_prompt}\n\n{problem}"
            else:
                content = problem
            
            messages = [{"role": "user", "content": content}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(formatted_prompt)
        
        # Batch generate for all active samples
        if lora_request:
            outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling_params=sampling_params)
        
        # Process outputs and prepare next iteration
        next_active_indices = []
        for i, idx in enumerate(active_indices):
            generated_text = outputs[i].outputs[0].text
            
            # Try to parse the generated grid
            generated_grid = parse_grid_from_string(generated_text)
            
            # If this is the last iteration or parsing failed, finalize this sample
            if iteration == max_iterations - 1 or generated_grid is None:
                final_outputs[idx] = generated_text
                continue
            
            # Update problem with new output for next iteration
            if "Test Case:" in current_problems[idx]:
                # Format: ...Test Case:\nInput:\n[grid]\nOutput Placeholder:...\nOutput:
                # Split at "Test Case:" and keep everything before it, then reconstruct with new output
                before_test = current_problems[idx].split("Test Case:")[0]
                test_part = "Test Case:" + current_problems[idx].split("Test Case:")[1]
                # Find where "Output:" appears (the final one where we generate)
                if "Output:" in test_part:
                    test_section = test_part.split("Output:")[0]  # Everything up to final "Output:"
                    current_problems[idx] = before_test + test_section + f"Output:\n{grid_to_string(generated_grid)}"
                else:
                    # Fallback
                    current_problems[idx] = current_problems[idx].rsplit('Output:', 1)[0] + f"Output:\n{grid_to_string(generated_grid)}"
            elif "Test Input:" in current_problems[idx]:
                # Alternative format: Test Input:
                before_test = current_problems[idx].split("Test Input:")[0]
                test_section = "Test Input:" + current_problems[idx].split("Test Input:")[1].split("Output:")[0]
                current_problems[idx] = before_test + test_section + f"Output:\n{grid_to_string(generated_grid)}"
            else:
                # Fallback: just append to the problem
                current_problems[idx] = current_problems[idx].rsplit('Output:', 1)[0] + f"Output:\n{grid_to_string(generated_grid)}"
            
            # Keep this sample active for next iteration
            next_active_indices.append(idx)
        
        active_indices = next_active_indices
        print(f"    Iteration {iteration + 1}/{max_iterations}: {len(active_indices)} samples still refining")
    
    # Fill in any remaining samples (shouldn't happen, but just in case)
    for idx in range(num_samples):
        if final_outputs[idx] is None:
            final_outputs[idx] = ""
    
    return final_outputs


def augmented_voting_inference_batched(
    llm,
    tokenizer,
    samples: List[Dict[str, Any]],
    system_prompt: str,
    sampling_params,
    lora_request=None,
    num_augmentations: int = 30,
) -> List[str]:
    """
    Batched augmented voting inference: apply augmentations to all problems, infer in one batch, reverse, and vote.
    Processes ALL problems together for maximum vLLM efficiency.
    
    Args:
        llm: vLLM model instance
        tokenizer: Tokenizer instance
        samples: List of problem samples with 'problem' field
        system_prompt: System prompt to prepend
        sampling_params: Sampling parameters for generation
        lora_request: LoRA request if using LoRA
        num_augmentations: Number of augmented versions to create (default: 30)
    
    Returns:
        List of majority vote answers (one per sample)
    """
    from reward_functions import parse_grid_from_string
    import re
    
    num_samples = len(samples)
    
    # Parse all problems upfront
    parsed_problems = []
    fallback_indices = []  # Samples that need standard inference
    
    for sample_idx, sample in enumerate(samples):
        problem_text = sample['problem']
        
        # Extract input/output examples
        examples = []
        example_pattern = r'Example \d+:\s*Input:\s*(.*?)\s*Output:\s*(.*?)(?=Example \d+:|Test Case:|$)'
        matches = re.findall(example_pattern, problem_text, re.DOTALL)
        
        for input_section, output_section in matches:
            input_grid = parse_grid_from_string(input_section.strip())
            output_grid = parse_grid_from_string(output_section.strip())
            
            if input_grid and output_grid:
                examples.append((input_grid, output_grid))
        
        # Extract test input
        test_input_match = re.search(r'Test Case:\s*Input:\s*(.*?)(?:\s*Output Placeholder:|$)', problem_text, re.DOTALL)
        if test_input_match:
            test_input_section = test_input_match.group(1).strip()
            test_input_grid = parse_grid_from_string(test_input_section)
        else:
            test_input_grid = None
        
        if not test_input_grid or not examples:
            # Mark for fallback
            fallback_indices.append(sample_idx)
            parsed_problems.append(None)
        else:
            # Store parsed data
            intro = ""
            if "Training Examples:" in problem_text:
                intro = problem_text.split("Training Examples:")[0].strip() + "\n\nTraining Examples:\n"
            elif "Example 1:" in problem_text:
                intro = problem_text.split("Example 1:")[0].strip() + "\n\n"
            
            parsed_problems.append({
                'examples': examples,
                'test_input': test_input_grid,
                'intro': intro
            })
    
    # Handle fallback cases with standard inference
    fallback_results = {}
    if fallback_indices:
        print(f"    {len(fallback_indices)} problems need fallback to standard inference")
        fallback_prompts = []
        for idx in fallback_indices:
            problem_text = samples[idx]['problem']
            content = f"{system_prompt}\n\n{problem_text}" if system_prompt else problem_text
            messages = [{"role": "user", "content": content}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            fallback_prompts.append(formatted_prompt)
        
        if lora_request:
            fallback_outputs = llm.generate(fallback_prompts, sampling_params=sampling_params, lora_request=lora_request)
        else:
            fallback_outputs = llm.generate(fallback_prompts, sampling_params=sampling_params)
        
        for i, idx in enumerate(fallback_indices):
            fallback_results[idx] = fallback_outputs[i].outputs[0].text
    
    # Generate augmentations (same for all problems)
    augmentation_configs = []
    for i in range(num_augmentations):
        aug_list = assign_random_augmentations(seed=i)
        
        aug_with_kwargs = []
        for aug_func in aug_list:
            if aug_func == apply_color_permutation:
                random.seed(i)
                colors = list(range(10))
                shuffled_colors = colors.copy()
                while True:
                    random.shuffle(shuffled_colors)
                    if all(original != shuffled for original, shuffled in zip(colors, shuffled_colors)):
                        break
                color_map = dict(zip(colors, shuffled_colors))
                aug_with_kwargs.append((aug_func, {'color_map': color_map}))
            else:
                aug_with_kwargs.append((aug_func, {}))
        
        augmentation_configs.append(aug_with_kwargs)
    
    # Create ALL augmented problems for ALL samples in ONE giant batch
    all_prompts = []
    prompt_metadata = []  # Track which sample/augmentation each prompt belongs to
    
    for sample_idx, parsed in enumerate(parsed_problems):
        if parsed is None:
            # Skip fallback samples
            continue
        
        examples = parsed['examples']
        test_input_grid = parsed['test_input']
        intro = parsed['intro']
        
        for aug_idx, aug_config in enumerate(augmentation_configs):
            # Apply augmentations to all grids
            augmented_examples = []
            for input_grid, output_grid in examples:
                aug_input = input_grid
                aug_output = output_grid
                for aug_func, kwargs in aug_config:
                    aug_input = aug_func(aug_input, **kwargs)
                    aug_output = aug_func(aug_output, **kwargs)
                augmented_examples.append((aug_input, aug_output))
            
            # Apply to test input
            aug_test_input = test_input_grid
            for aug_func, kwargs in aug_config:
                aug_test_input = aug_func(aug_test_input, **kwargs)
            
            # Reconstruct problem
            augmented_problem = intro
            for idx, (aug_input, aug_output) in enumerate(augmented_examples, 1):
                augmented_problem += f"Example {idx}:\nInput:\n{grid_to_string(aug_input)}\n\nOutput:\n{grid_to_string(aug_output)}\n\n"
            
            augmented_problem += f"Test Case:\nInput:\n{grid_to_string(aug_test_input)}\n\nOutput:"
            
            # Create prompt
            content = f"{system_prompt}\n\n{augmented_problem}" if system_prompt else augmented_problem
            messages = [{"role": "user", "content": content}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(formatted_prompt)
            
            # Track metadata
            prompt_metadata.append({
                'sample_idx': sample_idx,
                'aug_idx': aug_idx,
                'aug_config': aug_config
            })
    
    # Generate ALL outputs in ONE GIANT batch
    if all_prompts:
        print(f"  Generating {len(all_prompts)} augmented predictions in ONE batch ({len(parsed_problems) - len(fallback_indices)} problems × {num_augmentations} augmentations)...")
        if lora_request:
            outputs = llm.generate(all_prompts, sampling_params=sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    else:
        outputs = []
    
    # Collect votes per sample
    sample_votes = {i: [] for i in range(num_samples) if i not in fallback_indices}
    
    for output, metadata in zip(outputs, prompt_metadata):
        sample_idx = metadata['sample_idx']
        aug_config = metadata['aug_config']
        
        generated_text = output.outputs[0].text
        generated_grid = parse_grid_from_string(generated_text)
        
        if generated_grid:
            try:
                # Try to reverse the augmentations
                reversed_grid = reverse_augmentations_on_grid(generated_grid, aug_config)
                grid_string = grid_to_string(reversed_grid)
                sample_votes[sample_idx].append(grid_string)
            except Exception:
                # Skip failed reversals (as per user's request)
                pass
    
    # Compute majority vote for each sample
    final_results = [None] * num_samples
    
    for sample_idx in range(num_samples):
        if sample_idx in fallback_results:
            # Use fallback result
            final_results[sample_idx] = fallback_results[sample_idx]
        elif sample_votes[sample_idx]:
            # Vote on most common
            vote_counter = Counter(sample_votes[sample_idx])
            most_common = vote_counter.most_common(1)[0][0]
            final_results[sample_idx] = most_common
        else:
            # All augmentations failed, use empty grid
            final_results[sample_idx] = ""
    
    return final_results


def evaluate_model_vllm(
    model_path: str,
    eval_data_path: str,
    max_samples_per_level: int = 20,
    attempts_per_problem: int = 1,
    temperature: float = 0.7,
    use_system_prompt: bool = True,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
    print_examples: bool = False,
    inference_mode: str = "standard",  # "standard", "deep_dive", "augmented_voting"
    deep_dive_iterations: int = 16,
    num_augmentations: int = 30,
) -> Dict[str, Any]:
    """
    Evaluate model using vLLM for efficient batch generation.
    Uses reward functions to check validity of outputs.
    
    Args:
        model_path: Path to the model to evaluate
        eval_data_path: Path to evaluation data JSON file
        max_samples_per_level: Maximum samples to test per level (1-6)
        attempts_per_problem: Number of generation attempts per problem (ignored if inference_mode != "standard")
        temperature: Sampling temperature (default: 0.7)
        use_system_prompt: Whether to prepend optimized prompt to each problem
        use_lora: Whether to use LoRA adapter (default: False)
        lora_path: Path to LoRA adapter directory (required if use_lora=True)
        print_examples: Whether to print first example of first 10 problems (default: False)
        inference_mode: Inference strategy - "standard", "deep_dive", or "augmented_voting"
        deep_dive_iterations: Number of iterations for deep_dive mode (default: 16)
        num_augmentations: Number of augmentations for augmented_voting mode (default: 30)
        
    Returns:
        Dictionary with results per level and overall accuracy
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from reward_functions import parse_grid_from_string, check_value
    
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Inference mode: {inference_mode}")
    if inference_mode == "deep_dive":
        print(f"  Deep dive iterations: {deep_dive_iterations}")
    elif inference_mode == "augmented_voting":
        print(f"  Num augmentations: {num_augmentations}")
    print(f"Using system prompt: {use_system_prompt}")
    print(f"Using LoRA: {use_lora}")
    if use_lora:
        print(f"LoRA path: {lora_path}")
    print(f"Print examples: {print_examples}")
    print(f"{'='*60}")
    
    # Validate LoRA settings
    if use_lora and not lora_path:
        raise ValueError("lora_path must be provided when use_lora=True")
    
    # Load system prompt if requested
    system_prompt = ""
    if use_system_prompt:
        system_prompt = load_optimized_prompt()
        if system_prompt:
            print(f"Loaded system prompt ({len(system_prompt)} characters)")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize LLM with LoRA support if requested
    if use_lora:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            max_model_len=16000,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=256,
        )
        # Create LoRARequest for generation
        lora_request = LoRARequest("eval-lora", 1, lora_path)
    else:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            max_model_len=16000
        )
        lora_request = None
    
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=temperature,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
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
    
    # Handle different inference modes
    if inference_mode == "standard":
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
        
        # Generate ALL outputs in ONE batch (with LoRA if enabled)
        if use_lora:
            outputs = llm.generate(all_prompts, sampling_params=sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(all_prompts, sampling_params=sampling_params)
        
        # Convert outputs to list of (output, metadata) tuples
        outputs_with_metadata = list(zip(outputs, prompt_metadata))
        
    elif inference_mode == "deep_dive":
        # Deep dive: batched iterative refinement
        print(f"\nRunning deep dive inference (up to {deep_dive_iterations} iterations)...")
        
        # Collect all samples with metadata
        all_samples = []
        all_metadata = []
        for level, samples in level_samples.items():
            for sample_idx, sample in enumerate(samples):
                all_samples.append(sample)
                all_metadata.append({
                    'level': level,
                    'sample_idx': sample_idx,
                    'sample': sample,
                    'attempt': 0
                })
        
        print(f"  Processing {len(all_samples)} problems in batched mode...")
        
        # Run batched deep dive inference
        generated_texts = deep_dive_inference_batched(
            llm=llm,
            tokenizer=tokenizer,
            samples=all_samples,
            system_prompt=system_prompt if use_system_prompt else "",
            sampling_params=sampling_params,
            lora_request=lora_request,
            max_iterations=deep_dive_iterations
        )
        
        # Create output objects for compatibility
        class FakeOutput:
            def __init__(self, text):
                self.text = text
        
        class FakeOutputWrapper:
            def __init__(self, text):
                self.outputs = [FakeOutput(text)]
        
        outputs_with_metadata = [
            (FakeOutputWrapper(text), metadata)
            for text, metadata in zip(generated_texts, all_metadata)
        ]
        
    elif inference_mode == "augmented_voting":
        # Augmented voting: batched augmentation and voting
        print(f"\nRunning augmented voting inference ({num_augmentations} augmentations per problem)...")
        
        # Collect all samples with metadata
        all_samples = []
        all_metadata = []
        for level, samples in level_samples.items():
            for sample_idx, sample in enumerate(samples):
                all_samples.append(sample)
                all_metadata.append({
                    'level': level,
                    'sample_idx': sample_idx,
                    'sample': sample,
                    'attempt': 0
                })
        
        print(f"  Processing {len(all_samples)} problems in batched mode...")
        
        # Run batched augmented voting inference
        generated_texts = augmented_voting_inference_batched(
            llm=llm,
            tokenizer=tokenizer,
            samples=all_samples,
            system_prompt=system_prompt if use_system_prompt else "",
            sampling_params=sampling_params,
            lora_request=lora_request,
            num_augmentations=num_augmentations
        )
        
        # Create output objects for compatibility
        class FakeOutput:
            def __init__(self, text):
                self.text = text
        
        class FakeOutputWrapper:
            def __init__(self, text):
                self.outputs = [FakeOutput(text)]
        
        outputs_with_metadata = [
            (FakeOutputWrapper(text), metadata)
            for text, metadata in zip(generated_texts, all_metadata)
        ]
    else:
        raise ValueError(f"Unknown inference mode: {inference_mode}")
    
    print("Generation complete! Evaluating results...\n")
    
    # Print examples if requested (first 10 problems)
    if print_examples:
        print(f"\n{'='*60}")
        print("EXAMPLE OUTPUTS (First 10 Problems)")
        print(f"{'='*60}\n")
        examples_printed = 0
        for i, (output, metadata) in enumerate(outputs_with_metadata):
            # Only print first attempt of each problem
            if metadata['attempt'] == 0 and examples_printed < 10:
                level = metadata['level']
                sample_idx = metadata['sample_idx']
                sample = metadata['sample']
                generated_text = output.outputs[0].text
                
                print(f"Problem {examples_printed + 1} (Level {level}, Sample {sample_idx}):")
                print(f"Input:\n{sample['problem'][:200]}..." if len(sample['problem']) > 200 else f"Input:\n{sample['problem']}")
                print(f"\nGenerated Output:\n{generated_text}\n")
                print(f"Expected:\n{sample['answer']}\n")
                print("-" * 60)
                examples_printed += 1
                
                if examples_printed >= 10:
                    break
        print(f"{'='*60}\n")
    
    # Initialize results tracking
    results = {f"level_{level}": {"correct": 0, "total": 0} for level in range(1, 7)}
    results["overall"] = {"correct": 0, "total": 0}
    
    # Track which problems have been solved (for attempts_per_problem > 1)
    solved_problems = set()
    
    # Process outputs
    for i, (output, metadata) in enumerate(outputs_with_metadata):
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
                        help="Attempts per problem (default: 1, only for standard mode)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Disable system prompt")
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA adapter")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter directory (required if --use-lora)")
    parser.add_argument("--print-examples", action="store_true",
                        help="Print first example of first 10 problems")
    parser.add_argument("--inference-mode", type=str, default="standard",
                        choices=["standard", "deep_dive", "augmented_voting"],
                        help="Inference strategy (default: standard)")
    parser.add_argument("--deep-dive-iterations", type=int, default=16,
                        help="Number of iterations for deep_dive mode (default: 16)")
    parser.add_argument("--num-augmentations", type=int, default=30,
                        help="Number of augmentations for augmented_voting mode (default: 30)")
    
    args = parser.parse_args()
    
    results = evaluate_model_vllm(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        max_samples_per_level=args.samples_per_level,
        attempts_per_problem=args.attempts,
        temperature=args.temperature,
        use_system_prompt=not args.no_system_prompt,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        print_examples=args.print_examples,
        inference_mode=args.inference_mode,
        deep_dive_iterations=args.deep_dive_iterations,
        num_augmentations=args.num_augmentations,
    )
    
    # Save results
    output_file = f"{args.model_path.replace('/', '_')}_eval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

