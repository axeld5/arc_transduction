"""
Creates augmented training data from ARC datasets.

This module:
1. Processes problems from any folder of JSON files (training_data/, evaluation_data/, train_conceptarc/, eval_conceptarc/)
2. Uses leave-one-out sampling from available I/O pairs (train/test/arc-gen)
3. Applies random augmentations to entire problems (geometric + color transformations)
4. Creates text prompts without placeholders - just the problem and answer
5. Always includes one unaugmented sample per problem
6. For evaluation data: only creates unaugmented samples (no augmentation)

Key Functions:
- from_data_to_problem(): Converts problem dict to text prompt
- augment_problem(): Applies random augmentations to a problem
- create_training_sample_with_leave_one_out(): Randomly samples I/O pairs with leave-one-out
- create_training_examples_for_problem(): Creates multiple augmented samples for one problem
- create_ttft_examples_for_problem(): Creates Test-Time Fine-Tuning examples (only train data)
- create_training_data_from_folder(): Processes an entire folder of problems
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random

# No loader imports needed - we work directly with folders
from augment import (
    assign_random_augmentations,
    apply_augmentations_to_grids
)


def from_data_to_problem(data: Dict[str, Any], task_hash: str = None) -> Dict[str, str]:
    """
    Convert problem data to formatted text representation.
    
    Args:
        data: Problem data with train/test examples
        task_hash: Optional hash identifier for the task (consistent across augmentations)
        
    Returns:
        Dictionary with 'problem' text and 'answer' text
    """
    if task_hash:
        problem_text = f"Solve task {task_hash}\n\n"
    
    # Add training examples
    if 'train' in data:
        for i, example in enumerate(data['train'], 1):
            problem_text += "I:\n"
            for row in example['input']:
                problem_text += " ".join(map(str, row)) + "\n"
            problem_text += "O:\n"
            for row in example['output']:
                problem_text += " ".join(map(str, row)) + "\n"
            problem_text += "\n"
    
    # Add test case
    answer = ""
    if 'test' in data and len(data['test']) > 0:
        test_case = data['test'][0]
        problem_text += "I:\n"
        for row in test_case['input']:
            problem_text += " ".join(map(str, row)) + "\n"
        problem_text += "O:\n"
        # Store the actual answer
        for row in test_case['output']:
            answer += " ".join(map(str, row)) + "\n"
        answer = answer.strip()
    
    return {
        'problem': problem_text.strip(),
        'answer': answer
    }


def augment_problem(problem_data: Dict[str, Any], seed: int = None, shuffle_train_order: bool = True) -> Dict[str, Any]:
    """
    Apply random augmentations to an entire problem (train and test examples).
    
    Args:
        problem_data: Original problem data
        seed: Random seed for reproducibility
        shuffle_train_order: If True, randomly shuffle the order of training examples
        
    Returns:
        Augmented problem data
    """
    if seed is not None:
        random.seed(seed)
    
    # Choose augmentations
    augmentations = assign_random_augmentations(seed=seed)
    
    # Collect ALL grids from the entire problem to ensure consistent color mapping
    all_grids = []
    train_count = len(problem_data.get('train', []))
    test_count = len(problem_data.get('test', []))
    
    # Collect training grids
    for example in problem_data.get('train', []):
        all_grids.append(example['input'])
        all_grids.append(example['output'])
    
    # Collect test grids
    for example in problem_data.get('test', []):
        all_grids.append(example['input'])
        all_grids.append(example['output'])
    
    # Apply augmentations to ALL grids at once (ensures consistent color mapping)
    augmented_all = apply_augmentations_to_grids(all_grids, augmentations)
    
    # Reconstruct the problem structure
    augmented_data = {
        'train': [],
        'test': []
    }
    
    grid_idx = 0
    
    # Rebuild training examples
    for i in range(train_count):
        augmented_data['train'].append({
            'input': augmented_all[grid_idx],
            'output': augmented_all[grid_idx + 1]
        })
        grid_idx += 2
    
    # Shuffle training examples order if enabled (50% chance for variety)
    if shuffle_train_order and len(augmented_data['train']) > 1 and random.random() < 0.5:
        random.shuffle(augmented_data['train'])
    
    # Rebuild test examples
    for i in range(test_count):
        augmented_data['test'].append({
            'input': augmented_all[grid_idx],
            'output': augmented_all[grid_idx + 1]
        })
        grid_idx += 2
    
    return augmented_data


def create_training_sample_with_leave_one_out(
    problem_data: Dict[str, Any],
    seed: int = None
) -> Dict[str, Any]:
    """
    Create a training sample using leave-one-out from available I/O pairs.
    
    Randomly samples from all available I/O pairs (train, test, arc-gen),
    leaves one out as the test case, and uses the rest as training examples.
    
    Args:
        problem_data: Problem data with train/test/arc-gen examples
        seed: Random seed for reproducibility
        
    Returns:
        Modified problem data with leave-one-out structure
    """
    if seed is not None:
        random.seed(seed)
    
    # Collect all available I/O pairs
    all_pairs = []
    
    # Add train examples
    for example in problem_data.get('train', []):
        all_pairs.append(example.copy())
    
    # Add test examples
    for example in problem_data.get('test', []):
        all_pairs.append(example.copy())
    
    # Add arc-gen examples if available
    for example in problem_data.get('arc-gen', []):
        all_pairs.append(example.copy())
    
    if len(all_pairs) < 2:
        # Not enough pairs for leave-one-out, return original
        return problem_data
    
    # Randomly select one pair to be the test case
    random.shuffle(all_pairs)
    test_pair = all_pairs[0]
    train_pairs = all_pairs[1:]
    
    # Get the original number of train examples
    original_train_count = len(problem_data.get('train', []))
    
    # Sample randomly from remaining pairs to match original train count
    if len(train_pairs) > original_train_count:
        train_pairs = random.sample(train_pairs, original_train_count)
    
    return {
        'train': train_pairs,
        'test': [test_pair]
    }


def create_training_examples_for_problem(
    problem_name: str,
    problem_data: Dict[str, Any],
    source: str,
    num_samples: int = 30,
    data_dir: str = ".",
    include_augmented: bool = True
) -> List[Dict[str, Any]]:
    """
    Create training examples for a problem using leave-one-out sampling.
    
    Args:
        problem_name: Name of the problem
        problem_data: Original problem data
        source: Source of the problem (concept name, "training", or "evaluation")
        num_samples: Number of samples to create
        data_dir: Root directory
        include_augmented: If False, only create unaugmented samples (for evaluation)
        
    Returns:
        List of training examples with metadata
    """
    training_examples = []
    
    # Generate consistent task hash for this problem
    task_hash = f"{hash(problem_name) % (2**32):08x}"
    
    # First, create one UNAUGMENTED sample
    seed = hash(f"{problem_name}_unaugmented") % (2**32)
    unaugmented_data = create_training_sample_with_leave_one_out(problem_data, seed=seed)
    
    formatted = from_data_to_problem(unaugmented_data, task_hash=task_hash)
    
    training_examples.append({
        'problem': formatted['problem'],
        'answer': formatted['answer'],
        'metadata': {
            'source': source,
            'problem_name': problem_name,
            'augmented': False,
            'sample_idx': 0,
            'seed': seed
        }
    })
    
    # Create augmented samples only if requested
    if include_augmented:
        for sample_idx in range(1, num_samples):
            seed = hash(f"{problem_name}_sample_{sample_idx}") % (2**32)
            
            # Apply leave-one-out sampling
            sampled_data = create_training_sample_with_leave_one_out(problem_data, seed=seed)
            
            # Apply augmentations to the entire problem
            augmented_data = augment_problem(sampled_data, seed=seed, shuffle_train_order=True)
            
            # Convert to formatted text
            formatted = from_data_to_problem(augmented_data, task_hash=task_hash)
            
            training_examples.append({
                'problem': formatted['problem'],
                'answer': formatted['answer'],
                'metadata': {
                    'source': source,
                    'problem_name': problem_name,
                    'augmented': True,
                    'sample_idx': sample_idx,
                    'seed': seed
                }
            })
    
    return training_examples


def create_ttft_examples_for_problem(
    problem_name: str,
    problem_data: Dict[str, Any],
    source: str,
    num_samples: int = 30,
    data_dir: str = "."
) -> List[Dict[str, Any]]:
    """
    Create Test-Time Fine-Tuning examples using only train data with leave-one-out.
    
    Args:
        problem_name: Name of the problem
        problem_data: Original problem data
        source: Source of the problem
        num_samples: Number of samples to create
        data_dir: Root directory
        
    Returns:
        List of TTFT examples with metadata
    """
    training_examples = []
    
    # Generate consistent task hash for this problem
    task_hash = f"{hash(problem_name) % (2**32):08x}"
    
    # Only use train examples
    train_examples = problem_data.get('train', [])
    
    if len(train_examples) < 2:
        # Not enough for leave-one-out
        return training_examples
    
    # First, create one UNAUGMENTED sample
    seed = hash(f"{problem_name}_ttft_unaugmented") % (2**32)
    random.seed(seed)
    
    # Randomly select one train example to hold out
    held_out_idx = random.randint(0, len(train_examples) - 1)
    
    unaugmented_data = {
        'train': [ex for i, ex in enumerate(train_examples) if i != held_out_idx],
        'test': [train_examples[held_out_idx]]
    }
    
    formatted = from_data_to_problem(unaugmented_data, task_hash=task_hash)
    
    training_examples.append({
        'problem': formatted['problem'],
        'answer': formatted['answer'],
        'metadata': {
            'source': source,
            'problem_name': problem_name,
            'method': 'ttft',
            'augmented': False,
            'sample_idx': 0,
            'seed': seed
        }
    })
    
    # Create augmented samples
    for sample_idx in range(1, num_samples):
        seed = hash(f"{problem_name}_ttft_sample_{sample_idx}") % (2**32)
        random.seed(seed)
        
        # Randomly select one train example to hold out
        held_out_idx = random.randint(0, len(train_examples) - 1)
        
        sampled_data = {
            'train': [ex for i, ex in enumerate(train_examples) if i != held_out_idx],
            'test': [train_examples[held_out_idx]]
        }
        
        # Apply augmentations
        augmented_data = augment_problem(sampled_data, seed=seed, shuffle_train_order=True)
        
        # Convert to formatted text
        formatted = from_data_to_problem(augmented_data, task_hash=task_hash)
        
        training_examples.append({
            'problem': formatted['problem'],
            'answer': formatted['answer'],
            'metadata': {
                'source': source,
                'problem_name': problem_name,
                'method': 'ttft',
                'augmented': True,
                'sample_idx': sample_idx,
                'seed': seed
            }
        })
    
    return training_examples


def create_training_data_from_folder(
    folder_path: str,
    num_samples: int = 30,
    output_file: str = None,
    verbose: bool = True,
    include_augmented: bool = True
) -> List[Dict[str, Any]]:
    """
    Create training data from a folder of JSON problem files.
    
    Works with:
    - training_data/
    - evaluation_data/
    - train_conceptarc/
    - eval_conceptarc/
    
    Args:
        folder_path: Path to folder containing JSON problem files
        num_samples: Number of samples per problem
        output_file: File to save output
        verbose: Print progress
        include_augmented: If False, only create unaugmented samples (for evaluation)
        
    Returns:
        List of training examples
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    all_examples = []
    
    # Get all JSON files in the folder
    problem_files = list(folder.glob("*.json"))
    
    if verbose:
        print(f"\nProcessing folder: {folder_path}")
        print(f"Found {len(problem_files)} problems")
        if not include_augmented:
            print("  Mode: Unaugmented only (evaluation)")
    
    for problem_file in problem_files:
        problem_name = problem_file.stem
        
        # Extract source/concept from problem name
        # For ConceptARC: "AboveBelow1" -> "AboveBelow"
        # For ARC: just use the problem ID as-is
        if any(c.isdigit() for c in problem_name) and any(c.isalpha() for c in problem_name):
            # Has both letters and numbers, extract concept
            source = ''.join([c for c in problem_name if not c.isdigit()])
        else:
            # Just use the folder name as source
            source = folder.name
        
        if verbose:
            print(f"  Processing {problem_name}...")
        
        try:
            with open(problem_file, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
        except Exception as e:
            print(f"    Error loading {problem_name}: {e}")
            continue
        
        examples = create_training_examples_for_problem(
            problem_name=problem_name,
            problem_data=problem_data,
            source=source,
            num_samples=num_samples,
            data_dir=".",
            include_augmented=include_augmented
        )
        
        all_examples.extend(examples)
        
        if verbose:
            print(f"    Created {len(examples)} examples")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2)
        if verbose:
            print(f"\nSaved {len(all_examples)} examples to {output_file}")
    
    return all_examples


if __name__ == "__main__":
    import os
    
    print("="*60)
    print("Training Data Generation for ARC")
    print("="*60)
    print("")
    print("Strategy: Leave-One-Out Sampling")
    print("  - Randomly samples from available I/O pairs (train/test/arc-gen)")
    print("  - Leaves one out as test case")
    print("  - Applies random augmentations to entire problems")
    print("  - Always includes one unaugmented sample")
    print("")
    print("="*60)
    print("")
    
    # Create output directory
    os.makedirs("generated_data", exist_ok=True)
    
    # Generate training data from different sources
    print("STEP 1: Creating training data from train_conceptarc...")
    train_conceptarc_examples = create_training_data_from_folder(
        folder_path="train_conceptarc",
        num_samples=30,
        output_file="generated_data/train_conceptarc_data.json",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("STEP 2: Creating evaluation data from eval_conceptarc...")
    eval_conceptarc_examples = create_training_data_from_folder(
        folder_path="eval_conceptarc",
        num_samples=30,
        output_file="generated_data/eval_conceptarc_data.json",
        verbose=True,
        include_augmented=False
    )
    
    print("\n" + "="*60)
    print("STEP 3: Creating training data from ARC training_data...")
    arc_train_examples = create_training_data_from_folder(
        folder_path="training_data",
        num_samples=30,
        output_file="generated_data/train_arc_data.json",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("STEP 4: Creating evaluation data from ARC evaluation_data...")
    arc_eval_examples = create_training_data_from_folder(
        folder_path="evaluation_data",
        num_samples=30,
        output_file="generated_data/eval_arc_data.json",
        verbose=True,
        include_augmented=False
    )
    
    print("\n" + "="*60)
    print("All data generation complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  ConceptARC Train: {len(train_conceptarc_examples)} examples")
    print(f"  ConceptARC Eval: {len(eval_conceptarc_examples)} examples")
    print(f"  ARC Train: {len(arc_train_examples)} examples")
    print(f"  ARC Eval: {len(arc_eval_examples)} examples")
    print(f"  Total: {len(train_conceptarc_examples) + len(eval_conceptarc_examples) + len(arc_train_examples) + len(arc_eval_examples)} examples")
    print("="*60)
