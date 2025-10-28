"""
Creates augmented training data from ConceptARC corpus.

This module:
1. Loads problems from the ConceptARC corpus
2. Augments each problem 30 times using geometric and color transformations
3. Creates 20 placeholder solutions for each augmented version (various difficulty levels)
4. Formats data for training with metadata
5. Saves the processed training data
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random

from loader import (
    list_corpus_concepts,
    list_corpus_problems_by_concept,
    load_corpus_problem_by_name
)
from augment import (
    assign_random_augmentations,
    apply_augmentations_to_grids
)
from placeholder_creation import create_placeholder


def from_data_to_problem(data: Dict[str, Any], include_test_output: bool = False) -> Dict[str, str]:
    """
    Convert problem data to formatted text representation.
    
    Args:
        data: Problem data with train/test examples
        include_test_output: If True, include test output (for training). If False, exclude it.
        
    Returns:
        Dictionary with 'problem' text and 'answer' text
    """
    problem_text = "Identify the pattern within the grids and solve the problem.\n\n"
    
    # Add training examples
    if 'train' in data:
        problem_text += "Training Examples:\n"
        for i, example in enumerate(data['train'], 1):
            problem_text += f"Example {i}:\n"
            problem_text += "Input:\n"
            for row in example['input']:
                problem_text += " ".join(map(str, row)) + "\n"
            problem_text += "Output:\n"
            for row in example['output']:
                problem_text += " ".join(map(str, row)) + "\n"
            problem_text += "\n"
    
    # Add test case
    answer = ""
    if 'test' in data and len(data['test']) > 0:
        test_case = data['test'][0]
        problem_text += "Test Case:\n"
        problem_text += "Input:\n"
        for row in test_case['input']:
            problem_text += " ".join(map(str, row)) + "\n"
        
        if include_test_output:
            problem_text += "Output Placeholder:\n"
            for row in test_case['output']:
                problem_text += " ".join(map(str, row)) + "\n"
        problem_text += "Output:\n"
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


def create_training_examples_for_problem_leaveoneout(
    problem_name: str,
    problem_data: Dict[str, Any],
    concept: str,
    num_augmentations: int = 30,
    placeholders_per_augmentation: int = 20,
    data_dir: str = "."
) -> List[Dict[str, Any]]:
    """
    Create training examples using leave-one-out strategy (only train examples).
    
    Randomly selects one training example to hold out as test, uses rest as training.
    This creates self-supervised training data without using actual test cases.
    
    Args:
        problem_name: Name of the problem (e.g., "AboveBelow1")
        problem_data: Original problem data
        concept: Concept category (e.g., "AboveBelow")
        num_augmentations: Number of augmented versions to create
        placeholders_per_augmentation: Number of placeholders per augmented version
        data_dir: Root directory
        
    Returns:
        List of training examples with metadata
    """
    training_examples = []
    
    # Check if there are enough training examples
    train_examples = problem_data.get('train', [])
    if len(train_examples) < 2:
        print(f"Warning: {problem_name} has fewer than 2 training examples, skipping leave-one-out...")
        return training_examples
    
    # Randomly select ONE training example to hold out
    held_out_idx = random.randint(0, len(train_examples) - 1)
    
    # Create modified problem data with leave-one-out
    modified_data = {
        'train': [ex for i, ex in enumerate(train_examples) if i != held_out_idx],
        'test': [train_examples[held_out_idx]]
    }
    
    ground_truth = modified_data['test'][0]['output']
    
    # Create augmented versions
    for aug_idx in range(num_augmentations):
        seed = hash(f"{problem_name}_loo_{held_out_idx}_{aug_idx}") % (2**32)
        augmented_data = augment_problem(modified_data, seed=seed)
        augmented_ground_truth = augmented_data['test'][0]['output']
        
        # First, save the unmodified problem (no placeholder, just ground truth)
        unmodified_problem = {
            'train': augmented_data['train'],
            'test': [{
                'input': augmented_data['test'][0]['input'],
                'output': augmented_ground_truth
            }]
        }
        
        formatted_unmodified = from_data_to_problem(unmodified_problem, include_test_output=True)
        answer = ""
        for row in augmented_ground_truth:
            answer += " ".join(map(str, row)) + "\n"
        answer = answer.strip()
        
        training_examples.append({
            'problem': formatted_unmodified['problem'],
            'answer': answer,
            'metadata': {
                'concept': concept,
                'problem_name': problem_name,
                'method': 'leave_one_out',
                'held_out_idx': held_out_idx,
                'augmentation_idx': aug_idx,
                'augmentation_seed': seed,
                'level': 0,  # 0 means unmodified/ground truth
                'placeholder_idx': -1,
                'placeholder_seed': -1
            }
        })
        
        # Create placeholders for each level (4 levels, placeholders_per_augmentation per level)
        for level in range(1, 5):
            for placeholder_idx in range(placeholders_per_augmentation):
                # Create placeholder
                placeholder_seed = hash(f"{problem_name}_loo_{held_out_idx}_{aug_idx}_{level}_{placeholder_idx}") % (2**32)
                random.seed(placeholder_seed)
                
                placeholder = create_placeholder(augmented_data, augmented_ground_truth, level=level)
                
                # Create problem data with placeholder as output
                problem_with_placeholder = {
                    'train': augmented_data['train'],
                    'test': [{
                        'input': augmented_data['test'][0]['input'],
                        'output': placeholder
                    }]
                }
                
                # Convert to formatted text (placeholder in problem)
                formatted = from_data_to_problem(problem_with_placeholder, include_test_output=True)
                
                # Create the answer from the ground truth (NOT the placeholder)
                answer = ""
                for row in augmented_ground_truth:
                    answer += " ".join(map(str, row)) + "\n"
                answer = answer.strip()
                
                # Add metadata
                training_example = {
                    'problem': formatted['problem'],
                    'answer': answer,  # Always the ground truth, not the placeholder
                    'metadata': {
                        'concept': concept,
                        'problem_name': problem_name,
                        'method': 'leave_one_out',
                        'held_out_idx': held_out_idx,
                        'augmentation_idx': aug_idx,
                        'augmentation_seed': seed,
                        'level': level,
                        'placeholder_idx': placeholder_idx,
                        'placeholder_seed': placeholder_seed
                    }
                }
                
                training_examples.append(training_example)
    
    return training_examples


def create_training_examples_for_problem_eval(
    problem_name: str,
    problem_data: Dict[str, Any],
    concept: str,
    num_augmentations: int = 30,
    placeholders_per_augmentation: int = 20,
    data_dir: str = "."
) -> List[Dict[str, Any]]:
    """
    Create training examples using actual test examples (for evaluation/testing).
    
    Uses the actual test cases provided in the problem data.
    
    Args:
        problem_name: Name of the problem (e.g., "AboveBelow1")
        problem_data: Original problem data
        concept: Concept category (e.g., "AboveBelow")
        num_augmentations: Number of augmented versions to create
        placeholders_per_augmentation: Number of placeholders per augmented version
        data_dir: Root directory
        
    Returns:
        List of training examples with metadata
    """
    training_examples = []
    
    # Get the ground truth from the first test case
    if not problem_data.get('test') or len(problem_data['test']) == 0:
        print(f"Warning: {problem_name} has no test cases, skipping...")
        return training_examples
    
    ground_truth = problem_data['test'][0]['output']
    
    # Create augmented versions
    for aug_idx in range(num_augmentations):
        seed = hash(f"{problem_name}_eval_{aug_idx}") % (2**32)
        augmented_data = augment_problem(problem_data, seed=seed)
        augmented_ground_truth = augmented_data['test'][0]['output']
        
        # First, save the unmodified problem (no placeholder, just ground truth)
        unmodified_problem = {
            'train': augmented_data['train'],
            'test': [{
                'input': augmented_data['test'][0]['input'],
                'output': augmented_ground_truth
            }]
        }
        
        formatted_unmodified = from_data_to_problem(unmodified_problem, include_test_output=True)
        answer = ""
        for row in augmented_ground_truth:
            answer += " ".join(map(str, row)) + "\n"
        answer = answer.strip()
        
        training_examples.append({
            'problem': formatted_unmodified['problem'],
            'answer': answer,
            'metadata': {
                'concept': concept,
                'problem_name': problem_name,
                'method': 'eval',
                'augmentation_idx': aug_idx,
                'augmentation_seed': seed,
                'level': 0,  # 0 means unmodified/ground truth
                'placeholder_idx': -1,
                'placeholder_seed': -1
            }
        })
        
        # Create placeholders for each level (4 levels, placeholders_per_augmentation per level)
        for level in range(1, 5):
            for placeholder_idx in range(placeholders_per_augmentation):
                # Create placeholder
                placeholder_seed = hash(f"{problem_name}_eval_{aug_idx}_{level}_{placeholder_idx}") % (2**32)
                random.seed(placeholder_seed)
                
                placeholder = create_placeholder(augmented_data, augmented_ground_truth, level=level)
                
                # Create problem data with placeholder as output
                problem_with_placeholder = {
                    'train': augmented_data['train'],
                    'test': [{
                        'input': augmented_data['test'][0]['input'],
                        'output': placeholder
                    }]
                }
                
                # Convert to formatted text (placeholder in problem)
                formatted = from_data_to_problem(problem_with_placeholder, include_test_output=True)
                
                # Create the answer from the ground truth (NOT the placeholder)
                answer = ""
                for row in augmented_ground_truth:
                    answer += " ".join(map(str, row)) + "\n"
                answer = answer.strip()
                
                # Add metadata
                training_example = {
                    'problem': formatted['problem'],
                    'answer': answer,  # Always the ground truth, not the placeholder
                    'metadata': {
                        'concept': concept,
                        'problem_name': problem_name,
                        'method': 'eval',
                        'augmentation_idx': aug_idx,
                        'augmentation_seed': seed,
                        'level': level,
                        'placeholder_idx': placeholder_idx,
                        'placeholder_seed': placeholder_seed
                    }
                }
                
                training_examples.append(training_example)
    
    return training_examples


def create_training_data_for_concept(
    concept: str,
    method: str = "eval",
    num_augmentations: int = 30,
    placeholders_per_augmentation: int = 20,
    data_dir: str = ".",
    verbose: bool = True,
    problem_filter: callable = None
) -> List[Dict[str, Any]]:
    """
    Create training data for selected problems in a concept category.
    
    Args:
        concept: Concept category name (e.g., "AboveBelow")
        method: "eval" (use test examples) or "leave_one_out" (use only train examples)
        num_augmentations: Number of augmented versions per problem
        placeholders_per_augmentation: Number of placeholders per augmentation
        data_dir: Root directory
        verbose: Print progress information
        problem_filter: Optional function to filter which problems to process
        
    Returns:
        List of all training examples for the concept
    """
    if verbose:
        print(f"\nProcessing concept: {concept} (method: {method})")
    
    # Select the appropriate function based on method
    if method == "leave_one_out":
        creation_func = create_training_examples_for_problem_leaveoneout
    elif method == "eval":
        creation_func = create_training_examples_for_problem_eval
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'eval' or 'leave_one_out'")
    
    # Get all problems for this concept
    problem_names = list_corpus_problems_by_concept(concept, data_dir=data_dir)
    
    # Apply filter if provided
    if problem_filter:
        problem_names = [name for name in problem_names if problem_filter(name)]
    
    all_training_examples = []
    
    for i, problem_name in enumerate(problem_names, 1):
        if verbose:
            print(f"  Processing {problem_name} ({i}/{len(problem_names)})...")
        
        # Load problem
        try:
            problem_data = load_corpus_problem_by_name(problem_name, data_dir=data_dir)
        except Exception as e:
            print(f"    Error loading {problem_name}: {e}")
            continue
        
        # Create training examples using the selected method
        examples = creation_func(
            problem_name=problem_name,
            problem_data=problem_data,
            concept=concept,
            num_augmentations=num_augmentations,
            placeholders_per_augmentation=placeholders_per_augmentation,
            data_dir=data_dir
        )
        
        all_training_examples.extend(examples)
        
        if verbose:
            print(f"    Created {len(examples)} training examples")
    
    if verbose:
        print(f"Total examples for {concept}: {len(all_training_examples)}")
    
    return all_training_examples


def create_training_data_for_corpus(
    method: str = "eval",
    num_augmentations: int = 30,
    placeholders_per_augmentation: int = 20,
    data_dir: str = ".",
    output_dir: str = None,
    save_by_concept: bool = False,
    save_combined: bool = True,
    verbose: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create training data for the entire ConceptARC corpus.
    
    Args:
        method: "eval" (use test examples) or "leave_one_out" (use only train examples)
        num_augmentations: Number of augmented versions per problem
        placeholders_per_augmentation: Number of placeholders per augmentation
        data_dir: Root directory containing corpus
        output_dir: Directory to save output files (default: data_dir)
        save_by_concept: Save separate files for each concept
        save_combined: Save one combined file with all data
        verbose: Print progress information
        
    Returns:
        Dictionary mapping concept names to their training examples
    """
    if output_dir is None:
        output_dir = data_dir
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all concepts
    concepts = list_corpus_concepts(data_dir=data_dir)
    
    if verbose:
        print(f"Found {len(concepts)} concepts in corpus")
        print(f"Concepts: {', '.join(concepts)}")
        print(f"Method: {method}")
    
    all_data = {}
    combined_examples = []
    
    # Process each concept
    for concept_idx, concept in enumerate(concepts, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing concept {concept_idx}/{len(concepts)}: {concept}")
            print(f"{'='*60}")
        
        examples = create_training_data_for_concept(
            concept=concept,
            method=method,
            num_augmentations=num_augmentations,
            placeholders_per_augmentation=placeholders_per_augmentation,
            data_dir=data_dir,
            verbose=verbose
        )
        
        all_data[concept] = examples
        combined_examples.extend(examples)
        
        # Save individual concept file
        if save_by_concept:
            concept_file = os.path.join(output_dir, f"{concept}_training_data.json")
            with open(concept_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2)
            if verbose:
                print(f"  Saved to: {concept_file}")
    
    # Save combined file
    if save_combined:
        # Use eval_data.json or train_data.json based on method
        if method == "eval":
            combined_file = os.path.join(output_dir, "eval_data.json")
        else:
            combined_file = os.path.join(output_dir, "train_data.json")
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_examples, f, indent=2)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Total examples across all concepts: {len(combined_examples)}")
            print(f"Saved combined data to: {combined_file}")
            print(f"{'='*60}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, f"{method}_data_summary.json")
    summary = {
        'method': method,
        'total_concepts': len(concepts),
        'total_examples': len(combined_examples),
        'num_augmentations_per_problem': num_augmentations,
        'placeholders_per_augmentation': placeholders_per_augmentation,
        'concepts': {
            concept: {
                'num_examples': len(examples),
                'num_problems': len(set(ex['metadata']['problem_name'] for ex in examples))
            }
            for concept, examples in all_data.items()
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\nSaved summary to: {summary_file}")
    
    return all_data


def create_training_data_leave_one_file_out(
    num_augmentations: int = 30,
    placeholders_per_augmentation: int = 20,
    data_dir: str = ".",
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Create training and evaluation data using leave-one-file-out strategy.
    
    For each concept folder:
    - All JSON files except the last one are used for training
    - The last JSON file is used for evaluation
    
    Args:
        num_augmentations: Number of augmented versions per problem
        placeholders_per_augmentation: Number of placeholders per augmentation
        data_dir: Root directory containing corpus
        output_dir: Directory to save output files (default: data_dir)
        verbose: Print progress information
        
    Returns:
        Tuple of (train_data, eval_data) dictionaries
    """
    if output_dir is None:
        output_dir = data_dir
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all concepts
    concepts = list_corpus_concepts(data_dir=data_dir)
    
    if verbose:
        print(f"Found {len(concepts)} concepts in corpus")
        print(f"Concepts: {', '.join(concepts)}")
        print(f"Strategy: Leave-one-file-out (last file of each concept for eval)")
        print("="*60)
    
    train_data = {}
    eval_data = {}
    train_combined = []
    eval_combined = []
    
    # Process each concept
    for concept_idx, concept in enumerate(concepts, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing concept {concept_idx}/{len(concepts)}: {concept}")
            print(f"{'='*60}")
        
        # Get all problems for this concept
        all_problems = list_corpus_problems_by_concept(concept, data_dir=data_dir)
        
        if len(all_problems) == 0:
            if verbose:
                print(f"  Warning: No problems found for concept {concept}")
            continue
        
        # Split: all but last for training, last for eval
        train_problems = all_problems[:-1]
        eval_problems = [all_problems[-1]]
        
        if verbose:
            print(f"  Total problems: {len(all_problems)}")
            print(f"  Training problems: {len(train_problems)} - {train_problems}")
            print(f"  Evaluation problem: {eval_problems[0]}")
        
        # Create training examples (using eval method on train problems)
        if verbose:
            print(f"\n  Creating TRAINING data from {len(train_problems)} problems...")
        
        train_examples = create_training_data_for_concept(
            concept=concept,
            method="eval",  # Use eval method (actual test examples)
            num_augmentations=num_augmentations,
            placeholders_per_augmentation=placeholders_per_augmentation,
            data_dir=data_dir,
            verbose=verbose,
            problem_filter=lambda name: name in train_problems
        )
        
        train_data[concept] = train_examples
        train_combined.extend(train_examples)
        
        # Create evaluation examples (using eval method on eval problem)
        if verbose:
            print(f"\n  Creating EVALUATION data from {len(eval_problems)} problem...")
        
        eval_examples = create_training_data_for_concept(
            concept=concept,
            method="eval",  # Use eval method (actual test examples)
            num_augmentations=num_augmentations,
            placeholders_per_augmentation=placeholders_per_augmentation,
            data_dir=data_dir,
            verbose=verbose,
            problem_filter=lambda name: name in eval_problems
        )
        
        eval_data[concept] = eval_examples
        eval_combined.extend(eval_examples)
    
    # Save training data
    train_file = os.path.join(output_dir, "train_data.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_combined, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING DATA:")
        print(f"  Total examples: {len(train_combined)}")
        print(f"  Saved to: {train_file}")
    
    # Save evaluation data
    eval_file = os.path.join(output_dir, "eval_data.json")
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_combined, f, indent=2)
    
    if verbose:
        print(f"\nEVALUATION DATA:")
        print(f"  Total examples: {len(eval_combined)}")
        print(f"  Saved to: {eval_file}")
        print(f"{'='*60}")
    
    # Save summary for training data
    train_summary = {
        'method': 'leave_one_file_out_train',
        'total_concepts': len(concepts),
        'total_examples': len(train_combined),
        'num_augmentations_per_problem': num_augmentations,
        'placeholders_per_augmentation': placeholders_per_augmentation,
        'concepts': {
            concept: {
                'num_examples': len(examples),
                'num_problems': len(set(ex['metadata']['problem_name'] for ex in examples))
            }
            for concept, examples in train_data.items()
        }
    }
    
    train_summary_file = os.path.join(output_dir, "train_data_summary.json")
    with open(train_summary_file, 'w', encoding='utf-8') as f:
        json.dump(train_summary, f, indent=2)
    
    # Save summary for evaluation data
    eval_summary = {
        'method': 'leave_one_file_out_eval',
        'total_concepts': len(concepts),
        'total_examples': len(eval_combined),
        'num_augmentations_per_problem': num_augmentations,
        'placeholders_per_augmentation': placeholders_per_augmentation,
        'concepts': {
            concept: {
                'num_examples': len(examples),
                'num_problems': len(set(ex['metadata']['problem_name'] for ex in examples)),
                'eval_problem': list(set(ex['metadata']['problem_name'] for ex in examples))[0] if examples else None
            }
            for concept, examples in eval_data.items()
        }
    }
    
    eval_summary_file = os.path.join(output_dir, "eval_data_summary.json")
    with open(eval_summary_file, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2)
    
    if verbose:
        print(f"\nSaved summaries:")
        print(f"  Training: {train_summary_file}")
        print(f"  Evaluation: {eval_summary_file}")
    
    return train_data, eval_data


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Training Data Generation for ConceptARC Corpus")
    print("="*60)
    print("")
    print("Strategy: Leave-One-File-Out")
    print("  - For each concept folder:")
    print("    * All JSON files except the last one -> TRAINING")
    print("    * The last JSON file -> EVALUATION")
    print("")
    print("Configuration:")
    print("  - 30 augmented versions per problem")
    print("  - Augmentations include:")
    print("    * Geometric transformations (rotation, flip, transpose)")
    print("    * Color permutations")
    print("    * Training example order shuffling (50% chance)")
    print("  - Level 0: Unmodified (ground truth in test output)")
    print("  - Levels 1-4: 5 placeholders per level")
    print("    * Level 1: Input/ground truth with random pixel modifications")
    print("    * Level 2: 50% dimension changes, 50% crop/upscale")
    print("    * Level 3: 3x3 zeros matrix")
    print("    * Level 4: 50% random matrix, 50% matrix from problem data")
    print(f"  - Total per problem: 30x(1 + 5x4) = {30*(1+5*4)} examples")
    print("="*60)
    print("")
    
    # Run leave-one-file-out strategy
    train_data, eval_data = create_training_data_leave_one_file_out(
        num_augmentations=30,
        placeholders_per_augmentation=5,
        data_dir=".",
        output_dir="./generated_data/",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("All training data generation complete!")
    print("="*60)

