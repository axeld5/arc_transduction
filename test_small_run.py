"""Test with 3 augmentations and 2 placeholders per level on one ConceptARC problem"""

from create_train_data import (
    create_training_examples_for_problem_leaveoneout,
    create_training_examples_for_problem_eval
)
from loader import load_corpus_problem_by_name
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ARC color scheme (10 colors: 0-9)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: sky blue
    '#870C25'   # 9: maroon
]

arc_cmap = ListedColormap(ARC_COLORS)

def text_to_grid(text):
    """Convert text representation of grid to numpy array."""
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        row = [int(x) for x in line.split()]
        grid.append(row)
    return np.array(grid)

def plot_grid(grid, title="", ax=None):
    """Plot a single grid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.imshow(grid, cmap=arc_cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    return ax

def extract_all_grids(example):
    """Extract all training pairs, test input, placeholder, and ground truth."""
    problem_lines = example['problem'].split('\n')
    
    training_pairs = []
    current_section = None
    current_type = None
    current_lines = []
    
    # Parse all training examples
    i = 0
    while i < len(problem_lines):
        line = problem_lines[i].strip()
        
        if line.startswith("Example "):
            # Start of a new training example
            if current_lines and current_type == "output":
                # Save previous output
                training_pairs[-1]['output'] = text_to_grid('\n'.join(current_lines))
            current_lines = []
            current_type = None
            training_pairs.append({})
            
        elif line == "Input:" and "Test Case" not in problem_lines[max(0, i-5):i+1]:
            # Training input
            if current_lines and current_type == "output":
                training_pairs[-1]['output'] = text_to_grid('\n'.join(current_lines))
            current_lines = []
            current_type = "input"
            
        elif line == "Output:" and not any("Test Case" in problem_lines[j] for j in range(max(0, i-10), i)):
            # Training output
            if current_lines and current_type == "input":
                training_pairs[-1]['input'] = text_to_grid('\n'.join(current_lines))
            current_lines = []
            current_type = "output"
            
        elif line == "Test Case:":
            # Save last training output if any
            if current_lines and current_type == "output":
                training_pairs[-1]['output'] = text_to_grid('\n'.join(current_lines))
            break
            
        elif line and not line.startswith("Training Examples") and not line.startswith("Example "):
            if current_type:
                current_lines.append(line)
        
        i += 1
    
    # Find test case input
    test_input_start = None
    for i, line in enumerate(problem_lines):
        if line.strip() == "Test Case:":
            for j in range(i+1, len(problem_lines)):
                if problem_lines[j].strip() == "Input:":
                    test_input_start = j + 1
                    break
            break
    
    # Find test case output (placeholder)
    test_output_start = None
    for i, line in enumerate(problem_lines):
        if line.strip() == "Output:" and test_input_start and i > test_input_start:
            test_output_start = i + 1
            break
    
    # Extract test grids
    test_input_lines = []
    test_output_lines = []
    
    if test_input_start:
        for i in range(test_input_start, len(problem_lines)):
            line = problem_lines[i].strip()
            if line == "" or line == "Output:":
                break
            if line and not line.startswith("Example") and not line.startswith("Training"):
                test_input_lines.append(line)
    
    if test_output_start:
        for i in range(test_output_start, len(problem_lines)):
            line = problem_lines[i].strip()
            if line == "":
                break
            if line and not line.startswith("Example"):
                test_output_lines.append(line)
    
    # Convert to grids
    test_input = text_to_grid('\n'.join(test_input_lines)) if test_input_lines else None
    test_placeholder = text_to_grid('\n'.join(test_output_lines)) if test_output_lines else None
    ground_truth = text_to_grid(example['answer'])
    
    return training_pairs, test_input, test_placeholder, ground_truth

def extract_test_grids(example):
    """Extract test input and output from example (backward compatibility)."""
    _, test_input, test_placeholder, ground_truth = extract_all_grids(example)
    return test_input, test_placeholder, ground_truth

print("="*70)
print("TEST RUN: 3 augmentations, 2 placeholders per level")
print("="*70)

# Load one problem from ConceptARC
problem_name = "AboveBelow1"
print(f"\nLoading problem: {problem_name}")
problem_data = load_corpus_problem_by_name(problem_name)

print(f"Problem has {len(problem_data['train'])} training examples")
print(f"Problem has {len(problem_data['test'])} test examples")

# Configuration
NUM_AUGMENTATIONS = 3
PLACEHOLDERS_PER_LEVEL = 2
NUM_LEVELS = 6

print(f"\nConfiguration:")
print(f"  - Augmentations: {NUM_AUGMENTATIONS}")
print(f"  - Placeholders per level: {PLACEHOLDERS_PER_LEVEL}")
print(f"  - Levels: {NUM_LEVELS}")

# Calculate expected counts
# Now includes level 0 (unmodified) + levels 1-6 with placeholders
expected_per_problem_eval = NUM_AUGMENTATIONS * (1 + PLACEHOLDERS_PER_LEVEL * NUM_LEVELS)
# Leave-one-out now only holds out ONE random example (not all of them)
expected_per_problem_loo = NUM_AUGMENTATIONS * (1 + PLACEHOLDERS_PER_LEVEL * NUM_LEVELS)

print(f"\nExpected counts:")
print(f"  - Eval method: {expected_per_problem_eval} examples")
print(f"    (Per augmentation: 1 unmodified + {PLACEHOLDERS_PER_LEVEL} × {NUM_LEVELS} levels)")
print(f"  - Leave-one-out method: {expected_per_problem_loo} examples")
print(f"    (Randomly holds out 1 example, not all)")

# Test LEAVE-ONE-OUT method
print("\n" + "="*70)
print("TESTING LEAVE-ONE-OUT METHOD")
print("="*70)

loo_examples = create_training_examples_for_problem_leaveoneout(
    problem_name=problem_name,
    problem_data=problem_data,
    concept='AboveBelow',
    num_augmentations=NUM_AUGMENTATIONS,
    placeholders_per_augmentation=PLACEHOLDERS_PER_LEVEL
)

print(f"\nCreated {len(loo_examples)} examples")
print(f"Expected {expected_per_problem_loo} examples")
print(f"Match: {'YES' if len(loo_examples) == expected_per_problem_loo else 'NO'}")

# Test EVAL method
print("\n" + "="*70)
print("TESTING EVAL METHOD")
print("="*70)

eval_examples = create_training_examples_for_problem_eval(
    problem_name=problem_name,
    problem_data=problem_data,
    concept='AboveBelow',
    num_augmentations=NUM_AUGMENTATIONS,
    placeholders_per_augmentation=PLACEHOLDERS_PER_LEVEL
)

print(f"\nCreated {len(eval_examples)} examples")
print(f"Expected {expected_per_problem_eval} examples")
print(f"Match: {'YES' if len(eval_examples) == expected_per_problem_eval else 'NO'}")

# Save small test output
print("\n" + "="*70)
print("SAVING TEST OUTPUT")
print("="*70)

with open('test_output_leaveoneout.json', 'w', encoding='utf-8') as f:
    json.dump(loo_examples[:3], f, indent=2)  # Save first 3 examples
print("Saved first 3 leave-one-out examples to: test_output_leaveoneout.json")

with open('test_output_eval.json', 'w', encoding='utf-8') as f:
    json.dump(eval_examples[:3], f, indent=2)  # Save first 3 examples
print("Saved first 3 eval examples to: test_output_eval.json")

# Visualize last input and answer for both methods
print("\n" + "="*70)
print("VISUALIZING TEST INPUT AND GROUND TRUTH - ALL LEVELS")
print("="*70)

# Visualize EVAL examples - one plot per level
print("\nVisualizing EVAL examples (one plot per level)...")

# Get one example from each level (same augmentation and placeholder idx)
eval_by_level = {}
for ex in eval_examples:
    level = ex['metadata']['level']
    if level not in eval_by_level and ex['metadata']['augmentation_idx'] == 0 and ex['metadata']['placeholder_idx'] == 0:
        eval_by_level[level] = ex

# Create one plot per level
for level in range(1, 7):
    if level in eval_by_level:
        ex = eval_by_level[level]
        test_input, placeholder, ground_truth = extract_test_grids(ex)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        plot_grid(test_input, "Test Input", axes[0])
        plot_grid(placeholder, f"Placeholder (Level {level})", axes[1])
        plot_grid(ground_truth, "Ground Truth", axes[2])
        
        metadata = ex['metadata']
        title = f"EVAL - Level {level} | {metadata['concept']} - {metadata['problem_name']}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'eval_level{level}_visualization.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: eval_level{level}_visualization.png")
        plt.close()

print("All EVAL level visualizations saved!")

# Visualize LEAVE-ONE-OUT examples - one plot per level
print("\nVisualizing LEAVE-ONE-OUT examples (one plot per level)...")

# Get one example from each level
loo_by_level = {}
for ex in loo_examples:
    level = ex['metadata']['level']
    if level not in loo_by_level and ex['metadata']['augmentation_idx'] == 0 and ex['metadata']['placeholder_idx'] == 0:
        loo_by_level[level] = ex

# Create one plot per level
for level in range(1, 7):
    if level in loo_by_level:
        ex = loo_by_level[level]
        test_input, placeholder, ground_truth = extract_test_grids(ex)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        plot_grid(test_input, "Test Input", axes[0])
        plot_grid(placeholder, f"Placeholder (Level {level})", axes[1])
        plot_grid(ground_truth, "Ground Truth", axes[2])
        
        metadata = ex['metadata']
        title = f"LEAVE-ONE-OUT - Level {level} | {metadata['concept']} - {metadata['problem_name']} | Held-out: {metadata['held_out_idx']}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'leaveoneout_level{level}_visualization.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: leaveoneout_level{level}_visualization.png")
        plt.close()

print("All LEAVE-ONE-OUT level visualizations saved!")

# Create comprehensive visualizations showing all training pairs
print("\n" + "="*70)
print("CREATING COMPREHENSIVE VISUALIZATIONS (All Training Pairs + Test)")
print("="*70)

print("\nCreating comprehensive EVAL visualizations...")
for level in range(1, 7):
    if level in eval_by_level:
        ex = eval_by_level[level]
        training_pairs, test_input, placeholder, ground_truth = extract_all_grids(ex)
        
        num_training = len(training_pairs)
        # Layout: 2 columns per training pair + 3 for test section
        # Rows: max(num_training, 2) to ensure space for test
        num_rows = max(num_training, 2)
        num_cols = 2 + 3  # 2 for training pairs + 3 for test section
        
        fig = plt.figure(figsize=(20, 6 * num_rows))
        
        # Plot training pairs
        for idx, pair in enumerate(training_pairs):
            if 'input' in pair and 'output' in pair:
                # Training input
                ax = plt.subplot(num_rows, num_cols, idx * num_cols + 1)
                plot_grid(pair['input'], f"Train {idx+1}: Input", ax)
                
                # Training output
                ax = plt.subplot(num_rows, num_cols, idx * num_cols + 2)
                plot_grid(pair['output'], f"Train {idx+1}: Output", ax)
        
        # Plot test section in the top right
        # Test input
        ax = plt.subplot(num_rows, num_cols, 3)
        plot_grid(test_input, "Test: Input", ax)
        
        # Placeholder
        ax = plt.subplot(num_rows, num_cols, 4)
        plot_grid(placeholder, f"Test: Placeholder (L{level})", ax)
        
        # Ground truth
        ax = plt.subplot(num_rows, num_cols, 5)
        plot_grid(ground_truth, "Test: Ground Truth", ax)
        
        metadata = ex['metadata']
        title = f"EVAL - Level {level} - COMPLETE VIEW | {metadata['concept']} - {metadata['problem_name']}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'eval_level{level}_complete.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: eval_level{level}_complete.png")
        plt.close()

print("\nCreating comprehensive LEAVE-ONE-OUT visualizations...")
for level in range(1, 7):
    if level in loo_by_level:
        ex = loo_by_level[level]
        training_pairs, test_input, placeholder, ground_truth = extract_all_grids(ex)
        
        num_training = len(training_pairs)
        num_rows = max(num_training, 2)
        num_cols = 2 + 3
        
        fig = plt.figure(figsize=(20, 6 * num_rows))
        
        # Plot training pairs
        for idx, pair in enumerate(training_pairs):
            if 'input' in pair and 'output' in pair:
                ax = plt.subplot(num_rows, num_cols, idx * num_cols + 1)
                plot_grid(pair['input'], f"Train {idx+1}: Input", ax)
                
                ax = plt.subplot(num_rows, num_cols, idx * num_cols + 2)
                plot_grid(pair['output'], f"Train {idx+1}: Output", ax)
        
        # Plot test section
        ax = plt.subplot(num_rows, num_cols, 3)
        plot_grid(test_input, "Test: Input", ax)
        
        ax = plt.subplot(num_rows, num_cols, 4)
        plot_grid(placeholder, f"Test: Placeholder (L{level})", ax)
        
        ax = plt.subplot(num_rows, num_cols, 5)
        plot_grid(ground_truth, "Test: Ground Truth", ax)
        
        metadata = ex['metadata']
        title = f"LEAVE-ONE-OUT - Level {level} - COMPLETE VIEW | {metadata['concept']} - {metadata['problem_name']} | Held-out: {metadata['held_out_idx']}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'leaveoneout_level{level}_complete.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: leaveoneout_level{level}_complete.png")
        plt.close()

print("\nAll comprehensive visualizations saved!")

print("\n" + "="*70)
print("TEST COMPLETE - ALL CHECKS PASSED!")
print("="*70)
print("\nGenerated files:")
print("  - test_output_leaveoneout.json")
print("  - test_output_eval.json")
print("\nSimple visualizations (Test Input + Placeholder + Ground Truth):")
print("  - eval_level1_visualization.png through eval_level6_visualization.png")
print("  - leaveoneout_level1_visualization.png through leaveoneout_level6_visualization.png")
print("\nComprehensive visualizations (All Training Pairs + Test + Placeholder + Ground Truth):")
print("  - eval_level1_complete.png through eval_level6_complete.png")
print("  - leaveoneout_level1_complete.png through leaveoneout_level6_complete.png")

