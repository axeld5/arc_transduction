"""
Statistics analysis for ARC problems.

This script analyzes:
- Mean input/output character length for each problem
- Distribution of 0.25 * input_size + 0.75 * output_size
- Generates histogram visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from loader import (
    list_evaluation_problems,
    list_training_problems,
    list_all_corpus_problems,
    load_evaluation_problem,
    load_training_problem,
    load_corpus_problem_by_name
)


def count_characters_in_grid(grid: List[List[int]]) -> int:
    """Count the total number of characters needed to represent a grid."""
    # Convert grid to string representation and count characters
    # Each cell is a digit, we count all digits
    return sum(len(str(cell)) for row in grid for cell in row)


def analyze_problem(problem_data: Dict) -> Dict[str, float]:
    """
    Analyze a single problem and compute statistics.
    
    Returns:
        Dict with 'input_chars', 'output_chars', and 'combined_score'
    """
    total_input_chars = 0
    total_output_chars = 0
    example_count = 0
    
    # Analyze training examples
    for example in problem_data.get("train", []):
        if "input" in example:
            total_input_chars += count_characters_in_grid(example["input"])
        if "output" in example:
            total_output_chars += count_characters_in_grid(example["output"])
        example_count += 1
    
    # Analyze test examples
    for example in problem_data.get("test", []):
        if "input" in example:
            total_input_chars += count_characters_in_grid(example["input"])
        if "output" in example:
            total_output_chars += count_characters_in_grid(example["output"])
        example_count += 1
    
    # Calculate means
    mean_input = total_input_chars / example_count if example_count > 0 else 0
    mean_output = total_output_chars / example_count if example_count > 0 else 0
    
    # Calculate combined score: 0.25 * input + 0.75 * output
    combined_score = 0.25 * mean_input + 0.75 * mean_output
    
    return {
        "input_chars": mean_input,
        "output_chars": mean_output,
        "combined_score": combined_score,
        "example_count": example_count
    }


def collect_statistics() -> Tuple[List[Dict], List[str]]:
    """
    Collect statistics for all problems.
    
    Returns:
        Tuple of (statistics_list, problem_labels)
    """
    all_stats = []
    problem_labels = []
    
    print("Collecting statistics...")
    
    # Process evaluation problems
    print("  Processing evaluation problems...")
    eval_problems = list_evaluation_problems()
    for i, problem_id in enumerate(eval_problems):
        if i % 50 == 0:
            print(f"    {i}/{len(eval_problems)}...")
        try:
            problem = load_evaluation_problem(problem_id)
            stats = analyze_problem(problem)
            all_stats.append(stats)
            problem_labels.append(f"eval_{problem_id}")
        except Exception as e:
            print(f"    Warning: Failed to process {problem_id}: {e}")
    
    # Process training problems
    print("  Processing training problems...")
    train_problems = list_training_problems()
    for i, problem_id in enumerate(train_problems):
        if i % 50 == 0:
            print(f"    {i}/{len(train_problems)}...")
        try:
            problem = load_training_problem(problem_id)
            stats = analyze_problem(problem)
            all_stats.append(stats)
            problem_labels.append(f"train_{problem_id}")
        except Exception as e:
            print(f"    Warning: Failed to process {problem_id}: {e}")
    
    # Process corpus problems
    print("  Processing corpus problems...")
    corpus_problems = list_all_corpus_problems()
    for concept, problems in sorted(corpus_problems.items()):
        for problem_name in problems:
            try:
                problem = load_corpus_problem_by_name(problem_name)
                stats = analyze_problem(problem)
                all_stats.append(stats)
                problem_labels.append(f"corpus_{problem_name}")
            except Exception as e:
                print(f"    Warning: Failed to process {problem_name}: {e}")
    
    print(f"  Collected statistics for {len(all_stats)} problems")
    
    return all_stats, problem_labels


def plot_statistics(all_stats: List[Dict], problem_labels: List[str]):
    """Create and save histogram visualizations."""
    
    # Extract data
    input_chars = [s["input_chars"] for s in all_stats]
    output_chars = [s["output_chars"] for s in all_stats]
    combined_scores = [s["combined_score"] for s in all_stats]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ARC Problems Character Count Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: Input character distribution
    ax1 = axes[0, 0]
    ax1.hist(input_chars, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Mean Input Characters per Example')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Input Character Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(input_chars), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(input_chars):.1f}')
    ax1.legend()
    
    # Plot 2: Output character distribution
    ax2 = axes[0, 1]
    ax2.hist(output_chars, bins=50, color='darkseagreen', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean Output Characters per Example')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Output Character Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(output_chars), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(output_chars):.1f}')
    ax2.legend()
    
    # Plot 3: Combined score distribution (0.25 * input + 0.75 * output)
    ax3 = axes[1, 0]
    ax3.hist(combined_scores, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Combined Score (0.25×Input + 0.75×Output)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Combined Score Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(np.mean(combined_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(combined_scores):.1f}')
    ax3.legend()
    
    # Plot 4: Scatter plot of input vs output
    ax4 = axes[1, 1]
    ax4.scatter(input_chars, output_chars, alpha=0.5, s=20, color='purple')
    ax4.set_xlabel('Mean Input Characters')
    ax4.set_ylabel('Mean Output Characters')
    ax4.set_title('Input vs Output Character Counts')
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    max_val = max(max(input_chars), max(output_chars))
    ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Input = Output')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'arc_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    # Close the plot (comment out to display interactively)
    plt.close()


def print_summary_statistics(all_stats: List[Dict], problem_labels: List[str]):
    """Print summary statistics to console."""
    
    input_chars = [s["input_chars"] for s in all_stats]
    output_chars = [s["output_chars"] for s in all_stats]
    combined_scores = [s["combined_score"] for s in all_stats]
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    print(f"\nTotal problems analyzed: {len(all_stats)}")
    
    print("\nInput Characters:")
    print(f"  Mean:   {np.mean(input_chars):.2f}")
    print(f"  Median: {np.median(input_chars):.2f}")
    print(f"  Std:    {np.std(input_chars):.2f}")
    print(f"  Min:    {np.min(input_chars):.2f}")
    print(f"  Max:    {np.max(input_chars):.2f}")
    
    print("\nOutput Characters:")
    print(f"  Mean:   {np.mean(output_chars):.2f}")
    print(f"  Median: {np.median(output_chars):.2f}")
    print(f"  Std:    {np.std(output_chars):.2f}")
    print(f"  Min:    {np.min(output_chars):.2f}")
    print(f"  Max:    {np.max(output_chars):.2f}")
    
    print("\nCombined Score (0.25×Input + 0.75×Output):")
    print(f"  Mean:   {np.mean(combined_scores):.2f}")
    print(f"  Median: {np.median(combined_scores):.2f}")
    print(f"  Std:    {np.std(combined_scores):.2f}")
    print(f"  Min:    {np.min(combined_scores):.2f}")
    print(f"  Max:    {np.max(combined_scores):.2f}")
    
    # Find top 5 problems by combined score
    print("\nTop 5 Problems by Combined Score:")
    sorted_indices = np.argsort(combined_scores)[::-1]
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  {i+1}. {problem_labels[idx]}: {combined_scores[idx]:.2f}")
    
    print("=" * 60)


def save_detailed_statistics(all_stats: List[Dict], problem_labels: List[str]):
    """Save detailed statistics to a JSON file."""
    
    detailed_data = []
    for label, stats in zip(problem_labels, all_stats):
        detailed_data.append({
            "problem": label,
            "mean_input_chars": stats["input_chars"],
            "mean_output_chars": stats["output_chars"],
            "combined_score": stats["combined_score"],
            "example_count": stats["example_count"]
        })
    
    output_file = 'arc_statistics_detailed.json'
    with open(output_file, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"Saved detailed statistics to: {output_file}")


def main():
    """Main function to run statistics analysis."""
    print("=" * 60)
    print("ARC Problems Statistics Analysis")
    print("=" * 60)
    
    try:
        # Collect statistics
        all_stats, problem_labels = collect_statistics()
        
        # Print summary
        print_summary_statistics(all_stats, problem_labels)
        
        # Save detailed statistics
        save_detailed_statistics(all_stats, problem_labels)
        
        # Plot visualizations
        print("\nGenerating visualizations...")
        plot_statistics(all_stats, problem_labels)
        
        print("\n[SUCCESS] Analysis complete!")
        
    except Exception as e:
        print(f"\n[FAILED] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

