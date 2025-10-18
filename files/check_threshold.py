"""
Quick script to check how many problems would be included at different thresholds.
"""

import json

def analyze_thresholds():
    """Analyze how many problems would be included at various thresholds."""
    
    with open('arc_statistics_detailed.json', 'r') as f:
        stats = json.load(f)
    
    # Separate by dataset type
    train_scores = []
    eval_scores = []
    
    for entry in stats:
        score = entry['combined_score']
        if entry['problem'].startswith('train_'):
            train_scores.append(score)
        elif entry['problem'].startswith('eval_'):
            eval_scores.append(score)
    
    # Test different thresholds
    thresholds = [100, 120, 140, 160, 180, 200, 250, 300]
    
    print("=" * 70)
    print("Problem Count by Combined Score Threshold")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Train':<15} {'Eval':<15} {'Total':<15} {'% of Total':<12}")
    print("-" * 70)
    
    for threshold in thresholds:
        train_count = sum(1 for s in train_scores if s < threshold)
        eval_count = sum(1 for s in eval_scores if s < threshold)
        total_count = train_count + eval_count
        percentage = (total_count / 800) * 100
        
        print(f"{threshold:<12} {train_count:>4}/400 ({train_count/4:>5.1f}%) {eval_count:>4}/400 ({eval_count/4:>5.1f}%) "
              f"{total_count:>4}/800 ({percentage:>5.1f}%)  {'<-- SELECTED' if threshold == 160 else ''}")
    
    print("-" * 70)
    print(f"{'No filter':<12} {'400/400 (100%)':<15} {'400/400 (100%)':<15} {'800/800 (100%)':<15}")
    print("=" * 70)
    
    # Show distribution statistics
    all_scores = train_scores + eval_scores
    print(f"\nCombined Score Statistics:")
    print(f"  Mean:   {sum(all_scores)/len(all_scores):.2f}")
    print(f"  Median: {sorted(all_scores)[len(all_scores)//2]:.2f}")
    print(f"  Min:    {min(all_scores):.2f}")
    print(f"  Max:    {max(all_scores):.2f}")
    print(f"\nCurrent threshold (160) is at the {(sum(1 for s in all_scores if s < 160)/len(all_scores)*100):.1f}th percentile")


if __name__ == "__main__":
    analyze_thresholds()

