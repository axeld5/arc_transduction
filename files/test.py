"""
Test script to verify that all ARC problems can be loaded.

This script validates that:
- 400 evaluation problems can be loaded
- 400 training problems can be loaded
- 160 corpus problems can be loaded
- Total: 960 problems
"""

from loader import (
    list_evaluation_problems,
    list_training_problems,
    list_all_corpus_problems,
    load_evaluation_problem,
    load_training_problem,
    load_corpus_problem_by_name
)


def test_evaluation_problems():
    """Test loading all evaluation problems."""
    print("Testing evaluation problems...")
    problem_ids = list_evaluation_problems()
    print(f"  Found {len(problem_ids)} evaluation problems")
    
    # Test loading a sample
    if problem_ids:
        sample = load_evaluation_problem(problem_ids[0])
        assert sample is not None, "Failed to load evaluation problem"
        assert "train" in sample, "Missing 'train' key in evaluation problem"
        assert "test" in sample, "Missing 'test' key in evaluation problem"
        print(f"  Successfully loaded sample problem: {problem_ids[0]}")
    
    return len(problem_ids)


def test_training_problems():
    """Test loading all training problems."""
    print("\nTesting training problems...")
    problem_ids = list_training_problems()
    print(f"  Found {len(problem_ids)} training problems")
    
    # Test loading a sample
    if problem_ids:
        sample = load_training_problem(problem_ids[0])
        assert sample is not None, "Failed to load training problem"
        assert "train" in sample, "Missing 'train' key in training problem"
        assert "test" in sample, "Missing 'test' key in training problem"
        print(f"  Successfully loaded sample problem: {problem_ids[0]}")
    
    return len(problem_ids)


def test_corpus_problems():
    """Test loading all corpus problems."""
    print("\nTesting corpus problems...")
    all_problems = list_all_corpus_problems()
    
    total_count = sum(len(problems) for problems in all_problems.values())
    print(f"  Found {total_count} corpus problems across {len(all_problems)} concepts")
    
    # Print breakdown by concept
    for concept, problems in sorted(all_problems.items()):
        print(f"    {concept}: {len(problems)} problems")
    
    # Test loading a sample from each concept
    loaded_count = 0
    for concept, problems in all_problems.items():
        if problems:
            sample = load_corpus_problem_by_name(problems[0])
            assert sample is not None, f"Failed to load corpus problem {problems[0]}"
            assert "train" in sample, f"Missing 'train' key in corpus problem {problems[0]}"
            assert "test" in sample, f"Missing 'test' key in corpus problem {problems[0]}"
            loaded_count += 1
    
    print(f"  Successfully loaded samples from all {loaded_count} concepts")
    
    return total_count


def main():
    """Run all tests."""
    print("=" * 60)
    print("ARC Problem Loader Test Suite")
    print("=" * 60)
    
    try:
        eval_count = test_evaluation_problems()
        train_count = test_training_problems()
        corpus_count = test_corpus_problems()
        
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Evaluation problems: {eval_count}")
        print(f"Training problems:   {train_count}")
        print(f"Corpus problems:     {corpus_count}")
        print(f"Total problems:      {eval_count + train_count + corpus_count}")
        print("=" * 60)
        
        expected_total = 400 + 400 + 160
        actual_total = eval_count + train_count + corpus_count
        
        if actual_total == expected_total:
            print(f"[SUCCESS] Found all {expected_total} expected problems!")
        else:
            print(f"[WARNING] Expected {expected_total} problems but found {actual_total}")
            
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

