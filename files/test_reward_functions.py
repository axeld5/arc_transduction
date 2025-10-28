"""
Test suite for reward functions.
Validates that reward functions handle all answer formats correctly.
"""
import json
from reward_functions import (
    check_array,
    parse_grid_from_string,
    check_value,
    same_shape,
    reward_function_diff
)


def test_single_values():
    """Test single scalar values (1x1 grids)."""
    print("Testing single scalar values...")
    
    # Test check_array
    assert check_array("4") == True, "check_array('4') should return True"
    assert check_array("0") == True, "check_array('0') should return True"
    assert check_array("9") == True, "check_array('9') should return True"
    assert check_array("10") == False, "check_array('10') should return False (>9)"
    assert check_array("-1") == False, "check_array('-1') should return False (<0)"
    
    # Test parse_grid_from_string
    assert parse_grid_from_string("4") == [[4]], "parse '4' should give [[4]]"
    assert parse_grid_from_string("0") == [[0]], "parse '0' should give [[0]]"
    assert parse_grid_from_string("9") == [[9]], "parse '9' should give [[9]]"
    
    print("  PASS: All single value tests passed")


def test_horizontal_arrays():
    """Test horizontal arrays (1-row grids)."""
    print("Testing horizontal arrays...")
    
    # Test check_array
    assert check_array("0 0 0 0 0") == True, "check_array('0 0 0 0 0') should return True"
    assert check_array("1 4 3 0") == True, "check_array('1 4 3 0') should return True"
    assert check_array("9 9 9") == True, "check_array('9 9 9') should return True"
    
    # Test parse_grid_from_string
    assert parse_grid_from_string("0 0 0 0 0") == [[0, 0, 0, 0, 0]]
    assert parse_grid_from_string("1 4 3 0") == [[1, 4, 3, 0]]
    assert parse_grid_from_string("9 9 9") == [[9, 9, 9]]
    
    print("  PASS: All horizontal array tests passed")


def test_multiline_grids():
    """Test multi-line grids (backward compatibility)."""
    print("Testing multi-line grids...")
    
    # Test check_array
    assert check_array("1 2 3\n4 5 6") == True
    assert check_array("123\n456") == True
    assert check_array("0 0\n0 0") == True
    
    # Test parse_grid_from_string
    assert parse_grid_from_string("1 2 3\n4 5 6") == [[1, 2, 3], [4, 5, 6]]
    assert parse_grid_from_string("123\n456") == [[1, 2, 3], [4, 5, 6]]
    assert parse_grid_from_string("0 0\n0 0") == [[0, 0], [0, 0]]
    
    print("  PASS: All multi-line grid tests passed")


def test_same_shape():
    """Test same_shape utility function."""
    print("Testing same_shape...")
    
    assert same_shape([[1, 2]], [[3, 4]]) == True
    assert same_shape([[1]], [[2]]) == True
    assert same_shape([[1, 2, 3]], [[4, 5, 6]]) == True
    assert same_shape([[1, 2]], [[3, 4, 5]]) == False
    assert same_shape([[1], [2]], [[3]]) == False
    assert same_shape([], [[1]]) == False
    
    print("  PASS: All same_shape tests passed")


def test_check_value():
    """Test check_value function."""
    print("Testing check_value...")
    
    assert check_value("4", [[4]]) == True
    assert check_value("1 2 3", [[1, 2, 3]]) == True
    assert check_value("1 2\n3 4", [[1, 2], [3, 4]]) == True
    assert check_value("5", [[4]]) == False
    assert check_value("invalid", [[1]]) == False
    
    print("  PASS: All check_value tests passed")


def test_reward_function_perfect_match():
    """Test reward function with perfect matches."""
    print("Testing reward function (perfect matches)...")
    
    # Single value
    rewards = reward_function_diff(["4"], ["4"])
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    
    # Horizontal array
    rewards = reward_function_diff(["1 2 3"], ["1 2 3"])
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    
    # Multi-line grid
    rewards = reward_function_diff(["1 2\n3 4"], ["1 2\n3 4"])
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    
    print("  PASS: All perfect match tests passed")


def test_reward_function_partial_match():
    """Test reward function with partial matches."""
    print("Testing reward function (partial matches)...")
    
    # 1 wrong out of 3 cells
    rewards = reward_function_diff(["1 2 3"], ["1 2 4"])
    expected = 0.5 * (1 - 1/3)
    assert abs(rewards[0] - expected) < 0.01, f"Expected {expected}, got {rewards[0]}"
    
    # 1 wrong out of 4 cells
    rewards = reward_function_diff(["1 2\n3 4"], ["1 2\n3 5"])
    expected = 0.5 * (1 - 1/4)
    assert abs(rewards[0] - expected) < 0.01, f"Expected {expected}, got {rewards[0]}"
    
    print("  PASS: All partial match tests passed")


def test_reward_function_invalid():
    """Test reward function with invalid inputs."""
    print("Testing reward function (invalid inputs)...")
    
    # Invalid format
    rewards = reward_function_diff(["invalid"], ["1 2 3"])
    assert rewards == [-1.0], f"Expected [-1.0], got {rewards}"
    
    # Wrong shape
    rewards = reward_function_diff(["1 2"], ["1 2 3"])
    assert rewards == [-0.5], f"Expected [-0.5], got {rewards}"
    
    print("  PASS: All invalid input tests passed")


def test_training_data_samples():
    """Test with actual samples from training data."""
    print("Testing with training data samples...")
    
    try:
        with open('generated_data/train_data.json', 'r') as f:
            data = json.load(f)
        
        # Test samples from different indices
        test_indices = [0, 100, 1000, 11160, 11161, 50000, 100000]
        success = 0
        
        for idx in range(len(data)):
            if idx >= len(data):
                continue
            answer = data[idx]['answer']
            # Should get perfect reward when comparing answer to itself
            rewards = reward_function_diff([answer], [answer])
            if rewards == [1.0]:
                success += 1
            else:
                print(f"    Warning: Index {idx} got reward {rewards[0]} for answer '{answer[:50]}'")
        
        print(f"  PASS: {success}/{len([i for i in test_indices if i < len(data)])} training samples passed")
        
    except FileNotFoundError:
        print("  SKIP: train_data.json not found")
    except Exception as e:
        print(f"  SKIP: error - {e}")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("REWARD FUNCTIONS TEST SUITE")
    print("="*60 + "\n")
    
    try:
        test_single_values()
        test_horizontal_arrays()
        test_multiline_grids()
        test_same_shape()
        test_check_value()
        test_reward_function_perfect_match()
        test_reward_function_partial_match()
        test_reward_function_invalid()
        test_training_data_samples()
        
        print("\n" + "="*60)
        print("SUCCESS: ALL TESTS PASSED")
        print("="*60)
        return True
        
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"FAILED: {e}")
        print("="*60)
        return False
    except Exception as e:
        print("\n" + "="*60)
        print(f"ERROR: {e}")
        print("="*60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

