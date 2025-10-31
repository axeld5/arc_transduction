"""
Reward functions for ARC transduction training.
Handles validation and reward calculation for grid-based outputs.
"""
import re
from typing import List, Optional, Any


def check_array(output_string: str) -> bool:
    """
    Check if output_string is a valid grid format.
    Supports:
    - Single digits: "4"
    - Horizontal arrays: "1 2 3 4"
    - Multi-line grids: "1 2 3\n4 5 6"
    """
    if not output_string or not isinstance(output_string, str):
        return False
    response = output_string.strip()
    if not response:
        return False
    
    # Handle single-line grids (including single values)
    if '\n' not in response:
        # Check if it's space-separated
        if ' ' in response:
            parts = response.split()
            try:
                grid_row = [int(p) for p in parts if p.strip()]
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return False
                return len(grid_row) > 0
            except ValueError:
                return False
        else:
            # Single digit value
            try:
                digit = int(response)
                return 0 <= digit <= 9
            except ValueError:
                return False
    
    # Handle multi-line grids
    grid_match = re.search(r'[0-9\n\s]+', response)
    if not grid_match:
        return False
    grid_str = grid_match.group()
    try:
        rows = grid_str.split('\n')
        if not rows:
            return False
        grid = []
        expected_width = None
        for row in rows:
            if not row.strip():
                return False
            parts = row.strip().split()
            if len(parts) > 1:
                try:
                    grid_row = [int(p) for p in parts if p.strip()]
                except ValueError:
                    return False
            else:
                if not row.strip().isdigit():
                    return False
                grid_row = [int(char) for char in row.strip()]
            if any(digit < 0 or digit > 9 for digit in grid_row):
                return False
            if expected_width is None:
                expected_width = len(grid_row)
            elif len(grid_row) != expected_width:
                return False
            grid.append(grid_row)
        return len(grid) > 0 and len(grid[0]) > 0
    except (ValueError, IndexError):
        return False


def parse_grid_from_string(output_string: str) -> Optional[List[List[int]]]:
    """
    Parse output_string into a 2D grid (list of lists).
    Supports:
    - Single digits: "4" -> [[4]]
    - Horizontal arrays: "1 2 3" -> [[1, 2, 3]]
    - Multi-line grids: "1 2 3\n4 5 6" -> [[1, 2, 3], [4, 5, 6]]
    """
    if not output_string or not isinstance(output_string, str):
        return None
    response = output_string.strip()
    if not response:
        return None
    
    # Handle single-line grids
    if '\n' not in response:
        # Check if it's space-separated
        if ' ' in response:
            parts = response.split()
            try:
                grid_row = [int(p) for p in parts if p.strip()]
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return None
                if not grid_row:
                    return None
                return [grid_row]  # Return as single-row grid
            except ValueError:
                return None
        else:
            # Single digit value
            try:
                digit = int(response)
                if 0 <= digit <= 9:
                    return [[digit]]  # Return as 1x1 grid
                return None
            except ValueError:
                return None
    
    # Handle multi-line grids
    grid_match = re.search(r'[0-9\n\s]+', response)
    if not grid_match:
        return None
    grid_str = grid_match.group()
    try:
        rows = grid_str.split('\n')
        grid = []
        for row in rows:
            if not row.strip():
                continue
            parts = row.strip().split()
            if len(parts) > 1:
                try:
                    grid_row = [int(p) for p in parts if p.strip()]
                except ValueError:
                    return None
            else:
                if not row.strip().isdigit():
                    return None
                grid_row = [int(char) for char in row.strip()]
            if any(digit < 0 or digit > 9 for digit in grid_row):
                return None
            grid.append(grid_row)
        return grid if grid else None
    except (ValueError, IndexError):
        return None


def check_value(output_string: str, expected_value: List[List[int]]) -> bool:
    """
    Check if output_string matches the expected value grid.
    
    Args:
        output_string: The string representation of the grid
        expected_value: The expected grid as a 2D list
        
    Returns:
        True if the parsed grid matches expected_value, False otherwise
    """
    if not isinstance(expected_value, list) or not expected_value:
        return False
    if not check_array(output_string):
        return False
    parsed_grid = parse_grid_from_string(output_string)
    if parsed_grid is None:
        return False
    return parsed_grid == expected_value


def same_shape(a: List[List[int]], b: List[List[int]]) -> bool:
    """
    Check if two grids have the same shape.
    
    Args:
        a: First grid
        b: Second grid
        
    Returns:
        True if both grids have the same dimensions, False otherwise
    """
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) for ra, rb in zip(a, b))


def reward_function_diff(
    completions: List[str],
    expected_output: List[str],
    use_dense_reward: bool = True,
    **kwargs: Any
) -> List[float]:
    """
    Reward function based on cell-wise accuracy.
    Returns 1.0 for perfect match, proportional reward for partial match (if dense).
    
    Args:
        completions: List of completion strings from the model
        expected_output: List of expected output strings
        use_dense_reward: If True, gives partial credit (0.5 * accuracy) for correct shape.
                         If False, only gives full reward for perfect match.
        **kwargs: Additional arguments (unused)
        
    Returns:
        List of reward values for each completion
        
    Reward Scale:
        Dense (use_dense_reward=True):
            1.0: Perfect match
            0.5 * (1 - diff_ratio): Partial match (correct shape, some wrong cells)
            -0.5: Wrong shape or unparseable but valid array
            -1.0: Invalid array format
        
        Discrete (use_dense_reward=False):
            1.0: Perfect match
            -0.5: Wrong shape or unparseable but valid array
            -1.0: Invalid array format
    """
    rewards: List[float] = []
    for completion, expected in zip(completions, expected_output, strict=False):
        value = completion[0]["content"] if isinstance(completion, list) else completion
        if not check_array(value):
            rewards.append(-1.0)
            continue
        comp_grid = parse_grid_from_string(value)
        exp_grid = parse_grid_from_string(expected)
        if comp_grid is None or exp_grid is None or not same_shape(comp_grid, exp_grid):
            rewards.append(-0.5)
            continue
        rows = len(exp_grid)
        cols = len(exp_grid[0]) if rows else 0
        diffs = 0
        for r in range(rows):
            for c in range(cols):
                if comp_grid[r][c] != exp_grid[r][c]:
                    diffs += 1
        if diffs != 0:
            # Dense reward: give partial credit for partially correct grids
            # Discrete reward: treat as wrong (-0.5 already set for wrong shape cases)
            if use_dense_reward:
                rewards.append(0.5 * (1 - diffs / (rows * cols)))
            else:
                rewards.append(-0.5)
        else:
            rewards.append(1.0)
    return rewards


class DynamicRewardFunction:
    """
    Wrapper for reward function that switches from dense to discrete rewards.
    First 1/4 of training uses dense rewards, remaining 3/4 uses discrete.
    """
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.dense_threshold = max_steps // 4
        self.current_step = 0
        
    def __call__(
        self,
        completions: List[str],
        expected_output: List[str],
        **kwargs: Any
    ) -> List[float]:
        """
        Call the reward function with appropriate dense/discrete setting.
        """
        use_dense = self.current_step < self.dense_threshold
        self.current_step += 1
        
        if self.current_step == 1:
            print(f"[Reward] Starting with DENSE rewards (steps 0-{self.dense_threshold})")
        elif self.current_step == self.dense_threshold + 1:
            print(f"[Reward] Switching to DISCRETE rewards (steps {self.dense_threshold + 1}-{self.max_steps})")
        
        return reward_function_diff(completions, expected_output, use_dense_reward=use_dense, **kwargs)

