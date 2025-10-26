"""Creates placeholder solutions for a problem with 6 difficulty levels"""

from typing import Dict, Any, List
import random


def create_level_1_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]]
) -> List[List[int]]:
    """
    Level 1: Ground truth or input matrix modified by random pixels.
    
    Can randomly choose to start with the input matrix or ground truth,
    then modify 0 to max(width*2, 10) random pixels.
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
    
    Returns:
        Modified matrix as list of lists
    """
    target_height = len(ground_truth)
    target_width = len(ground_truth[0]) if target_height > 0 else 0
    
    # Randomly decide whether to use input matrix or ground truth
    use_input = random.random() < 0.5
    
    if use_input and problem_data.get('test'):
        # Use the input matrix from the test example
        test_input = problem_data['test'][0]['input']
        # If input dimensions match, use it; otherwise fall back to ground truth
        if len(test_input) > 0 and len(test_input[0]) > 0:
            placeholder_matrix = [[cell for cell in row] for row in test_input]
            target_height = len(placeholder_matrix)
            target_width = len(placeholder_matrix[0])
        else:
            placeholder_matrix = [[cell for cell in row] for row in ground_truth]
    else:
        placeholder_matrix = [[cell for cell in row] for row in ground_truth]
    
    # Modify random pixels
    max_modifications = max(target_width * 2, 10)
    num_modifications = random.randint(0, max_modifications)
    total_pixels = target_height * target_width
    num_modifications = min(num_modifications, total_pixels)
    
    if num_modifications > 0:
        positions = [(i, j) for i in range(target_height) for j in range(target_width)]
        positions_to_modify = random.sample(positions, num_modifications)
        for i, j in positions_to_modify:
            placeholder_matrix[i][j] = random.randint(0, 9)
    
    return placeholder_matrix


def create_level_2_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]]
) -> List[List[int]]:
    """
    Level 2: Ground truth with random addition OR removal of up to 2 rows and/or up to 2 columns.
    
    50% chance to add, 50% chance to remove.
    At least one row OR one column must be added/removed.
    Added cells are filled with zeros.
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
    
    Returns:
        Modified matrix as list of lists
    """
    target_height = len(ground_truth)
    target_width = len(ground_truth[0]) if target_height > 0 else 0
    
    # 50/50 chance to add or remove
    should_add = random.random() < 0.5
    
    if should_add:
        # ADD rows and/or columns
        rows_to_add = random.randint(0, 2)
        cols_to_add = random.randint(0, 2)
        
        # Ensure at least one row OR one column is added
        if rows_to_add == 0 and cols_to_add == 0:
            if random.random() < 0.5:
                rows_to_add = random.randint(1, 2)
            else:
                cols_to_add = random.randint(1, 2)
        
        new_height = target_height + rows_to_add
        new_width = target_width + cols_to_add
        
        # Create expanded matrix filled with zeros
        placeholder_matrix = [[0] * new_width for _ in range(new_height)]
        
        # Copy ground truth into the expanded matrix (at a random position)
        if rows_to_add > 0:
            start_row = random.randint(0, rows_to_add)
        else:
            start_row = 0
        
        if cols_to_add > 0:
            start_col = random.randint(0, cols_to_add)
        else:
            start_col = 0
        
        # Copy ground truth values
        for i in range(target_height):
            for j in range(target_width):
                placeholder_matrix[start_row + i][start_col + j] = ground_truth[i][j]
    
    else:
        # REMOVE rows and/or columns
        max_rows_to_remove = min(2, target_height - 1)  # Keep at least 1 row
        max_cols_to_remove = min(2, target_width - 1)   # Keep at least 1 col
        
        rows_to_remove = random.randint(0, max_rows_to_remove)
        cols_to_remove = random.randint(0, max_cols_to_remove)
        
        # Ensure at least one row OR one column is removed
        if rows_to_remove == 0 and cols_to_remove == 0:
            if max_rows_to_remove > 0 and max_cols_to_remove > 0:
                # Both are possible, randomly pick one
                if random.random() < 0.5:
                    rows_to_remove = random.randint(1, max_rows_to_remove)
                else:
                    cols_to_remove = random.randint(1, max_cols_to_remove)
            elif max_rows_to_remove > 0:
                rows_to_remove = random.randint(1, max_rows_to_remove)
            elif max_cols_to_remove > 0:
                cols_to_remove = random.randint(1, max_cols_to_remove)
            else:
                # Both dimensions are 1, can't remove anything - fallback
                return [[0]]
        
        new_height = target_height - rows_to_remove
        new_width = target_width - cols_to_remove
        
        placeholder_matrix = [[ground_truth[i][j] for j in range(new_width)] 
                              for i in range(new_height)]
    
    return placeholder_matrix


def create_level_3_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]]
) -> List[List[int]]:
    """
    Level 3: Random crop with dimensions below 7x7.
    
    With 50% probability, the crop is upscaled to a size up to 30x30 by padding with zeros.
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
    
    Returns:
        Cropped (and possibly upscaled) matrix as list of lists
    """
    target_height = len(ground_truth)
    target_width = len(ground_truth[0]) if target_height > 0 else 0
    
    # Create random crop with dimensions below 7x7
    max_dim = 7
    crop_height = random.randint(1, min(max_dim - 1, target_height))
    crop_width = random.randint(1, min(max_dim - 1, target_width))
    
    # Random crop position
    if target_height > crop_height:
        start_row = random.randint(0, target_height - crop_height)
    else:
        start_row = 0
    
    if target_width > crop_width:
        start_col = random.randint(0, target_width - crop_width)
    else:
        start_col = 0
    
    end_row = start_row + crop_height
    end_col = start_col + crop_width
    
    cropped_matrix = [[ground_truth[i][j] for j in range(start_col, end_col)] 
                      for i in range(start_row, end_row)]
    
    # 50% chance to upscale the crop
    if random.random() < 0.5:
        # Upscale to a random size up to 30x30
        max_upscale = 30
        new_height = random.randint(crop_height, min(max_upscale, crop_height + 23))
        new_width = random.randint(crop_width, min(max_upscale, crop_width + 23))
        
        # Create larger matrix filled with zeros
        upscaled_matrix = [[0] * new_width for _ in range(new_height)]
        
        # Place crop at random position in the upscaled matrix
        if new_height > crop_height:
            start_row_offset = random.randint(0, new_height - crop_height)
        else:
            start_row_offset = 0
        
        if new_width > crop_width:
            start_col_offset = random.randint(0, new_width - crop_width)
        else:
            start_col_offset = 0
        
        # Copy cropped content
        for i in range(crop_height):
            for j in range(crop_width):
                upscaled_matrix[start_row_offset + i][start_col_offset + j] = cropped_matrix[i][j]
        
        return upscaled_matrix
    
    return cropped_matrix


def create_level_4_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]]
) -> List[List[int]]:
    """
    Level 4: Simple 3x3 zeros matrix.
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
    
    Returns:
        3x3 zeros matrix as list of lists
    """
    return [[0] * 3 for _ in range(3)]


def create_level_5_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]]
) -> List[List[int]]:
    """
    Level 5: Random matrix with same dimensions as ground truth.
    
    Each cell is filled with a random value from 0-9.
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
    
    Returns:
        Random matrix as list of lists
    """
    target_height = len(ground_truth)
    target_width = len(ground_truth[0]) if target_height > 0 else 0
    
    return [[random.randint(0, 9) for _ in range(target_width)] 
            for _ in range(target_height)]


def create_level_6_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]]
) -> List[List[int]]:
    """
    Level 6: Random matrix selected from problem data.
    
    Randomly selects a matrix from the problem's train/test/arc-gen examples.
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
    
    Returns:
        Random matrix from problem data as list of lists
    """
    target_height = len(ground_truth)
    target_width = len(ground_truth[0]) if target_height > 0 else 0
    
    all_matrices = []
    
    # Collect all matrices from train examples
    for example in problem_data.get('train', []):
        all_matrices.append(example['input'])
        all_matrices.append(example['output'])
    
    # Collect from test examples (inputs only)
    for example in problem_data.get('test', []):
        all_matrices.append(example['input'])
    
    # Collect from arc-gen if available
    for example in problem_data.get('arc-gen', []):
        all_matrices.append(example['input'])
        all_matrices.append(example['output'])
    
    if all_matrices:
        chosen_matrix = random.choice(all_matrices)
        return [[cell for cell in row] for row in chosen_matrix]
    else:
        # Fallback if no matrices available
        return [[0] * target_width for _ in range(target_height)]


def create_placeholder(
    problem_data: Dict[str, Any], 
    ground_truth: List[List[int]], 
    level: int = 1
) -> List[List[int]]:
    """
    Creates a placeholder solution based on difficulty level (1-6).
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
        level: Difficulty level (1-6)
            1: Input matrix or ground truth modified by random pixels (0 to max(width*2, 10))
            2: Ground truth with 50/50 random addition OR removal of up to 2 rows and/or up to 2 columns (at least one dimension changed)
            3: Random crop with dimensions below 7x7, with 50% chance of upscaling to up to 30x30 with zero padding
            4: 3x3 zeros matrix
            5: Random matrix (same size as ground truth)
            6: Random matrix from problem data
    
    Returns:
        Placeholder matrix as list of lists
    """
    if level == 1:
        return create_level_1_placeholder(problem_data, ground_truth)
    elif level == 2:
        return create_level_2_placeholder(problem_data, ground_truth)
    elif level == 3:
        return create_level_3_placeholder(problem_data, ground_truth)
    elif level == 4:
        return create_level_4_placeholder(problem_data, ground_truth)
    elif level == 5:
        return create_level_5_placeholder(problem_data, ground_truth)
    elif level == 6:
        return create_level_6_placeholder(problem_data, ground_truth)
    else:
        raise ValueError(f"Invalid level: {level}. Must be between 1 and 6.")