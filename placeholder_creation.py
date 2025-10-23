"""Creates placeholder solutions for a problem with 6 difficulty levels"""

from typing import Dict, Any, List
import random

def create_placeholder(problem_data: Dict[str, Any], ground_truth: List[List[int]], level: int = 1) -> List[List[int]]:
    """
    Creates a placeholder solution based on difficulty level (1-6).
    
    Args:
        problem_data: The problem data containing train/test examples
        ground_truth: The actual solution matrix
        level: Difficulty level (1-6)
            1: Ground truth modified by random pixels (0 to max(width*2, 10))
            2: Ground truth with random removal of up to 2 rows and up to 2 columns (at least one dimension removed)
            3: Random crop with dimensions below 7x7 (random position and size)
            4: 3x3 zeros matrix
            5: Random matrix (same size as ground truth)
            6: Random matrix from problem data
    
    Returns:
        Placeholder matrix as list of lists
    """
    target_height = len(ground_truth)
    target_width = len(ground_truth[0]) if target_height > 0 else 0
    
    if level == 1:
        # Level 1: Ground truth modified by random amount of pixels
        placeholder_matrix = [[cell for cell in row] for row in ground_truth]
        max_modifications = max(target_width * 2, 10)
        num_modifications = random.randint(0, max_modifications)
        total_pixels = target_height * target_width
        num_modifications = min(num_modifications, total_pixels)
        
        if num_modifications > 0:
            positions = [(i, j) for i in range(target_height) for j in range(target_width)]
            positions_to_modify = random.sample(positions, num_modifications)
            for i, j in positions_to_modify:
                placeholder_matrix[i][j] = random.randint(0, 9)
    
    elif level == 2:
        # Level 2: Random removal of up to 2 rows and up to 2 columns
        # At least one row OR one column must be removed
        
        # Determine how many rows and columns can be removed
        max_rows_to_remove = min(2, target_height - 1)  # Keep at least 1 row
        max_cols_to_remove = min(2, target_width - 1)   # Keep at least 1 col
        
        # Randomly choose how many to remove
        rows_to_remove = random.randint(0, max_rows_to_remove)
        cols_to_remove = random.randint(0, max_cols_to_remove)
        
        # Ensure at least one row OR one column is removed
        if rows_to_remove == 0 and cols_to_remove == 0:
            # Force removal of at least one dimension
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
                placeholder_matrix = [[0]]
                return placeholder_matrix
        
        new_height = target_height - rows_to_remove
        new_width = target_width - cols_to_remove
        
        placeholder_matrix = [[ground_truth[i][j] for j in range(new_width)] 
                              for i in range(new_height)]
    
    elif level == 3:
        # Level 3: Random crop with dimensions below 7x7 that fit within ground truth
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
        
        placeholder_matrix = [[ground_truth[i][j] for j in range(start_col, end_col)] 
                              for i in range(start_row, end_row)]
    
    elif level == 4:
        # Level 4: 3x3 zeros matrix
        placeholder_matrix = [[0] * 3 for _ in range(3)]
    
    elif level == 5:
        # Level 5: Random matrix (same size as ground truth)
        placeholder_matrix = [[random.randint(0, 9) for _ in range(target_width)] 
                              for _ in range(target_height)]
    
    elif level == 6:
        # Level 6: Random matrix from problem data
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
            placeholder_matrix = [[cell for cell in row] for row in chosen_matrix]
        else:
            # Fallback if no matrices available
            placeholder_matrix = [[0] * target_width for _ in range(target_height)]
    
    else:
        raise ValueError(f"Invalid level: {level}. Must be between 1 and 6.")
    
    return placeholder_matrix