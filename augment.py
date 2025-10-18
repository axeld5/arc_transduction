"""All augmentations applied to a problem"""

from typing import List, Dict, Callable

def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    rotated = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            rotated[j][rows - 1 - i] = grid[i][j]
    return rotated

def rotate_180(grid: List[List[int]]) -> List[List[int]]:
    return [row[::-1] for row in grid[::-1]]

def rotate_270(grid: List[List[int]]) -> List[List[int]]:    
    rows, cols = len(grid), len(grid[0])
    rotated = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            rotated[cols - 1 - j][i] = grid[i][j]
    return rotated

def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    return grid[::-1]

def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    return [row[::-1] for row in grid]

def double_flip(grid: List[List[int]]) -> List[List[int]]:
    return flip_horizontal(flip_vertical(grid))

def apply_color_permutation(grid: List[List[int]], color_map: Dict[int, int] = None) -> List[List[int]]:
    import random
    if color_map is None:
        colors = list(range(10))
        shuffled_colors = colors.copy()
        while True:
            random.shuffle(shuffled_colors)
            if all(original != shuffled for original, shuffled in zip(colors, shuffled_colors)):
                break
        color_map = dict(zip(colors, shuffled_colors))
    return [[color_map.get(cell, cell) for cell in row] for row in grid]

groups = {
    "rotate": [None, rotate_90, rotate_180, rotate_270],
    "flip": [None, flip_vertical, flip_horizontal, double_flip],
    "color": [None, apply_color_permutation]
}

def assign_random_augmentations(seed: int = None) -> List[List[int]]:
    import random
    if seed is not None:
        random.seed(seed)
    chosen_augmentations = []
    for group_name, augmentations in groups.items():
        selected_augmentation = random.choice(augmentations)
        if selected_augmentation is not None:
            chosen_augmentations.append(selected_augmentation)
    return chosen_augmentations

def apply_augmentations_to_grids(grids: List[List[List[int]]], 
                                augmentations: List[Callable]) -> List[List[List[int]]]:
    augmented_grids = grids.copy()
    for augmentation in augmentations:
        if augmentation == apply_color_permutation:
            import random
            colors = list(range(10))
            shuffled_colors = colors.copy()
            while True:
                random.shuffle(shuffled_colors)
                if all(original != shuffled for original, shuffled in zip(colors, shuffled_colors)):
                    break
            color_map = dict(zip(colors, shuffled_colors))
            augmented_grids = [augmentation(grid, color_map) for grid in augmented_grids]
        else:
            augmented_grids = [augmentation(grid) for grid in augmented_grids]
    
    return augmented_grids