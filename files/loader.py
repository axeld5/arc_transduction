"""
Loader functions for ARC (Abstraction and Reasoning Corpus) problems.

This module provides functions to load problems from:
- Evaluation dataset: Problems used for evaluation
- Training dataset: Problems used for training
- ConceptARC corpus: Categorized problems organized by concept

Problems are stored as JSON files with unique IDs or concept-based names.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_evaluation_problem(problem_id: str, data_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    Load a problem from the evaluation dataset given its ID.
    
    Args:
        problem_id (str): The unique identifier for the problem (e.g., "00576224")
        data_dir (str): The root directory containing the evaluation folder. Defaults to current directory.
        
    Returns:
        Optional[Dict[str, Any]]: The problem data as a dictionary, or None if not found.
        
    The returned dictionary contains:
        - "train": List of training examples, each with "input" and "output" grids
        - "test": List of test examples, each with "input" and "output" grids
        - "arc-gen": (if present) Additional generated examples
        
    Raises:
        FileNotFoundError: If the evaluation directory or problem file doesn't exist
        json.JSONDecodeError: If the problem file contains invalid JSON
    """
    evaluation_dir = Path(data_dir) / "evaluation_data"
    problem_file = evaluation_dir / f"{problem_id}.json"
    
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")
    
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        return problem_data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in problem file {problem_file}: {e}")


def load_training_problem(problem_id: str, data_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    Load a problem from the training dataset given its ID.
    
    Args:
        problem_id (str): The unique identifier for the problem (e.g., "007bbfb7")
        data_dir (str): The root directory containing the training folder. Defaults to current directory.
        
    Returns:
        Optional[Dict[str, Any]]: The problem data as a dictionary, or None if not found.
        
    The returned dictionary contains:
        - "train": List of training examples, each with "input" and "output" grids
        - "test": List of test examples, each with "input" and "output" grids
        - "arc-gen": (if present) Additional generated examples
        
    Raises:
        FileNotFoundError: If the training directory or problem file doesn't exist
        json.JSONDecodeError: If the problem file contains invalid JSON
    """
    training_dir = Path(data_dir) / "training_data"
    problem_file = training_dir / f"{problem_id}.json"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        return problem_data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in problem file {problem_file}: {e}")


def list_evaluation_problems(data_dir: str = ".") -> List[str]:
    """
    List all available problem IDs in the evaluation dataset.
    
    Args:
        data_dir (str): The root directory containing the evaluation folder. Defaults to current directory.
        
    Returns:
        List[str]: A list of problem IDs (without the .json extension)
        
    Raises:
        FileNotFoundError: If the evaluation directory doesn't exist
    """
    evaluation_dir = Path(data_dir) / "evaluation_data"
    
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")
    
    problem_files = evaluation_dir.glob("*.json")
    return [f.stem for f in problem_files]


def list_training_problems(data_dir: str = ".") -> List[str]:
    """
    List all available problem IDs in the training dataset.
    
    Args:
        data_dir (str): The root directory containing the training folder. Defaults to current directory.
        
    Returns:
        List[str]: A list of problem IDs (without the .json extension)
        
    Raises:
        FileNotFoundError: If the training directory doesn't exist
    """
    training_dir = Path(data_dir) / "training_data"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    problem_files = training_dir.glob("*.json")
    return [f.stem for f in problem_files]


def load_corpus_problem(concept: str, problem_number: int, data_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    Load a problem from the ConceptARC corpus given its concept and problem number.
    
    Args:
        concept (str): The concept name (e.g., "AboveBelow", "Center", "Copy")
        problem_number (int): The problem number (e.g., 1, 2, 3, etc.)
        data_dir (str): The root directory containing the corpus folder. Defaults to current directory.
        
    Returns:
        Optional[Dict[str, Any]]: The problem data as a dictionary, or None if not found.
        
    The returned dictionary contains:
        - "train": List of training examples, each with "input" and "output" grids
        - "test": List of test examples, each with "input" and "output" grids
        
    Raises:
        FileNotFoundError: If the corpus directory, concept directory, or problem file doesn't exist
        json.JSONDecodeError: If the problem file contains invalid JSON
    """
    corpus_dir = Path(data_dir) / "corpus"
    concept_dir = corpus_dir / concept
    problem_file = concept_dir / f"{concept}{problem_number}.json"
    
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    if not concept_dir.exists():
        raise FileNotFoundError(f"Concept directory not found: {concept_dir}")
    
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        return problem_data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in problem file {problem_file}: {e}")


def load_corpus_problem_by_name(problem_name: str, data_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    Load a problem from the ConceptARC corpus given its full name (e.g., "AboveBelow1").
    
    Args:
        problem_name (str): The full problem name (e.g., "AboveBelow1", "Center5")
        data_dir (str): The root directory containing the corpus folder. Defaults to current directory.
        
    Returns:
        Optional[Dict[str, Any]]: The problem data as a dictionary, or None if not found.
        
    The returned dictionary contains:
        - "train": List of training examples, each with "input" and "output" grids
        - "test": List of test examples, each with "input" and "output" grids
        
    Raises:
        FileNotFoundError: If the corpus directory or problem file doesn't exist
        json.JSONDecodeError: If the problem file contains invalid JSON
        ValueError: If the problem name format is invalid
    """
    corpus_dir = Path(data_dir) / "corpus"
    
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    # Try to find the file in any subdirectory
    for concept_dir in corpus_dir.iterdir():
        if concept_dir.is_dir():
            problem_file = concept_dir / f"{problem_name}.json"
            if problem_file.exists():
                try:
                    with open(problem_file, 'r', encoding='utf-8') as f:
                        problem_data = json.load(f)
                    return problem_data
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"Invalid JSON in problem file {problem_file}: {e}")
    
    raise FileNotFoundError(f"Problem file not found for: {problem_name}")


def list_corpus_concepts(data_dir: str = ".") -> List[str]:
    """
    List all available concept categories in the ConceptARC corpus.
    
    Args:
        data_dir (str): The root directory containing the corpus folder. Defaults to current directory.
        
    Returns:
        List[str]: A list of concept names (directory names)
        
    Raises:
        FileNotFoundError: If the corpus directory doesn't exist
    """
    corpus_dir = Path(data_dir) / "corpus"
    
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    concepts = [d.name for d in corpus_dir.iterdir() if d.is_dir()]
    return sorted(concepts)


def list_corpus_problems_by_concept(concept: str, data_dir: str = ".") -> List[str]:
    """
    List all available problems for a specific concept in the ConceptARC corpus.
    
    Args:
        concept (str): The concept name (e.g., "AboveBelow", "Center")
        data_dir (str): The root directory containing the corpus folder. Defaults to current directory.
        
    Returns:
        List[str]: A list of problem names (without the .json extension)
        
    Raises:
        FileNotFoundError: If the corpus or concept directory doesn't exist
    """
    corpus_dir = Path(data_dir) / "corpus"
    concept_dir = corpus_dir / concept
    
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    if not concept_dir.exists():
        raise FileNotFoundError(f"Concept directory not found: {concept_dir}")
    
    problem_files = concept_dir.glob("*.json")
    return sorted([f.stem for f in problem_files])


def list_all_corpus_problems(data_dir: str = ".") -> Dict[str, List[str]]:
    """
    List all available problems in the ConceptARC corpus, organized by concept.
    
    Args:
        data_dir (str): The root directory containing the corpus folder. Defaults to current directory.
        
    Returns:
        Dict[str, List[str]]: A dictionary mapping concept names to lists of problem names
        
    Raises:
        FileNotFoundError: If the corpus directory doesn't exist
    """
    corpus_dir = Path(data_dir) / "corpus"
    
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    all_problems = {}
    for concept_dir in corpus_dir.iterdir():
        if concept_dir.is_dir():
            concept_name = concept_dir.name
            problem_files = concept_dir.glob("*.json")
            all_problems[concept_name] = sorted([f.stem for f in problem_files])
    
    return dict(sorted(all_problems.items()))


def get_problem_stats(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about a problem.
    
    Args:
        problem_data (Dict[str, Any]): The problem data dictionary
        
    Returns:
        Dict[str, Any]: Statistics including number of training/test examples and grid sizes
    """
    stats = {
        "train_examples": len(problem_data.get("train", [])),
        "test_examples": len(problem_data.get("test", [])),
        "arc_gen_examples": len(problem_data.get("arc-gen", [])),
    }
    
    # Get grid size information from first training example if available
    if problem_data.get("train"):
        first_example = problem_data["train"][0]
        if "input" in first_example:
            input_grid = first_example["input"]
            stats["input_height"] = len(input_grid)
            stats["input_width"] = len(input_grid[0]) if input_grid else 0
        
        if "output" in first_example:
            output_grid = first_example["output"]
            stats["output_height"] = len(output_grid)
            stats["output_width"] = len(output_grid[0]) if output_grid else 0
    
    return stats


# Example usage
if __name__ == "__main__":
    # Example: Load an evaluation problem
    try:
        problem = load_evaluation_problem("00576224")
        if problem:
            print("Loaded evaluation problem successfully!")
            print(f"Training examples: {len(problem['train'])}")
            print(f"Test examples: {len(problem['test'])}")
            
            # Show stats
            stats = get_problem_stats(problem)
            print(f"Problem stats: {stats}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: Load a training problem
    try:
        problem = load_training_problem("007bbfb7")
        if problem:
            print("\nLoaded training problem successfully!")
            print(f"Training examples: {len(problem['train'])}")
            print(f"Test examples: {len(problem['test'])}")
            
            # Show stats
            stats = get_problem_stats(problem)
            print(f"Problem stats: {stats}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: List available problems
    try:
        eval_problems = list_evaluation_problems()
        print(f"\nFound {len(eval_problems)} evaluation problems")
        
        train_problems = list_training_problems()
        print(f"Found {len(train_problems)} training problems")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: Load a corpus problem by concept and number
    try:
        problem = load_corpus_problem("AboveBelow", 1)
        if problem:
            print("\nLoaded corpus problem AboveBelow1 successfully!")
            print(f"Training examples: {len(problem['train'])}")
            print(f"Test examples: {len(problem['test'])}")
            
            stats = get_problem_stats(problem)
            print(f"Problem stats: {stats}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: Load a corpus problem by name
    try:
        problem = load_corpus_problem_by_name("Center5")
        if problem:
            print("\nLoaded corpus problem Center5 successfully!")
            print(f"Training examples: {len(problem['train'])}")
            print(f"Test examples: {len(problem['test'])}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: List all corpus concepts
    try:
        concepts = list_corpus_concepts()
        print(f"\nFound {len(concepts)} corpus concepts:")
        print(f"Concepts: {', '.join(concepts)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: List problems for a specific concept
    try:
        problems = list_corpus_problems_by_concept("Copy")
        print(f"\nFound {len(problems)} problems in Copy concept")
        print(f"Problems: {problems}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: List all corpus problems
    try:
        all_problems = list_all_corpus_problems()
        total_count = sum(len(probs) for probs in all_problems.values())
        print(f"\nFound {total_count} total corpus problems across {len(all_problems)} concepts")
        for concept, probs in list(all_problems.items())[:3]:  # Show first 3 concepts
            print(f"  {concept}: {len(probs)} problems")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
