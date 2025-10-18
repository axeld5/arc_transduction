"""
DSPy optimization script for ARC (Abstraction and Reasoning Corpus) problems.
Loads augmented training and evaluation data, optimizes prompts using GEPA.
"""

from dotenv import load_dotenv
import dspy
import os
import json
from collections import defaultdict

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure DSPy with OpenAI GPT-4
lm = dspy.LM("openai/gpt-5-mini", temperature=1, api_key=api_key, max_tokens=32000)
dspy.configure(lm=lm)


def load_json_data(filepath):
    """Load JSON data from file."""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def filter_training_data(data, levels=[1, 2, 3, 4], num_augmentations=3):
    """
    Filter training data to get specific levels and augmentation indices.
    
    Args:
        data: List of training examples
        levels: List of levels to include (1-4)
        num_augmentations: Number of different augmentation indices to take per problem
    
    Returns:
        Filtered list of examples
    """
    # Group by problem_name and augmentation_idx
    grouped = defaultdict(lambda: defaultdict(list))
    
    for example in data:
        metadata = example.get('metadata', {})
        problem_name = metadata.get('problem_name')
        level = metadata.get('level')
        aug_idx = metadata.get('augmentation_idx')
        
        if problem_name and level is not None and aug_idx is not None:
            grouped[problem_name][aug_idx].append(example)
    
    # Filter and select examples
    filtered = []
    problem_count = 0
    
    for problem_name, aug_dict in grouped.items():
        problem_count += 1
        # Take first num_augmentations augmentation indices
        selected_aug_indices = sorted(aug_dict.keys())[:num_augmentations]
        
        for aug_idx in selected_aug_indices:
            examples_for_aug = aug_dict[aug_idx]
            # Filter by level (note: levels are 0-indexed in data, but user refers to them as 1-4)
            # level 1 = index 0, level 2 = index 1, etc.
            for example in examples_for_aug:
                ex_level = example['metadata']['level']
                # Check if level is in range (adjusting for 0-indexing)
                if ex_level in [l - 1 for l in levels]:
                    filtered.append(example)
    
    print(f"Filtered {len(filtered)} training examples from {problem_count} problems")
    print(f"  Levels: {levels} (adjusted to 0-indexed: {[l-1 for l in levels]})")
    print(f"  Augmentations per problem: {num_augmentations}")
    
    return filtered


def filter_eval_data(data, level=4):
    """
    Filter evaluation data to get only level 4 examples.
    
    Args:
        data: List of evaluation examples
        level: Level to filter (default 4)
    
    Returns:
        Filtered list of examples
    """
    filtered = []
    problem_names = set()
    
    for example in data:
        metadata = example.get('metadata', {})
        ex_level = metadata.get('level')
        problem_name = metadata.get('problem_name')
        
        # Level 4 corresponds to index 3 (0-indexed)
        if ex_level == level - 1:
            filtered.append(example)
            if problem_name:
                problem_names.add(problem_name)
    
    print(f"Filtered {len(filtered)} evaluation examples (level {level}) from {len(problem_names)} problems")
    
    return filtered


def prepare_dspy_examples(data):
    """
    Convert data to DSPy Example format.
    
    Args:
        data: List of examples with 'problem' and 'answer' fields
    
    Returns:
        List of DSPy Examples
    """
    examples = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in data
    ]
    return examples


# Define DSPy signature
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()


# Define metric for evaluation
def arc_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Metric to evaluate ARC outputs.
    
    Returns:
        int: 1 if the prediction matches the expected output exactly, 0 otherwise
    """
    try:
        expected_output = example["answer"].strip()
        predicted_output = prediction.answer.strip() if hasattr(prediction, 'answer') else str(prediction).strip()
        
        if expected_output == predicted_output:
            return 1
        else:
            return 0
            
    except Exception as e:
        return 0


def arc_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Metric to evaluate ARC outputs with feedback for GEPA optimization.
    
    Returns:
        dspy.Prediction: Contains score (1 if exact match, 0 otherwise) and feedback text
    """
    try:
        expected_output = example["answer"].strip()
        predicted_output = prediction.answer.strip() if hasattr(prediction, 'answer') else str(prediction).strip()
        score = 1 if expected_output == predicted_output else 0
        feedback_text = ""
        
        if score == 1:
            feedback_text = f"Your answer is correct. The expected output matches your prediction exactly."
        else:
            feedback_text = f"Your answer is incorrect. The expected output does not match your prediction.\n"
            feedback_text += f"Expected:\n{expected_output}\n\nYour output:\n{predicted_output}\n"
            feedback_text += "Please ensure your output grid matches the expected format exactly, including spacing and newlines."
        
        return dspy.Prediction(score=score, feedback=feedback_text)
        
    except Exception as e:
        feedback_text = f"Error processing your answer: {str(e)}. Please ensure your answer follows the correct format for ARC grid outputs."
        return dspy.Prediction(score=0, feedback=feedback_text)


def main():
    print("=" * 80)
    print("DSPy ARC Problem Optimization")
    print("=" * 80)
    
    # Load data
    train_data = load_json_data('generated_data/train_data.json')
    eval_data = load_json_data('generated_data/eval_data.json')
    
    print("\n" + "=" * 80)
    print("Filtering Data")
    print("=" * 80)
    
    # Filter training data: levels 1-4, 3 augmentations per problem
    filtered_train = filter_training_data(train_data, levels=[1, 2, 3, 4], num_augmentations=3)
    
    # Filter evaluation data: level 4 only
    filtered_eval = filter_eval_data(eval_data, level=4)
    
    # Prepare DSPy examples
    print("\n" + "=" * 80)
    print("Preparing DSPy Examples")
    print("=" * 80)
    
    train_set = prepare_dspy_examples(filtered_train)
    eval_set = prepare_dspy_examples(filtered_eval)
    
    print(f"Training set size: {len(train_set)}")
    print(f"Evaluation set size: {len(eval_set)}")
    
    # Split training set into train and validation
    split_point = int(len(train_set) * 0.8)
    train_subset = train_set[:split_point]
    val_subset = train_set[split_point:]
    
    print(f"Train subset: {len(train_subset)}")
    print(f"Validation subset: {len(val_subset)}")
    
    # Create program
    print("\n" + "=" * 80)
    print("Creating DSPy Program")
    print("=" * 80)
    
    program = dspy.ChainOfThought(GenerateResponse)
    
    # Evaluate baseline
    print("\n" + "=" * 80)
    print("Evaluating Baseline Program")
    print("=" * 80)
    
    evaluate = dspy.Evaluate(
        devset=eval_set[:10],  # Use small subset for quick baseline
        metric=arc_metric,
        num_threads=4,
        display_table=True,
        display_progress=True
    )
    
    baseline_score = evaluate(program)
    print(f"\nBaseline Score: {baseline_score}")
    
    # Optimize with GEPA
    print("\n" + "=" * 80)
    print("Optimizing with GEPA")
    print("=" * 80)
    
    from dspy import GEPA
    
    optimizer = GEPA(
        metric=arc_metric_with_feedback,
        auto="light",
        num_threads=32,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=dspy.LM(model="openai/gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key)
    )
    
    optimized_program = optimizer.compile(
        program,
        trainset=train_subset,
        valset=val_subset,
    )
    
    # Evaluate optimized program
    print("\n" + "=" * 80)
    print("Evaluating Optimized Program")
    print("=" * 80)
    
    evaluate_full = dspy.Evaluate(
        devset=eval_set,
        metric=arc_metric,
        num_threads=32,
        display_table=True,
        display_progress=True
    )
    
    optimized_score = evaluate_full(optimized_program)
    print(f"\nOptimized Score: {optimized_score}")
    print(f"Baseline Score: {baseline_score}")
    print(f"Improvement: {optimized_score - baseline_score:.4f}")
    
    # Save optimized program
    print("\n" + "=" * 80)
    print("Saving Optimized Program")
    print("=" * 80)
    
    # Save the full program
    optimized_program.save('optimized_program.json')
    print("Saved optimized program to: optimized_program.json")
    
    # Extract and save the prompt to a text file
    with open('optimized_prompt.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy Optimized Prompt for ARC Problems\n")
        f.write("=" * 80 + "\n\n")
        
        # Try to extract the prompt from the optimized program
        try:
            # Get the predictor
            if hasattr(optimized_program, 'predictors'):
                for i, predictor in enumerate(optimized_program.predictors()):
                    f.write(f"\n--- Predictor {i+1} ---\n\n")
                    
                    # Get signature
                    if hasattr(predictor, 'signature'):
                        f.write(f"Signature: {predictor.signature}\n\n")
                    
                    # Get extended signature if available
                    if hasattr(predictor, 'extended_signature'):
                        f.write(f"Extended Signature:\n{predictor.extended_signature}\n\n")
                    
                    # Get demos/examples if available
                    if hasattr(predictor, 'demos'):
                        f.write(f"Number of demos: {len(predictor.demos)}\n\n")
                        for j, demo in enumerate(predictor.demos[:3]):  # Show first 3 demos
                            f.write(f"Demo {j+1}:\n")
                            f.write(f"{demo}\n\n")
            
            # Write the full program representation
            f.write("\n" + "=" * 80 + "\n")
            f.write("Full Program Representation\n")
            f.write("=" * 80 + "\n\n")
            f.write(str(optimized_program))
            
        except Exception as e:
            f.write(f"Error extracting detailed prompt: {e}\n\n")
            f.write("Full program representation:\n")
            f.write(str(optimized_program))
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Optimization Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline Score: {baseline_score}\n")
        f.write(f"Optimized Score: {optimized_score}\n")
        f.write(f"Improvement: {optimized_score - baseline_score:.4f}\n")
        f.write(f"\nTraining examples: {len(train_subset)}\n")
        f.write(f"Validation examples: {len(val_subset)}\n")
        f.write(f"Evaluation examples: {len(eval_set)}\n")
    
    print("Saved optimized prompt to: optimized_prompt.txt")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

