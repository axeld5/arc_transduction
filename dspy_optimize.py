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
lm = dspy.LM("openai/gpt-5-nano", temperature=1, api_key=api_key, max_tokens=16000)
dspy.configure(lm=lm)


def load_json_data(filepath):
    """Load JSON data from file."""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def filter_training_data(data, levels=[1, 3, 4], num_augmentations=3):
    """
    Filter training data to get specific levels and augmentation indices.
    
    Args:
        data: List of training examples
        levels: List of levels to include (1-4)
        num_augmentations: Number of different augmentation indices to take per problem
    
    Returns:
        Filtered list of examples
    """
    # Collect up to `num_augmentations` examples per problem name, for the specified levels.
    # Only include problem names for which 3 augmentations are present for each selected level.
    per_problem_level_aug = defaultdict(lambda: defaultdict(dict))
    for example in data:
        metadata = example.get('metadata', {})
        problem_name = metadata.get('problem_name')
        level = metadata.get('level')
        augmentation_idx = metadata.get('augmentation_idx')
        if (
            problem_name
            and level in levels
            and augmentation_idx is not None
        ):
            # use dict to avoid duplicates if dataset is messy
            per_problem_level_aug[problem_name][level][augmentation_idx] = example

    filtered = []
    for problem_name, level_aug_dict in per_problem_level_aug.items():
        has_all = True
        for lvl in levels:
            aug_idxs = list(level_aug_dict[lvl].keys()) if lvl in level_aug_dict else []
            if len(aug_idxs) < num_augmentations:
                has_all = False
                break
        if has_all:
            for lvl in levels:
                aug_dict = level_aug_dict[lvl]
                selected_augs = sorted(aug_dict.keys())[:num_augmentations]
                for aug_idx in selected_augs:
                    filtered.append(aug_dict[aug_idx])
    print(f"Filtered {len(filtered)} training examples: 3 augmentations each for levels {levels}, per problem name.")
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
    seen_problems = set()
    for example in data:
        metadata = example.get('metadata', {})
        ex_level = metadata.get('level')
        problem_name = metadata.get('problem_name')
        if ex_level == 4 and problem_name not in seen_problems:
            filtered.append(example)
            seen_problems.add(problem_name)
    print(f"Filtered {len(filtered)} evaluation examples (level {level}) with one example per problem from {len(seen_problems)} problems")
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
    
    # Filter training data: levels 1-3-4, 3 augmentations per problem
    filtered_train = filter_training_data(train_data, levels=[1, 3, 4], num_augmentations=1)
    
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
        devset=eval_set,  # Use small subset for quick baseline
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
        reflection_lm=dspy.LM(model="openai/gpt-5", temperature=1, max_tokens=16000, api_key=api_key)
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
    
    # Save optimized program
    print("\n" + "=" * 80)
    print("Saving Optimized Program")
    print("=" * 80)

    # Save only optimized_program.predict.signature.instructions to text
    try:
        instructions = optimized_program.predict.signature.instructions
    except AttributeError:
        instructions = ""

    with open('optimized_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)

    print("Saved optimized prompt instructions to: optimized_prompt.txt")
    
    print("Saved optimized prompt to: optimized_prompt.txt")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

