# Inference Methods for Model Evaluation

This document describes the three inference methods available in `evaluate_model.py` for evaluating ARC transduction models.

## Usage

```bash
python evaluate_model.py <model_path> [options]
```

### Command Line Options

- `--inference-mode {standard,deep_dive,augmented_voting}`: Choose inference strategy (default: standard)
- `--deep-dive-iterations N`: Number of iterations for deep_dive mode (default: 16)
- `--num-augmentations N`: Number of augmentations for augmented_voting mode (default: 30)

## 1. Standard Inference (default)

The baseline method that generates predictions directly from the model.

**Usage:**
```bash
python evaluate_model.py model_path --inference-mode standard --attempts 3
```

**Features:**
- Straightforward batch generation
- Supports multiple attempts per problem
- Most efficient for quick evaluations

**Parameters:**
- `--attempts N`: Number of generation attempts per problem

## 2. Deep Dive Method

An iterative refinement approach that feeds the model's output back as input for up to N iterations. **All problems are processed in batches at each iteration for maximum efficiency.**

**Usage:**
```bash
python evaluate_model.py model_path --inference-mode deep_dive --deep-dive-iterations 16
```

**How it works:**
1. **Iteration 1**: Generate predictions for ALL problems in batch
2. Parse each output and update problem statements with generated answers
3. **Iteration 2**: Generate new predictions for all problems that succeeded (batched)
4. Continue iterating, batching all active problems together
5. Stop when max iterations reached or all problems fail to parse
6. Return the final iteration's output for each problem

**Features:**
- Allows the model to refine its answer iteratively
- **Efficient batching**: All problems processed together at each iteration
- Automatically removes problems from the batch if parsing fails
- Can help models "think through" complex problems step by step

**Parameters:**
- `--deep-dive-iterations N`: Maximum number of iterations (default: 16)

**Best for:**
- Models that can benefit from iterative refinement
- Complex problems where initial answers might be rough drafts
- Testing if models can self-correct

**Performance:**
- Processes all problems in batches at each iteration
- Much more efficient than per-problem iteration
- Example: 100 problems, 5 iterations = 5 batched vLLM calls (not 500!)

## 3. Augmented Voting Method

Applies multiple augmentations to ALL problems, generates predictions in one giant batch, reverses augmentations, and takes majority votes. **All problems are processed together for maximum vLLM efficiency.**

**Usage:**
```bash
python evaluate_model.py model_path --inference-mode augmented_voting --num-augmentations 30
```

**How it works:**
1. Parse ALL problem structures (examples + test input)
2. Generate N augmentation configurations (shared across all problems):
   - Rotations (90°, 180°, 270°)
   - Flips (vertical, horizontal, both)
   - Color permutations
3. Apply each augmentation to ALL grids in ALL problems
4. **Generate predictions for ALL augmented problems in ONE GIANT batch** (max efficiency!)
5. Reverse the augmentations on each prediction (skip if reversal fails)
6. Take the most common (majority vote) answer per problem

**Features:**
- Test-time augmentation for improved robustness
- **Ultra-efficient batching**: All problems × all augmentations in ONE vLLM pass
- Automatic augmentation reversal with graceful failure handling
- Majority voting for consensus per problem
- Skips failed reversals instead of crashing

**Parameters:**
- `--num-augmentations N`: Number of augmented versions (default: 30)

**Best for:**
- Models that might be sensitive to rotation or reflection
- Increasing confidence through ensemble voting
- Problems where the pattern is invariant to certain transformations

**Implementation Details:**
- Batches ALL problems together (not per-problem)
- Uses regex to parse problem structure robustly
- Tracks color maps for proper reversal
- Falls back to standard inference if parsing fails
- Gracefully handles reversal failures (skips in vote counter)
- Preserves original problem introduction text

## Performance Considerations

### Standard
- **Speed**: Fastest
- **vLLM Passes**: 1 (or N for multiple attempts)
- **Batching**: All problems in one batch
- **Best When**: Quick evaluation needed

### Deep Dive
- **Speed**: Medium (iterative but batched)
- **vLLM Passes**: Up to N iterations (batching all problems at each iteration)
- **Batching**: All active problems batched together at each iteration
- **Example**: 100 problems × 5 iterations = 5 batched calls (not 500!)
- **Best When**: Model benefits from iterative refinement

### Augmented Voting
- **Speed**: Medium-Slow (many augmentations but ultra-efficient batching)
- **vLLM Passes**: 1 total (all problems × all augmentations in ONE batch!)
- **Batching**: ALL problems × ALL augmentations in a single giant batch
- **Example**: 100 problems × 30 augmentations = 1 batched call with 3,000 prompts
- **Best When**: Seeking robust predictions through augmentation

## Examples

### Standard with 5 attempts per problem
```bash
python evaluate_model.py unsloth/Qwen2.5-3B-Instruct \
    --inference-mode standard \
    --attempts 5 \
    --samples-per-level 10
```

### Deep dive with 16 iterations
```bash
python evaluate_model.py unsloth/Qwen2.5-3B-Instruct \
    --inference-mode deep_dive \
    --deep-dive-iterations 16 \
    --samples-per-level 10
```

### Augmented voting with 30 augmentations
```bash
python evaluate_model.py unsloth/Qwen2.5-3B-Instruct \
    --inference-mode augmented_voting \
    --num-augmentations 30 \
    --samples-per-level 10
```

### Using with LoRA adapter
```bash
python evaluate_model.py unsloth/Qwen2.5-3B-Instruct \
    --use-lora \
    --lora-path outputs/checkpoint-1000 \
    --inference-mode augmented_voting \
    --num-augmentations 30
```

## Notes

- All methods support system prompts, LoRA adapters, and example printing
- The augmented voting method automatically handles failures gracefully
- Deep dive method preserves example grids while updating only test output
- Results are saved to a JSON file automatically

