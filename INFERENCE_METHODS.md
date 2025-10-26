# Inference Methods for Model Evaluation

This document describes the four inference methods available in `evaluate_model.py` for evaluating ARC transduction models.

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

## 4. Augmented Deep Dive Method (Hybrid)

Combines augmentation breadth with iterative depth: applies augmentations to all problems, runs deep dive on each augmented version, reverses augmentations, and votes. **The most powerful but computationally expensive method.**

**Usage:**
```bash
python evaluate_model.py model_path --inference-mode augmented_deep_dive --num-augmentations 30 --deep-dive-iterations 16
```

**How it works (Tree Search Analogy):**
1. Parse ALL problem structures
2. Generate N augmentation configs (breadth - exploring different views)
3. Apply augmentations to create N × Problems augmented versions
4. **Run batched deep dive on ALL augmented problems** (depth - iterative refinement)
5. Reverse augmentations on final outputs (skip failures)
6. Take majority vote per original problem

**Visual Analogy:**
```
Original Problem
    ├─ Augmentation 1 → Deep Dive (iter 1 → 2 → ... → 16) → Reverse → Vote
    ├─ Augmentation 2 → Deep Dive (iter 1 → 2 → ... → 16) → Reverse → Vote
    ├─ Augmentation 3 → Deep Dive (iter 1 → 2 → ... → 16) → Reverse → Vote
    └─ ... (30 total)
        → Majority Vote → Final Answer
```

**Features:**
- **Breadth + Depth**: Explores multiple perspectives AND refines each
- Ultra-efficient batching at each iteration level
- Best of both worlds: robustness AND refinement
- Graceful failure handling for reversals
- Batches ALL augmented problems together at each iteration

**Parameters:**
- `--num-augmentations N`: Number of augmentations (default: 30)
- `--deep-dive-iterations N`: Max iterations per augmented problem (default: 16)

**Best for:**
- Maximum accuracy when computational cost is acceptable
- Problems that benefit from both multiple perspectives AND refinement
- Final evaluation where you want the most robust results
- Competition or benchmark submissions

**Performance:**
- **Computation**: Most expensive (Problems × Augmentations × Iterations)
- **vLLM Passes**: Up to N iterations (all augmented problems batched at each iteration)
- **Example**: 100 problems × 30 augs × 5 avg iterations = 5 batched calls (3,000 prompts/call)
- **Batching**: Fully batched - all augmented problems together at each iteration

**Implementation Details:**
- Reuses deep_dive_inference_batched for efficiency
- All augmented problems go through deep dive together
- Reversal happens only on final outputs (not intermediate iterations)
- Voting happens after all augmented versions complete
- Combines best practices from both parent methods

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

### Augmented Deep Dive (Hybrid)
- **Speed**: Slowest (most comprehensive but computationally intensive)
- **vLLM Passes**: Up to N iterations (all augmented problems batched at each iteration)
- **Batching**: ALL problems × ALL augmentations batched together at each iteration
- **Example**: 100 problems × 30 augs × 5 avg iters = 5 batched calls (3,000 prompts each)
- **Best When**: Maximum accuracy needed, computational cost acceptable

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

