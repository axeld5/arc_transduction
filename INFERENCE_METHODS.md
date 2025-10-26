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

An iterative refinement approach that feeds the model's output back as input for up to N iterations.

**Usage:**
```bash
python evaluate_model.py model_path --inference-mode deep_dive --deep-dive-iterations 16
```

**How it works:**
1. Generate an initial prediction for the problem
2. Replace the test output section with the generated answer
3. Feed the updated problem back to the model
4. Repeat for up to N iterations (or until parsing fails)
5. Return the final iteration's output

**Features:**
- Allows the model to refine its answer iteratively
- Can help models "think through" complex problems step by step
- Stops early if the model produces unparseable output

**Parameters:**
- `--deep-dive-iterations N`: Maximum number of iterations (default: 16)

**Best for:**
- Models that can benefit from iterative refinement
- Complex problems where initial answers might be rough drafts
- Testing if models can self-correct

## 3. Augmented Voting Method

Applies multiple augmentations to the problem, generates predictions for each, reverses the augmentations, and takes a majority vote.

**Usage:**
```bash
python evaluate_model.py model_path --inference-mode augmented_voting --num-augmentations 30
```

**How it works:**
1. Parse the problem structure (examples + test input)
2. Generate N different augmentation configurations:
   - Rotations (90°, 180°, 270°)
   - Flips (vertical, horizontal, both)
   - Color permutations
3. Apply each augmentation to ALL grids in the problem (examples + test input)
4. Generate predictions for all augmented problems in ONE batch (efficient!)
5. Reverse the augmentations on each prediction
6. Take the most common (majority vote) answer

**Features:**
- Test-time augmentation for improved robustness
- Efficient batch generation (all augmentations in one vLLM pass)
- Automatic augmentation reversal
- Majority voting for consensus

**Parameters:**
- `--num-augmentations N`: Number of augmented versions (default: 30)

**Best for:**
- Models that might be sensitive to rotation or reflection
- Increasing confidence through ensemble voting
- Problems where the pattern is invariant to certain transformations

**Implementation Details:**
- Uses regex to parse problem structure robustly
- Tracks color maps for proper reversal
- Falls back to standard inference if parsing fails
- Preserves original problem introduction text

## Performance Considerations

### Standard
- **Speed**: Fastest
- **vLLM Passes**: 1 (or N for multiple attempts)
- **Best When**: Quick evaluation needed

### Deep Dive
- **Speed**: Slowest (sequential)
- **vLLM Passes**: Up to N per problem (16 by default)
- **Best When**: Model benefits from iterative refinement

### Augmented Voting
- **Speed**: Medium (efficient batching)
- **vLLM Passes**: 1 per problem (all augmentations batched)
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

