import json
import gc
import re
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Directory where BARC tasks will be saved
barc_dir = Path("barc_training_data")
barc_dir.mkdir(exist_ok=True)

print("Streaming BARC dataset and saving to barc_training_data/ ...")
print("-" * 80)

# Set batch size and initialize tracking
BATCH_SIZE = 500
SAMPLES_BEFORE_CLEANUP = 5000

batch = []
sample_count = 0
written_count = 0
error_count = 0

# Load dataset in streaming mode
dataset = load_dataset("barc0/transduction_heavy_100k_jsonl", split="train_sft", streaming=True)

# Color mapping from names to integers
COLOR_MAP = {
    "Black": 0,
    "Blue": 1,
    "Red": 2,
    "Green": 3,
    "Yellow": 4,
    "Gray": 5,
    "Pink": 6,
    "Orange": 7,
    "Cyan": 8,
    "Maroon": 9
}

def parse_grid(grid_text):
    """Parse a grid from text format to 2D array of integers."""
    lines = grid_text.strip().split('\n')
    grid = []
    for line in lines:
        row = []
        colors = line.strip().split()
        for color in colors:
            if color in COLOR_MAP:
                row.append(COLOR_MAP[color])
            else:
                row.append(0)  # Default to Black
        if row:
            grid.append(row)
    return grid

def extract_examples_from_conversation(messages):
    """Extract input-output pairs from the conversation format."""
    user_content = None
    assistant_content = None
    
    for msg in messages:
        if msg['role'] == 'user' and 'Given input-output grid pairs' in msg['content']:
            user_content = msg['content']
        elif msg['role'] == 'assistant' and 'output grid' in msg['content'].lower():
            assistant_content = msg['content']
    
    if not user_content:
        return None, None
    
    # Parse training examples
    train_examples = []
    
    # Find all "Example N" sections
    example_pattern = r'Example \d+\s*\nInput:\s*\n(.*?)\n\nOutput:\s*\n(.*?)(?=\n\n(?:Example \d+|Here is the input))'
    matches = re.finditer(example_pattern, user_content, re.DOTALL)
    
    for match in matches:
        input_text = match.group(1)
        output_text = match.group(2)
        
        train_examples.append({
            "input": parse_grid(input_text),
            "output": parse_grid(output_text)
        })
    
    # Parse test example input
    test_input_pattern = r'Here is the input grid for the test example:\s*\nInput:\s*\n(.*?)(?=\n\n(?:Directly provide|$))'
    test_input_match = re.search(test_input_pattern, user_content, re.DOTALL)
    
    if not test_input_match:
        return train_examples, None
    
    test_input_text = test_input_match.group(1)
    test_input = parse_grid(test_input_text)
    
    # Parse test example output from assistant response
    test_output = None
    if assistant_content:
        # Extract the grid from the code block
        output_pattern = r'```\s*\n(.*?)\n```'
        output_match = re.search(output_pattern, assistant_content, re.DOTALL)
        
        if output_match:
            test_output_text = output_match.group(1)
            test_output = parse_grid(test_output_text)
    
    # Create test example
    test_example = {
        "input": test_input,
        "output": test_output if test_output else []
    }
    
    return train_examples, [test_example]

def process_batch(batch, barc_dir):
    global written_count, error_count
    for b_idx, example in enumerate(tqdm(batch, desc="Processing batch", unit="example")):
        try:
            messages = example.get('messages', [])
            if not messages:
                error_count += 1
                continue
            
            train_examples, test_examples = extract_examples_from_conversation(messages)
            
            if not train_examples or not test_examples:
                error_count += 1
                continue
            
            task_id = f'barc_task_{written_count}'
            
            # Construct new task
            new_task = {
                "train": train_examples,
                "test": test_examples,
                "task_id": task_id
            }
            
            # Save to file
            out_path = barc_dir / f"{task_id}.json"
            with open(out_path, 'w') as f:
                json.dump(new_task, f, indent=2)
            written_count += 1
            
            if written_count % 100 == 0:
                tqdm.write(f"[SAVED] {out_path} ({len(new_task['train'])} train, {len(new_task['test'])} test)")
        except Exception as e:
            error_count += 1
            tqdm.write(f"[ERROR] Failed to process sample {b_idx}: {str(e)}")

# Stream through dataset
for sample in dataset:
    sample_count += 1
    
    # Append to batch for processing
    batch.append(sample)
    
    # Process the batch when big enough
    if len(batch) >= BATCH_SIZE:
        print(f"\nProcessing batch at sample {sample_count} ...")
        process_batch(batch, barc_dir)
        batch.clear()
        
        # Delete cache if needed, address memory
        if written_count % SAMPLES_BEFORE_CLEANUP == 0:
            print("\n[INFO] Running manual cache/memory cleanup after {} samples...".format(written_count))
            gc.collect()
    
    # Optional: limit total samples (remove or adjust as needed)
    if sample_count >= 10:
        break

# Process any remaining samples in the last batch
if batch:
    print(f"\nProcessing final partial batch ...")
    process_batch(batch, barc_dir)
    batch.clear()

print(f"\nDONE. Processed {sample_count} samples, wrote {written_count} files to barc_training_data/.")
print(f"Errors: {error_count}")