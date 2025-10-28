import json
import gc
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Directory where new tasks will be saved
soar_dir = Path("soar_training_data")
soar_dir.mkdir(exist_ok=True)

print("Streaming dataset and saving incorrect predictions into soar_training_data/ ...")
print("-" * 80)

# Set batch size and initialize tracking
BATCH_SIZE = 500
SAMPLES_BEFORE_CLEANUP = 5000

batch = []
seen_indices = dict()     # task_id -> set of indices for prediction
sample_count = 0
written_count = 0

dataset = load_dataset("julien31/soar_arc_train_5M", split="train", streaming=True)

def process_batch(batch, seen_indices, soar_dir):
    global written_count
    for b_idx, example in enumerate(tqdm(batch, desc="Processing batch", unit="example")):
        task_id = example.get('task_id')
        model = example.get('model', 'N/A')
        generation = example.get('generation', 'N/A')
        pred_idx = seen_indices.setdefault(task_id, set())
        pred_index = len(pred_idx)
        seen_indices[task_id].add(pred_index)  # only used to count up (not deduplication)

        # Find source file
        src_path = Path(f"training_data/{task_id}.json")
        if not src_path.exists():
            print(f"[ERROR] Original file for task_id {task_id} not found at {src_path}")
            continue

        # Load original ARC task
        with open(src_path, 'r') as f:
            original_task = json.load(f)

        # Construct new task as before
        new_task = {
            "train": [],
            "test": [],
            "task_id": task_id,
            "from_model": model,
            "generation": generation,
            "prediction_index": pred_index,
            "code": example.get('code', None)
        }
        predicted_train_outputs = example.get('predicted_train_output', [])
        predicted_test_outputs = example.get('predicted_test_output', [])

        for i, train_ex in enumerate(original_task.get("train", [])):
            new_task["train"].append({
                "input": train_ex["input"],
                "output": predicted_train_outputs[i] if i < len(predicted_train_outputs) else train_ex["output"]
            })
        for i, test_ex in enumerate(original_task.get("test", [])):
            new_task["test"].append({
                "input": test_ex["input"],
                "output": predicted_test_outputs[i] if i < len(predicted_test_outputs) else test_ex["output"]
            })

        # Use indexed filename for multiple preds per task_id
        out_path = soar_dir / f"{task_id}_pred_{pred_index}.json"
        with open(out_path, 'w') as f:
            json.dump(new_task, f, indent=2)
        written_count += 1
        tqdm.write(f"[SAVED] {out_path} ({len(new_task['train'])} train, {len(new_task['test'])} test)")


for sample in dataset:
    sample_count += 1

    # Flags for incorrect predictions
    has_incorrect_train = False in sample.get('correct_train_input', [])
    has_incorrect_test = False in sample.get('correct_test_input', [])

    if not (has_incorrect_train or has_incorrect_test):
        # Optionally print progress every 1000
        if sample_count % 1000 == 0:
            print(f"Samples checked: {sample_count} | Written: {written_count}")
        continue

    # Append to batch for processing
    batch.append(sample)

    # Process the batch when big enough
    if len(batch) >= BATCH_SIZE:
        print(f"\nProcessing batch at sample {sample_count} ...")
        process_batch(batch, seen_indices, soar_dir)
        batch.clear()

        # Delete cache if needed, address memory
        if written_count % SAMPLES_BEFORE_CLEANUP == 0:
            print("\n[INFO] Running manual cache/memory cleanup after {} predictions...".format(written_count))
            gc.collect()

# Process any remaining samples in the last batch
if batch:
    print(f"\nProcessing final partial batch ...")
    process_batch(batch, seen_indices, soar_dir)
    batch.clear()

print(f"\nDONE. Checked {sample_count} samples, wrote {written_count} erroneous prediction files to soar_training_data.")
