from datasets import load_dataset

# Load dataset in streaming mode
dataset = load_dataset("julien31/soar_arc_train_5M", split="train", streaming=True)

# Take only the first 100 samples
first_100 = []
for i, sample in enumerate(dataset):
    first_100.append(sample)
    if i >= 99:
        break

print(first_100[0])
print(f"Loaded {len(first_100)} samples.")