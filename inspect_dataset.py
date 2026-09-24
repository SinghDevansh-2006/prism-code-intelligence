from datasets import load_dataset

dataset = load_dataset("CoIR-Retrieval/apps")

print(dataset)

for split_name in dataset.keys():
    print(f"\nSPLIT: {split_name}")
    print("Rows:", len(dataset[split_name]))
    print("Columns:", dataset[split_name].column_names)

print("\nFIRST TRAIN EXAMPLE:")
print(dataset["train"][0])
