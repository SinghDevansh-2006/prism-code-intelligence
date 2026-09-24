import json
import random
from datasets import load_dataset

SEED = 42
VALIDATION_SIZE = 1000

qrels = load_dataset("CoIR-Retrieval/apps")["train"]

pairs = [
    {
        "query_id": row["query-id"],
        "doc_id": row["corpus-id"],
    }
    for row in qrels
]

random.Random(SEED).shuffle(pairs)

validation = pairs[:VALIDATION_SIZE]
train = pairs[VALIDATION_SIZE:]

split = {
    "seed": SEED,
    "train": train,
    "validation": validation,
}

with open("data/split.json", "w") as f:
    json.dump(split, f, indent=2)

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("\nFirst 5 validation pairs:")
for pair in validation[:5]:
    print(pair)
