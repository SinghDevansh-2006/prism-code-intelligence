import json
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3-Embedding-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open("data/split.json") as f:
    split = json.load(f)

validation_doc_ids = {
    pair["doc_id"]
    for pair in split["validation"]
}

corpus = load_dataset(
    "CoIR-Retrieval/apps",
    "corpus",
    split="corpus"
)

train_corpus = [
    row for row in corpus
    if row["partition"] == "train"
]

results = []

for i, row in enumerate(train_corpus, 1):
    token_ids = tokenizer(
        row["text"],
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    results.append(
        (
            len(token_ids),
            row["_id"],
            len(row["text"]),
            row["_id"] in validation_doc_ids
        )
    )

    if i % 1000 == 0:
        print(f"Processed {i}/{len(train_corpus)}")

results.sort(reverse=True)

print("\nTOP 15 LONGEST TRAIN DOCUMENTS")
print("=" * 80)

for token_count, doc_id, char_count, is_validation_target in results[:15]:
    print(
        f"{doc_id:8} | "
        f"tokens={token_count:7} | "
        f"chars={char_count:8} | "
        f"validation_target={is_validation_target}"
    )
