import json
from datasets import load_dataset
from transformers import AutoTokenizer
from src.code_normalizer import compact_large_literals

MODEL = "Qwen/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open("data/split.json") as f:
    split = json.load(f)

validation_targets = {
    x["doc_id"] for x in split["validation"]
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

long_docs = []

for row in train_corpus:
    compacted = compact_large_literals(row["text"])

    tokens = tokenizer(
        compacted,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    if len(tokens) > 2048:
        long_docs.append(
            (
                len(tokens),
                row["_id"],
                row["_id"] in validation_targets,
                compacted
            )
        )

long_docs.sort(reverse=True)

print("REMAINING DOCUMENTS OVER 2048 TOKENS:", len(long_docs))

for token_count, doc_id, is_validation, compacted in long_docs:
    print("\n" + "=" * 100)
    print("DOCUMENT:", doc_id)
    print("Compacted tokens:", token_count)
    print("Validation target:", is_validation)
    print("=" * 100)

    print("\nFIRST 1500 CHARACTERS:")
    print(compacted[:1500])

    print("\nLAST 800 CHARACTERS:")
    print(compacted[-800:])
