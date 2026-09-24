import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from src.code_normalizer import compact_large_literals

MODEL = "Qwen/Qwen3-Embedding-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

corpus = load_dataset(
    "CoIR-Retrieval/apps",
    "corpus",
    split="corpus"
)

train_corpus = [
    row for row in corpus
    if row["partition"] == "train"
]

original_lengths = []
compacted_lengths = []
changed_docs = 0

for i, row in enumerate(train_corpus, 1):
    original = row["text"]
    compacted = compact_large_literals(original)

    if compacted != original:
        changed_docs += 1

    original_tokens = tokenizer(
        original,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    compacted_tokens = tokenizer(
        compacted,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    original_lengths.append(len(original_tokens))
    compacted_lengths.append(len(compacted_tokens))

    if i % 1000 == 0:
        print(f"Processed {i}/{len(train_corpus)}")

original_lengths = np.array(original_lengths)
compacted_lengths = np.array(compacted_lengths)

print("\nDOCUMENTS CHANGED BY COMPACTION:", changed_docs)

print("\nORIGINAL")
print("Max:", int(original_lengths.max()))
print("Over 2048:", int((original_lengths > 2048).sum()))
print("Over 4096:", int((original_lengths > 4096).sum()))

print("\nCOMPACTED")
print("Max:", int(compacted_lengths.max()))
print("Over 2048:", int((compacted_lengths > 2048).sum()))
print("Over 4096:", int((compacted_lengths > 4096).sum()))

print("\nTOP 10 TOKEN REDUCTIONS")
reductions = original_lengths - compacted_lengths
top_indices = np.argsort(reductions)[-10:][::-1]

for idx in top_indices:
    row = train_corpus[int(idx)]
    print(
        f'{row["_id"]:8} | '
        f'original={original_lengths[idx]:7} | '
        f'compacted={compacted_lengths[idx]:7} | '
        f'reduced={reductions[idx]:7}'
    )
