import numpy as np
from transformers import AutoTokenizer
from src.data_loader import load_dev_benchmark

MODEL = "Qwen/Qwen3-Embedding-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

queries, corpus, _ = load_dev_benchmark()

def get_lengths(rows, label):
    lengths = []

    print(f"Tokenizing {label}...")

    for i, row in enumerate(rows, 1):
        tokens = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=False
        )

        lengths.append(len(tokens["input_ids"]))

        if i % 1000 == 0:
            print(f"  processed {i}/{len(rows)}")

    lengths = np.array(lengths)

    print(f"\n{label} TOKEN LENGTHS")
    print("Count:", len(lengths))
    print("Min:", int(lengths.min()))
    print("Median:", int(np.percentile(lengths, 50)))
    print("P90:", int(np.percentile(lengths, 90)))
    print("P95:", int(np.percentile(lengths, 95)))
    print("P99:", int(np.percentile(lengths, 99)))
    print("Max:", int(lengths.max()))
    print("Over 2048:", int((lengths > 2048).sum()))
    print("Over 4096:", int((lengths > 4096).sum()))
    print("Over 8192:", int((lengths > 8192).sum()))


get_lengths(queries, "VALIDATION QUERIES")
get_lengths(corpus, "TRAIN CORPUS")
