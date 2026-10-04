# Independent teammate check

Use a different physical computer if available. This is an unsigned local review candidate, not a newly published submission. Record actual results; do not mark a step passed unless you performed it.

## Preparation

1. Extract the candidate ZIP into a new directory. Do not copy the lead's `.venv`, model cache or credentials.
2. Record OS/version, CPU, RAM, Python version and whether Hugging Face models were already cached.
3. Use Python 3.11. Create a virtual environment and install `requirements.txt` using the main README instructions.
4. Run `python -m pip check`, authenticate with your own Hugging Face account if needed, and accept the EmbeddingGemma terms yourself.

## Start the full-corpus candidate

The candidate ZIP includes `runtime_index/full` and `runtime_index/git-review`. From its root, set these variables before running the launcher.

macOS/Linux:

```bash
export PRISM_DEVICE=cpu
export PRISM_INDEX_DIR=runtime_index/full
export PRISM_VERSION_INDEX=runtime_index/git-review
python launch_demo.py
```

PowerShell:

```powershell
$env:PRISM_DEVICE="cpu"
$env:PRISM_INDEX_DIR="runtime_index/full"
$env:PRISM_VERSION_INDEX="runtime_index/git-review"
.\.venv\Scripts\python.exe launch_demo.py
```

Use `PRISM_PORT=8002` if another PRISM server already occupies port 8000. In PowerShell set `$env:PRISM_PORT="8002"`. A first run may take minutes to download models.

## Functional checks

- `/health` reports `device: cpu`, `corpus_documents: 8765`, and `corpus_scope: official-full`.
- Leave “Search in” at automatic. Try `Find code that validates a tic-tac-toe board state` and inspect the returned source.
- Try `which files import collections?`. Every shown evidence entry should be an import rather than an unrelated call.
- Try `which functions call range before print?`. Inspect the function scope and source-line order. Do not claim this proves runtime order.
- Choose “All historical versions” and search `find the API health endpoint`. Both indexed `api.py` revisions should be available.
- Choose “Latest indexed snapshot” and repeat. Only the latest file state should be returned.
- Choose the earliest Git commit and repeat. The older implementation should be returned, not future changes.
- Try a blank query and `show dependency graph`. Expect helpful validation or an unsupported-feature explanation, not a fabricated result.
- Enter five questions of your own. For each, record the returned source and explain whether it answers your question; “returned something” is insufficient.
- Run `python -m unittest discover -s tests -v`. This uses fake embeddings for unit fixtures, not a new model benchmark.

## Report back

Send: hardware/OS/Python; cache status; installation time and errors; health output; test summary; one cold and one repeated-query timing; your five questions and relevance judgments; screenshots of any failure. Do not include passwords, tokens or private source code.

This report is evidence only after the teammate completes it. The main team's local checks do not substitute for this independent run.
