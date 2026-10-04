# Windows teammate review — 3 October 2026

## Scope and evidence

The teammate tested candidate SHA-256 `dc86be694ffeb7f0e66c14336084d6654d26e86e762dd35e2f35daf0bf3131a4` on Windows 11 Home Single Language (build 26200), Intel i5-13420H, 15.65 GiB RAM, isolated Python 3.11.17. Dependencies installed unchanged and `pip check` passed. Pinned models were already cached: this is not evidence of a fresh model download or new Hugging Face authentication. Distinct physical-machine independence was not explicitly attested. Raw evidence remains with the team; private local paths are omitted here.

- Report SHA-256: `77f4136803fa9442b45c18722b0778f7209877353da9e012135eeb8ddb258fd1`
- Evidence ZIP SHA-256: `1522599ac3ffe868e569f0f54f8df9bad80160be3ef98787aba41a74b14dfe09`; ZIP integrity checked.

## Passed checks

- Real CPU API and UI searches worked across semantic, exact-usage, structural and version-history routes.
- Full corpus and structural index contained 8,765 documents; six index-manifest hashes matched. Embeddings were finite, unit-normalized float32 arrays of dimensions 768 and 1,024.
- All/latest/selected-commit filtering worked across seven indexed snapshots and two actual `api.py` revisions.
- Blank and unsupported graph requests correctly returned HTTP 422. Launcher warmup succeeded.
- Example semantic request took 13.991 seconds cold with cached weights; repeat took 0.259 seconds. These are individual observations, not general latency guarantees.

## Findings and disposition

1. **Windows cleanup error:** 20 of 21 tests passed. One test failed while removing a still-open NumPy memory mapping (`WinError 32`), after its ranking assertion passed. A focused rerun reproduced it. The evaluator now owns and closes cached mappings on normal and exceptional exits. Two cleanup regressions were added; all 23 tests pass locally on macOS. **Resolved: the 4 October Windows rerun passed all 23 tests; see below.**
2. **Unsupported-request wording:** HTTP behavior was correct, but the UI suggested retrying a ready backend. It now says to revise the request; a browser check confirmed this and successful recovery with an import search.
3. **Relevance is not correctness:** retrieved tic-tac-toe source `d1896` returns True for `["XXX", "OOO", "XOX"]`, although both players win. The official source corpus remains unchanged. UI guidance now explicitly asks users to review source before reuse. Five manual relevance checks are qualitative examples, not a benchmark accuracy estimate.
4. **Environment restrictions:** Store Python alias, sandbox network restrictions and an unwritable default MTEB cache required setup adjustments. They are not retrieval failures. The recheck instructions use a workspace cache and explicit virtual-environment Python.

No new full MTEB benchmark was performed on Windows. The mapping cleanup and UI wording do not alter ranking mathematics, corpus data or frozen submission results. The final workflow includes Windows tests; see GitHub Actions for current status.

See [targeted Windows recheck](windows-recheck.md). Publication was authorized on 4 October 2026. CI, release assets and final-tag alignment are checked during publication.

## Successful recheck — 4 October 2026

A second teammate tested the exact revised candidate (`cb77efdbc0230b0342b6285770b687a5af856170776940559a0f079f9a002b72`) on an Intel i7-13650HX Windows computer with 15.78 GiB RAM and isolated Python 3.11.0. Observed hardware differs from both the development Mac and the first Windows reviewer; physical ownership was not independently attested.

All 157 manifest hashes matched before and after testing. Dependency checks, 4 focused adapter tests and all 23 tests passed, with no WinError 32. Real CPU operation covered 8,765 documents and all four routes. Unsupported-query recovery, source evidence, version filters and served-HTML identity passed. No packaged source changed.

Pinned model weights were already cached. Uvicorn was used directly; the launcher, fresh model download, full benchmark and GitHub CI were not repeated. Cold/warm semantic samples were 5.676/0.200 seconds, not general performance guarantees. The requested teammate recheck is complete. The final package uses the same verified index bytes and corrected code, with updated defaults and submission materials. GitHub Actions and the release track publication checks.

[Machine-readable record and original evidence hashes](review/windows-recheck.json). Original report and raw evidence are retained privately to avoid publishing machine-specific paths.
