# Pre-submission review evidence

Review updated: 3 October 2026. Base commit: `094317cd11b1feecba594e2b12faf21a5908d3d2`.
This record preserves the staged reviews against that base commit. Publication was authorized on 4 October 2026. The final package updates the presentation, demo link and signed disclosure; frozen submission rankings remain unchanged. Historical run descriptions below are not claims about the latest GitHub CI.

## What changed and why

| Change | Evidence | Judging benefit and limit |
|---|---|---|
| Strict ranking validation | All 3,765 query IDs and 100 unique corpus IDs per query validated against 8,765 official documents. Rejects duplicate JSON keys, missing/extra queries, invalid IDs and malformed lists. | Makes reported accuracy auditable; does not increase accuracy. |
| API resilience | Whitespace requests rejected, setup failures return 503, concurrent lazy initialization tested. | Reduces avoidable failures and duplicate model loads during trial use. General inference concurrency and resource exhaustion are not load-tested. |
| Symbolic precision | Case/boundary matching, import-only evidence, function-scoped calls, all requested import/call terms required, single-letter identifiers supported. Unsupported graph queries produce a clear error. | Reduces false evidence for unseen requests. Static matching still cannot resolve every alias or prove runtime order. |
| Real Git history | Read-only importer; actual `api.py` history and temporary Git add/edit/delete fixture tested. Repeat import is idempotent; attempts to append older or divergent history are rejected. Identical embeddings reused, deletion markers excluded from code hits, failed writes rolled back. | Supports version-retrieval claims with real history. Bounded first-parent file history; latest and selected-commit snapshots supported; no rename tracking, multi-writer support or crash-proof transactions. |
| MTEB adapter | Cached-array and fresh CPU evaluations completed through `mteb.evaluate`. The fresh run regenerated all document and query arrays, with same-run checkpoint reuse. | Adds CPU evidence; an audited precision mismatch was repaired, and the corrected float32 arrays were rescored over the entire test set. See the audit below. |
| Reproducibility and CI | Empty Python environment installed README requirements, `pip check` passed. Isolated source copy passed 16 route requests with new model downloads. 21 focused tests and existing four-route regression passed. | Stronger local setup evidence; the second teammate Windows recheck passed all 23 tests using cached models (see Windows follow-up). Updated Linux CI requires a future approved push. |
| UI and documentation consistency | Exact-search explanation updated to the actual matcher. Git history distinguished from the synthetic fixture, commit IDs abbreviated with full IDs in tooltips. | Judges see accurate provenance. Core demo workflow and frozen ranking configuration remain intact. |

## Benchmark evidence

The frozen ranking SHA256 remains `519fe7b0ad8bb4ae80f1da5192ceea067bc1c68b74cad6328ee7ac8a2de644d4`.

| Evaluation | NDCG@10 | MRR@10 | Method |
|---|---:|---:|---|
| Submitted frozen ranking order | 0.853100 | 0.821803 | Independently recomputed from the ordered JSON |
| New MTEB adapter, cached embeddings | 0.85287 | 0.821493 | Recomputed floating-point scores; MTEB ranking and tie handling |

These are distinct evaluations, not interchangeable files. Tied and nearly tied document scores can change ordering through sorting and floating-point operations. For example, `q7095` assigns identical scores to `d7095` and `d7150`; `q8121` has a difference around 2.4e-7 between its leading candidates in the new run. Do not choose tie handling or model weights based on which improves test scores.

The cached run took about 29.2 seconds, including evaluation overhead but **excluding fresh model inference**. Result and array-hash provenance: [result](review/mteb-cached-result.json), [provenance](review/mteb-cached-provenance.json).

The full CPU run completed on 3 October 2026 (India time), covering 8,765 documents and 3,765 test queries for both models. It reused checkpoints from the same fresh regeneration, not the original benchmark arrays. **Historical precision issue (now resolved):** the first 256 length-sorted Qwen document rows were generated before explicit float32 was configured; later documents and all Qwen queries used explicit float32. Checkpoint signatures did not include dtype. The earlier scores do not establish uniform precision. The follow-up regenerated all 256 affected rows in float32 and reran full MTEB scoring, resolving this limitation.

| Metric | Frozen submission | Final uniform float32 CPU result |
|---|---:|---:|
| NDCG@10 | 0.853100 | 0.853220 |
| MRR@10 | 0.821803 | 0.822044 |
| MRR@100 | 0.824138 | 0.824396 |
| Recall@10 | 0.949535 | 0.949270 |
| Recall@20 | 0.973440 | 0.973710 |
| Recall@100 | 0.993625 | 0.993630 |

NDCG and MRR are slightly higher; Recall@10 is slightly lower. These numerical differences are not proof of an improved algorithm. MTEB rounds some metrics to five decimal places. The frozen artifacts remain authoritative for the existing submission.

Historical mixed-precision evidence: [complete MTEB TaskResult](review/mteb-fresh-result.json), [original run provenance](review/mteb-fresh-provenance.json), and [artifact hashes, coverage, array checks and precision caveat](review/mteb-fresh-audit.json). All 3,765 prediction query IDs match the saved query IDs; each has 1,000 finite scores for valid corpus IDs. All four embedding arrays have the expected dimensions and finite values. Qwen's early document chunks have norms between 0.998089 and 1.003802; later chunks are unit-normalized within float32 tolerance.

The recorded 7,580.251 seconds (about 126 minutes) covers only the final resumed invocation. It excludes earlier inference and interruptions and must not be presented as total regeneration time. A separate local `PRISM_Fresh_CPU_Benchmark_Evidence.zip` preserves predictions, all four arrays, ID lists, checkpoint manifests, result, provenance and audit. Large arrays and predictions are kept outside the source repository; their SHA-256 hashes are recorded in the audit.


## CPU and setup evidence

macOS 26.5.1, ARM64, Python 3.11.16; CPU inference. New virtual environment installed `requirements.txt` and passed `pip check`. An isolated copy included tracked source plus the local review scripts, without the development virtual environment or embedding caches. This was not a second physical computer or an unmodified clone of the published tag.

The fresh Hub-cache run downloaded both models, recovered from one transient connection failure, and completed 16 searches: two requests for each of eight queries across all four routes. The first semantic request, including downloads and model loading, took 192.8 seconds. Subsequent semantic requests were approximately 428–576 ms; exact/structural 2–23 ms; evolution 39 ms. Process peak RSS was about 3.0 GiB. These few sequential requests are not a load test or a statistically robust latency benchmark.

The extra semantic questions asked about merging intervals, balanced parentheses and shortest paths. They check route behavior and return results, but have no independently annotated relevance judgments. No generalization-accuracy percentage is claimed. Raw samples: [fresh downloads](review/cpu-fresh-download.json), [existing model cache](review/cpu-cached.json).

Actual repository history indexed two `api.py` revisions in approximately 1.52 and 0.57 seconds after model loading; each stored vector uses 3,072 bytes. [Recorded events](review/git-history.json). The first selected revision represents the import window baseline, not necessarily the file's original creation.

## Re-run locally

```bash
python -m unittest discover -s tests -v
python test_full_agent_regression.py
python verify_submission.py
python benchmark_review.py --output cache/review/my-cpu-check.json
python index_git_history.py --repo . --path api.py --output cache/git-history
python evaluate_mteb_pipeline.py --output cache/review/new-mteb-run
```

Full MTEB mode needs model access, substantial CPU time and memory. Add `--embedding-cache cache/official` only when those original arrays exist; that mode records cache reuse explicitly. Metric verification also accepts `--qrels-arrow` and `--corpus-arrow` for offline official Arrow datasets.

## Submission implications

- The existing demo's semantic, import, call-order and synthetic-history workflows passed regression. No mandatory re-recording follows from these fixes. A new Git-history demonstration would require new footage only if the team chooses to showcase it.
- Keep the existing PPT metrics tied explicitly to the frozen submitted ranking file. Do not silently replace them with a different evaluation's values.
- The signed AI disclosure remains unchanged at the user's request; this review does not certify its completeness.
- Publication is authorized. The release tag must identify the final published commit; current execution status is available in GitHub Actions.
- These changes strengthen correctness, reproducibility and evidence. They cannot guarantee a Top 15 place, and they do not establish a measured retrieval-accuracy improvement.

## Full-corpus and version-scope candidate

The local full runtime contains all 8,765 official documents. Five sampled official questions whose frozen top result was outside the old 5,000-document index matched that top result through the actual API. This is integration evidence, not a new aggregate accuracy measurement. Long APPS questions reached about 28 seconds under concurrent benchmark load; earlier short-query timings are not representative of every request.

The UI now selects automatic routing, all historical versions, latest snapshot, or a specific indexed Git commit. Seven real repository snapshots cover two changed `api.py` implementations; unchanged files retain their last applicable revision. Unit fixtures cover deletion and invalid commits.

A predeclared validation-only reciprocal-rank fusion candidate was rejected: NDCG@10 fell from 0.917734 to 0.896828, with paired bootstrap delta interval [-0.028064, -0.013733]. MRR@100 also fell. The frozen 65/35 score fusion remains unchanged. This protects baseline performance; no accuracy gain is claimed.

Windows teammate results have been received; see [Windows review](windows-review.md). The patched Windows rerun passed on different observed hardware; physical ownership was not independently attested.

## Final local packaging checks

The extracted teammate ZIP passed all focused unit tests and a real API import-search smoke test in the clean Python environment. Health reported 8,765 documents; seven Git snapshots were listed; exact search returned five results. ZIP CRC and every manifest SHA-256 were verified. This is same-machine isolation, separate from the subsequently received Windows teammate evidence.

Repeated bounded Git imports now preserve unchanged file revisions as the import window advances. A regression fixture checks unrelated commits, then an actual code edit, without duplicate version records.

MRR cutoffs are kept explicit: frozen submitted rankings yield MRR@10 0.821803 and MRR@100 0.824138. The cached MTEB adapter yields MRR@10 0.821493, MRR@100 0.823828 and MRR@1000 0.823858. The supplied theme text does not specify the MRR cutoff; none is silently substituted for another.

## CPU regeneration follow-up

The regenerated Gemma document matrix contains 8,765 vectors. Its mean cosine similarity to the original matrix is 1.0 at float32 precision; minimum per-row similarity is 0.9999996423721313. This verifies embedding agreement, not completed fused-ranking accuracy.

The full evaluator now loads one model at a time, retains resumable chunks, uses Qwen batch size 1, and pins both Hub revisions. The app loaders use the same pinned revisions. All 21 focused tests still pass. Document and query generation are complete. The affected rows have since been regenerated and full MTEB scoring repeated; the uniform-precision follow-up below is complete.

## Final precision correction and verification

All 256 affected Qwen document embeddings were regenerated with the pinned model revision on CPU in explicit float32. The other 8,509 Qwen document rows and the three already-float32 arrays were retained from the preceding fresh CPU generation. Hashes and exact array comparisons confirm the retained values were unchanged. This avoids unnecessary recomputation; it does not reuse the original submitted embedding cache.

All 3,765 queries were then scored through `mteb.evaluate` against the complete corrected 8,765-document corpus. Final NDCG@10 is 0.853220, MRR@10 0.8220444149328615, MRR@100 0.8243958749403522 and MRR@1000 0.8244257678276949. The table above now shows this corrected evaluation; earlier mixed-precision results remain archived as historical evidence, not the final result.

- [Uniform TaskResult](review/mteb-uniform-result.json)
- [Final scoring provenance](review/mteb-uniform-provenance.json)
- [Regenerated IDs and retained-array lineage](review/mteb-uniform-repair.json)
- [Precision, normalization, coverage and SHA-256 audit](review/mteb-uniform-audit.json)

The scoring-stage provenance marks cached arrays because it consumes the verified corrected arrays; preceding CPU inference and the repair are recorded separately. No combined wall-clock regeneration time is claimed. PyTorch 2.14.0, Transformers 5.17.0, Sentence Transformers 6.1.0, NumPy 2.4.6 and MTEB 2.21.6 match between original and repair environments.

Checkpoint signatures now include dtype and device, with a regression test rejecting precision changes. Both live model loaders explicitly select float32. All 21 focused tests pass. The local review ZIP uses the corrected full-corpus index. The complete corrected arrays, predictions and provenance are preserved in `PRISM_Uniform_CPU_Benchmark_Evidence.zip` outside the source repository.

The precision issue is resolved. The Windows teammate report is recorded below; the targeted Windows post-fix rerun passed; final publishing/CI/tag checks are tracked through the GitHub release and Actions.

The corrected full-corpus API passed five semantic parity examples (all five match corrected uniform-benchmark top-1 IDs; four match frozen top-1 IDs), all/latest/selected-commit version searches, and invalid-commit validation. These are integration checks, not additional aggregate accuracy claims. [API verification record](review/uniform-runtime-verification.json).

## Windows follow-up

The teammate verified all four CPU routes, full-index integrity and version filtering on Windows 11. One test exposed an open memory-map cleanup failure. The fix explicitly closes mappings, including exceptional exits; 23 local tests now pass. UI request-revision guidance is corrected. See [evidence and limitations](windows-review.md) and [targeted rerun instructions](windows-recheck.md). The publication workflow includes Windows regression checks; see GitHub Actions for their result. Historical test counts above describe their respective earlier runs.

On 4 October, the revised candidate passed 4/4 adapter tests, 23/23 total tests and all four live routes on the second Windows machine. All 157 file hashes matched; no cleanup error remained. [Recorded verification](review/windows-recheck.json). Teammate verification is complete within the documented cached-model scope.
