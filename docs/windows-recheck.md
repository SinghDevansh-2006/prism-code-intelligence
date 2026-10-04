# Windows recheck of the revised candidate

Extract the newly supplied candidate ZIP into a new directory. Do not retest the old ZIP. Record its SHA-256 with `Get-FileHash <zip-path> -Algorithm SHA256`. The internal `CANDIDATE_MANIFEST.json` identifies the packaged files. This remains an unpublished candidate.

Use the existing working Python 3.11 installation and dependencies, or follow README setup. If `python` opens the Microsoft Store, use `py -3.11` or the explicit installed Python path to create the environment. Do not copy credentials or another person's virtual environment.

From the new candidate root in PowerShell (with its virtual environment installed):

```powershell
New-Item -ItemType Directory -Force cache/mteb | Out-Null
$env:MTEB_CACHE=(Resolve-Path cache/mteb).Path
$env:PRISM_DEVICE="cpu"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mteb_adapter.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PRISM_INDEX_DIR="runtime_index/full"
$env:PRISM_VERSION_INDEX="runtime_index/git-review"
$env:PRISM_PORT="8002"
.\.venv\Scripts\python.exe launch_demo.py
```

Expected: 4 focused adapter tests and 23 total tests pass, with no WinError 32. Stop any older PRISM server on port 8002 before launching, so the browser tests the new code.

In the UI:

1. Submit `show dependency graph`. Expect an unsupported-feature explanation, “Revise your request” and “Request needs revision”; no backend-retry suggestion.
2. Submit `which files import collections?`. Confirm results and import evidence appear after the rejected request.
3. Confirm the relevance/correctness guidance is visible on the welcome screen or result evidence.
4. Run one semantic, structural and version-history query to check for regressions.

Send the ZIP hash, OS/Python, both test outputs, UI results and any errors. State whether this was a different physical computer and whether models were already cached. No full embedding regeneration is needed for this file-cleanup/UI patch. Report actual results, not expected results.
