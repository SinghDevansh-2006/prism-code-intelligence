#!/usr/bin/env bash
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="${PRISM_PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo 'Create .venv and install requirements.txt first, or set PRISM_PYTHON.' >&2
    exit 1
fi
export PRISM_DEVICE="${PRISM_DEVICE:-cpu}"
exec "$PYTHON" launch_demo.py
