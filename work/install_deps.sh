#!/usr/bin/env bash
# Week 1: remaining dependencies for state_tracking/src/{generate_data,main}.py
# Run with the venv ACTIVE, from the repo root.
set -e
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"

# plain PyPI deps that main.py / generate_data.py import
pip install fire humanize polars pyrootutils ordered-set torcheval python-dotenv scikit-learn

# git-only deps. --no-deps is deliberate: sfirah[ssm] will otherwise drag in an
# old torch and re-download several hundred MB of CUDA wheels.
pip install --no-deps "git+https://github.com/jopetty/abstract_algebra"
pip install --no-deps "git+https://github.com/jopetty/sfirah"

echo
echo "--- import check ---"
python - << 'PY'
mods = ["fire","humanize","polars","pyrootutils","ordered_set","torcheval",
        "dotenv","sklearn","abstract_algebra","sfirah","fla","torch","triton"]
bad = []
for m in mods:
    try:
        __import__(m); print(f"  ok    {m}")
    except Exception as e:
        bad.append((m, e)); print(f"  FAIL  {m}: {type(e).__name__}: {e}")
print("\nALL IMPORTS OK" if not bad else f"\n{len(bad)} MISSING — paste these to Claude")
PY
