# Environment

The exact environment in which every result under `results/` was produced.
Captured 2026-09-18 12:10 UTC from the working installation.

## Versions

| | |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.6.0+cu126 |
| PyTorch CUDA runtime | 12.6 |
| cuDNN | 90501 |
| Triton | 3.2.0 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| NVIDIA driver | 592.82 |
| Driver CUDA | 13.1 |
| OS | Ubuntu 26.04 LTS |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Packages pinned | 99 |

`requirements.lock.txt` holds every package at its exact version.

## Rebuilding

Order matters. PyTorch must be installed from its own wheel index before
anything else pulls in a different build, and `flash-linear-attention` is a
local editable install of the vendored, patched copy in this repository — not
the PyPI package of the same name.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1. PyTorch first, from the index matching the CUDA build above
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 2. everything else, pinned
pip install -r requirements.lock.txt

# 3. the vendored, patched flash-linear-attention (editable, from this repo)
pip install --no-build-isolation -e ./flash-linear-attention

# 4. anything beyond upstream's requirements
bash work/install_deps.sh
```

`work/setup_env.sh` performs exactly these steps.

## Verifying

```bash
python work/check_env.py
```

Compares the live environment against `requirements.lock.txt` and reports any
package whose version differs, so a rebuild that silently drifted is visible
before it costs GPU hours. Then:

```bash
python work/check_gate_precision.py    # CPU, seconds
python work/check_gate_init.py         # needs the GPU
```

## Notes carried from experience

- `chunk_gated_delta_rule` asserts a non-float32 input, so the layer runs in
  bfloat16. The halting gate is held in float32 deliberately; see the PRECISION
  note in `src/gating.py`.
- The bfloat16 kernels use non-deterministic atomics, so identical
  configurations differ by roughly 0.005 token accuracy on stable cells, and far
  more on bistable ones.
- `state_tracking/data/` (about 246 MB of generated CSVs) is not tracked.
  Either copy it from a previous working folder or regenerate it with
  `work/gen_heterogeneous.py`; regeneration takes roughly 15 minutes per
  dataset.
