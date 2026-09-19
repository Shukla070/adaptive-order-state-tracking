#!/usr/bin/env bash
#
# Record the exact working environment, so it can be rebuilt rather than guessed.
#
# RUN THIS IN THE OLD FOLDER, WITH THE WORKING VENV ACTIVE:
#
#     mp                      # activates .venv in DeltaProduct
#     bash work/capture_env.sh
#
# It writes two files into the NEW repository:
#
#   requirements.lock.txt   every package at its exact installed version
#   ENVIRONMENT.md          python / CUDA / driver versions, and the exact
#                           commands that rebuild this environment
#
# It installs nothing and changes nothing.
#
set -euo pipefail

DEST="${DEST:-$(cd .. && pwd)/adaptive-order-state-tracking}"
[ -d "$DEST" ] || { echo "ERROR: destination not found: $DEST" >&2; exit 1; }

command -v python >/dev/null || { echo "ERROR: no python on PATH — run 'mp' first" >&2; exit 1; }
python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" \
  || { echo "ERROR: no virtualenv active — run 'mp' first" >&2; exit 1; }

echo "capturing from : $(python -c 'import sys; print(sys.prefix)')"
echo "writing into   : $DEST"

# ---------------------------------------------------------------- the versions
PY_VER="$(python -c 'import sys; print(sys.version.split()[0])')"
TORCH_VER="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?')"
CUDNN="$(python -c 'import torch; print(torch.backends.cudnn.version())' 2>/dev/null || echo '?')"
TRITON_VER="$(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
GPU_NAME="$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo '?')"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
SMI_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1 || echo '?')"
OS_DESC="$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -sr)"
KERNEL="$(uname -r)"

# the torch wheel index is NOT recoverable from pip freeze; derive it from the
# local version suffix (e.g. 2.5.1+cu121 -> cu121), which is how it was installed
TORCH_LOCAL="$(printf '%s' "$TORCH_VER" | sed -n 's/.*+\(.*\)/\1/p')"
if [ -n "$TORCH_LOCAL" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_LOCAL}"
  TORCH_INSTALL="pip install torch==${TORCH_VER} --index-url ${TORCH_INDEX}"
else
  TORCH_INDEX="(default PyPI — the wheel carries no +cuXXX suffix)"
  TORCH_INSTALL="pip install torch==${TORCH_VER}"
fi

# ------------------------------------------------------------------ the lock
# pip freeze records a local editable install as an absolute path belonging to
# THIS folder. That path is wrong everywhere else, so it is stripped here and
# reinstalled explicitly from ./flash-linear-attention during setup.
{
  echo "# Exact versions of the environment in which every result in results/ was"
  echo "# produced. Captured $(date -u '+%Y-%m-%d %H:%M UTC') from the working installation."
  echo "#"
  echo "# This file is self-sufficient: the --extra-index-url below is what makes"
  echo "# the pinned torch build resolvable. Installing torch from plain PyPI gives"
  echo "# a DIFFERENT build carrying the same version number, which fails silently"
  echo "# rather than loudly."
  echo "#"
  echo "# flash-linear-attention is NOT here. It is the patched copy vendored in"
  echo "# this repository and is installed editable:"
  echo "#     pip install --no-build-isolation -e ./flash-linear-attention"
  echo ""
  [ -n "$TORCH_LOCAL" ] && { echo "--extra-index-url ${TORCH_INDEX}"; echo ""; }
  # causal-conv1d compiles a CUDA extension and needs torch at BUILD time, so
  # it cannot be resolved from a requirements file under pip's default build
  # isolation. It is REQUIRED -- ShortConvolution raises without it -- so it is
  # recorded here as a comment and installed by setup_env.sh in its own step.
  pip freeze --exclude-editable 2>/dev/null \
    | grep -v -i '^flash.linear.attention' \
    | grep -v '^-e ' \
    | grep -v ' @ file://' \
    | sed 's/^\(causal.conv1d==.*\)$/# \1   # REQUIRED; installed separately with --no-build-isolation, see ENVIRONMENT.md/'
} > "$DEST/requirements.lock.txt"

PKG_COUNT="$(grep -vc '^#' "$DEST/requirements.lock.txt" || true)"

# --------------------------------------------------------------- the readme
cat > "$DEST/ENVIRONMENT.md" <<EOF
# Environment

The exact environment in which every result under \`results/\` was produced.
Captured $(date -u '+%Y-%m-%d %H:%M UTC') from the working installation.

## Versions

| | |
|---|---|
| Python | ${PY_VER} |
| PyTorch | ${TORCH_VER} |
| PyTorch CUDA runtime | ${TORCH_CUDA} |
| cuDNN | ${CUDNN} |
| Triton | ${TRITON_VER} |
| GPU | ${GPU_NAME} |
| NVIDIA driver | ${DRIVER} |
| Driver CUDA | ${SMI_CUDA} |
| OS | ${OS_DESC} |
| Kernel | ${KERNEL} |
| Packages pinned | ${PKG_COUNT} |

\`requirements.lock.txt\` holds every package at its exact version.

## Rebuilding

Order matters. PyTorch must be installed from its own wheel index before
anything else pulls in a different build, and \`flash-linear-attention\` is a
local editable install of the vendored, patched copy in this repository — not
the PyPI package of the same name.

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1. PyTorch first, from the index matching the CUDA build above
${TORCH_INSTALL}

# 2. everything else, pinned. requirements.lock.txt carries its own
#    --extra-index-url, so this step also works on its own if step 1 is skipped
pip install -r requirements.lock.txt

# 3. the vendored, patched flash-linear-attention (editable, from this repo)
pip install --no-build-isolation -e ./flash-linear-attention

# 4. anything beyond upstream's requirements
bash work/install_deps.sh
\`\`\`

\`work/setup_env.sh\` performs exactly these steps.

## Verifying

\`\`\`bash
python work/check_env.py
\`\`\`

Compares the live environment against \`requirements.lock.txt\` and reports any
package whose version differs, so a rebuild that silently drifted is visible
before it costs GPU hours. Then:

\`\`\`bash
python work/check_gate_precision.py    # CPU, seconds
python work/check_gate_init.py         # needs the GPU
\`\`\`

## Dependencies that need care

- **torch** must come from \`${TORCH_INDEX}\`. The same version number exists on
  PyPI as a different build; installing that one fails at kernel launch, not at
  install time.
- **abstract_algebra** and **sfirah** are git URLs pinned to exact commits.
  They are required by \`state_tracking/src/generate_data.py\`, \`main.py\` and
  \`work/check_data_histogram.py\`. Installing them needs network access to
  github.com; upstream's \`requirements.txt\` does not pin them, this lock does.
- **causal-conv1d** is required and is installed in its own step.
  \`fla/modules/convolution.py\` raises \`RuntimeError\` at construction unless it
  is importable, and \`GatedDeltaProduct\` builds a \`ShortConvolution\` with
  \`use_short_conv=True\` by default, so every result in \`results/\` was produced
  on that path. It compiles a CUDA extension and needs torch visible at build
  time, which pip's default build isolation prevents:
  \`pip install --no-build-isolation causal-conv1d==1.7.0\`.

## Notes carried from experience

- \`chunk_gated_delta_rule\` asserts a non-float32 input, so the layer runs in
  bfloat16. The halting gate is held in float32 deliberately; see the PRECISION
  note in \`src/gating.py\`.
- The bfloat16 kernels use non-deterministic atomics, so identical
  configurations differ by roughly 0.005 token accuracy on stable cells, and far
  more on bistable ones.
- \`state_tracking/data/\` (about 246 MB of generated CSVs) is not tracked.
  Either copy it from a previous working folder or regenerate it with
  \`work/gen_heterogeneous.py\`; regeneration takes roughly 15 minutes per
  dataset.
EOF

echo
echo "written:"
echo "  $DEST/requirements.lock.txt   (${PKG_COUNT} packages)"
echo "  $DEST/ENVIRONMENT.md"
echo
echo "key versions: python ${PY_VER} · torch ${TORCH_VER} (cuda ${TORCH_CUDA}) · triton ${TRITON_VER}"
echo "torch install line for the rebuild:"
echo "  ${TORCH_INSTALL}"
