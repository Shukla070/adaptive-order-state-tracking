#!/usr/bin/env bash
#
# Rebuild the recorded environment in this repository.
#
# Run it from the repository root, with no virtualenv active:
#
#     cd /path/to/adaptive-order-state-tracking
#     bash work/setup_env.sh
#
# Requires requirements.lock.txt and ENVIRONMENT.md, produced by
# work/capture_env.sh from a working installation. Without them this refuses to
# run rather than installing whatever happens to be current, because an
# environment assembled from latest-of-everything cannot be compared with the
# numbers already in results/.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -f requirements.lock.txt ] || die "requirements.lock.txt not found.
       Capture it from the working environment first:
           cd ../DeltaProduct && mp && bash work/capture_env.sh"
[ -d flash-linear-attention ] || die "flash-linear-attention/ not found in $REPO"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  die "a virtualenv is already active ($VIRTUAL_ENV).
       Run 'deactivate' first — this script creates its own."
fi

# the torch line recorded by capture_env.sh, index URL included
TORCH_LINE="$(sed -n 's/^\(pip install torch==.*\)$/\1/p' ENVIRONMENT.md | head -1)"
[ -n "$TORCH_LINE" ] || die "could not find the torch install line in ENVIRONMENT.md"

say "Plan"
echo "  repo        $REPO"
echo "  torch       $TORCH_LINE"
echo "  pinned      $(grep -vc '^#' requirements.lock.txt) packages"
echo
read -r -p "  proceed? [y/N] " reply
[ "$reply" = "y" ] || [ "$reply" = "Y" ] || die "cancelled"

say "Creating .venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
echo "  $(python -c 'import sys; print(sys.version.split()[0])') at $(which python)"

say "Installing PyTorch from its recorded wheel index"
eval "$TORCH_LINE"

say "Installing the pinned packages"
pip install -r requirements.lock.txt

say "Building causal-conv1d (CUDA extension, needs torch at build time)"
# Required, not optional: fla/modules/convolution.py raises at construction
# unless this is importable, and GatedDeltaProduct builds a ShortConvolution
# by default. --no-build-isolation is what lets its setup.py see the torch
# installed above; with pip's default isolation the build fails.
# This compiles and can take several minutes.
pip install --no-build-isolation causal-conv1d==1.7.0

say "Installing the two git dependencies, without their declared deps"
# sfirah declares s4, which declares gluonts, which caps pandas<3 and
# numpy<2.5 -- against the pandas 3.0.5 / numpy 2.5.2 this project runs on.
# Neither s4 nor gluonts was ever installed in the environment that produced
# the results, and nothing imports them. Their real runtime dependencies are
# already pinned above, so --no-deps installs exactly what was there before.
pip install --no-deps \
  git+https://github.com/jopetty/abstract_algebra@94226a088037c1c8a68848bd2a0749d5bdfff2c2 \
  git+https://github.com/jopetty/sfirah@30b5a87342882f42dc3ed4136ee1ade3633c266e

say "Installing the vendored, patched flash-linear-attention (editable)"
pip install --no-build-isolation -e ./flash-linear-attention

if [ -f work/install_deps.sh ]; then
  say "Running work/install_deps.sh"
  bash work/install_deps.sh
fi

say "Verifying"
python work/check_env.py || {
  echo
  echo "The environment does not match what was recorded. Read the FAIL lines"
  echo "above before running any experiment here."
  exit 1
}

cat <<'EOF'

Environment ready. Next:

    source .venv/bin/activate
    python work/check_gate_precision.py     # CPU, seconds
    python work/check_gate_init.py          # needs the GPU

state_tracking/data/ is not tracked. Copy it rather than regenerating:

    cp -r ../DeltaProduct/state_tracking/data state_tracking/
EOF
