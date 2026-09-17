#!/usr/bin/env bash
#
# Build a standalone repository from the current one.
#
# WHAT IT DOES
#   Creates a SIBLING directory containing a new git repo whose history is:
#     commit 1  the pristine upstream tree at d62241a, with its LICENSE, marked
#               plainly as third-party code
#     commit 2+ your work, in thematic commits
#
#   The result is a repo whose contributor list is you alone, while
#   `git diff <first commit>..HEAD` shows a reviewer exactly what you added.
#
# WHAT IT DOES NOT DO
#   It does not modify, move or delete the existing repository, and it does not
#   push anything. You review the result and push by hand.
#
# RUN IT FROM THE EXISTING REPO, in WSL (not from the Cowork shell, which
# cannot delete files and will leave git lock files behind):
#
#     mp
#     rm -f .git/index.lock
#     bash migrate_repo.sh
#
set -euo pipefail

NEW_NAME="${NEW_NAME:-adaptive-order-state-tracking}"
UPSTREAM_BASE="d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8"

OLD="$(pwd)"
NEW="$(cd .. && pwd)/${NEW_NAME}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preconditions
say "Checking preconditions"
[ -d .git ] || die "not a git repository: $OLD"
[ -f .git/index.lock ] && die "stale .git/index.lock — remove it first: rm -f .git/index.lock"
git cat-file -e "${UPSTREAM_BASE}^{commit}" 2>/dev/null \
  || die "upstream base ${UPSTREAM_BASE} not found. Run: git fetch upstream"
if [ -e "$NEW" ]; then
  if [ -z "$(ls -A "$NEW" 2>/dev/null)" ] || [ ! -e "$NEW/.git/HEAD" ] || \
     ! git -C "$NEW" rev-parse HEAD >/dev/null 2>&1; then
    die "$NEW exists and holds no commits — it is left over from a run that
       stopped early and is safe to delete. Remove it and re-run:
           rm -rf \"$NEW\"
       (That path is OUTSIDE your repository; nothing of yours is in it.)"
  fi
  die "$NEW already exists and contains commits. Remove it or set NEW_NAME=<other>."
fi
# Read the identity from THIS repo and carry it to the new one explicitly.
# Checking `git config user.name` here and relying on it later is wrong: it
# resolves against the current repo's local config, which says nothing about
# what a freshly initialised repo elsewhere will see.
GIT_NAME="$(git config user.name  || true)"
GIT_EMAIL="$(git config user.email || true)"
[ -n "$GIT_NAME" ]  || die "no git user.name found. Set one: git config --global user.name 'Your Name'"
[ -n "$GIT_EMAIL" ] || die "no git user.email found. Set one: git config --global user.email 'you@example.com'"
echo "  source      $OLD"
echo "  destination $NEW"
echo "  author      $GIT_NAME <$GIT_EMAIL>  (copied into the new repo)"

# --------------------------------------------------------- 1. upstream baseline
say "Commit 1 — vendoring the upstream tree at ${UPSTREAM_BASE:0:7}"
mkdir -p "$NEW"
git archive "$UPSTREAM_BASE" | tar -x -C "$NEW"
cd "$NEW"
git init -q -b main
# set identity on the NEW repo; it does not inherit the old repo's local config
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"

# a NOTICE file so the provenance is stated in the repo, not only in the README
cat > NOTICE <<'EOF'
This repository contains third-party code.

The initial commit of this repository is an unmodified snapshot of

    automl/DeltaProduct   (https://github.com/automl/DeltaProduct)
    commit d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8

released under the MIT License, which is reproduced in LICENSE and applies to
that code. It accompanies:

    Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., Grazzi, R.
    DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
    Products. NeurIPS 2025. arXiv:2502.10297.

The vendored copy of flash-linear-attention (flash-linear-attention/) is
likewise third-party code under its own licence.

Everything committed after the initial commit is the work of this project.
Modifications to third-party files are marked in-line with the comment
PATCH(adaptive-order), and the unmodified originals are retained alongside as
*.orig files.
EOF

git add -A
git -c commit.gpgsign=false commit -q -m "Vendor upstream DeltaProduct as the starting point

Unmodified snapshot of automl/DeltaProduct at d62241a, under the MIT licence
(see LICENSE and NOTICE). This commit is third-party code and is separated out
so that every later commit is this project's own contribution.

Upstream: Siems, Carstensen, Zela, Hutter, Pontil, Grazzi. DeltaProduct:
Improving State-Tracking in Linear RNNs via Householder Products. NeurIPS 2025.
arXiv:2502.10297."
echo "  $(git rev-parse --short HEAD)  $(git ls-files | wc -l) files"

# ------------------------------------------------------------ 2. copy our tree
say "Copying the working tree"
cd "$OLD"
EXCLUDES=(
  --exclude='.git/'
  --exclude='.venv/'
  --exclude='venv/'
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
  --exclude='state_tracking/data/'
  --exclude='state_tracking/checkpoints/'
  --exclude='wandb/'
  --exclude='*.deb'
  # multi-MB raw training logs: the numbers they support live in results/*.json
  --exclude='results_S5_*.log'
  # working scratch that does not belong in the record
  --exclude='Claude outputs/'
  --exclude='PROGRESS_REPORT-1.md'
  --exclude='nohup.out'
)
if command -v rsync >/dev/null; then
  rsync -a "${EXCLUDES[@]}" ./ "$NEW/"
else
  tar -cf - --exclude=.git --exclude=.venv --exclude=__pycache__ \
      --exclude='state_tracking/data' --exclude='results_S5_*.log' \
      --exclude='Claude outputs' --exclude='PROGRESS_REPORT-1.md' . \
    | tar -xf - -C "$NEW"
fi
cd "$NEW"

# upstream's own README is kept, ours takes its place
if [ -f README_upstream.md ] && [ -f README.md ]; then
  echo "  README.md is ours; README_upstream.md retains upstream's"
fi

say "Extending .gitignore"
cat >> .gitignore <<'EOF'

# ---- raw training logs ----
# Multi-MB console logs from long runs. Every figure quoted in the project
# documents is taken from the JSON in results/, which is committed.
results_S5_*.log
nohup.out
EOF

# ------------------------------------------------------- 3. thematic commits
commit_group () {          # commit_group "<message>" <paths...>
  local msg="$1"; shift
  local staged=0
  for p in "$@"; do [ -e "$p" ] && { git add -A -- "$p"; staged=1; }; done
  [ "$staged" -eq 1 ] || return 0
  git diff --cached --quiet && return 0
  git -c commit.gpgsign=false commit -q -m "$msg"
  printf '  %s  %s\n' "$(git rev-parse --short HEAD)" "${msg%%$'\n'*}"
}

say "Committing the project's work"

commit_group "Add project licence notice and upstream attribution" NOTICE

commit_group "Adapt the Householder layer for a per-token, learned factor count

The halting gate multiplies into beta after the sigmoid rather than selecting
between fixed values, which the determinant-parity bound requires: with beta
held at its reflection value only factor counts congruent to the target's
transposition length modulo two are reachable.

The halting nonlinearity is evaluated in float32 regardless of the layer's
dtype. In bfloat16 the largest value below 1.0 is 0.99609375, so the sigmoid
rounds to exactly 1.0 above logit 6.2364 and its derivative becomes exactly
zero, which freezes the gate permanently." \
  src/gating.py src/oracle_order.py flash-linear-attention/

commit_group "Add algebraic verification of the expressivity bounds

Explicit float64 constructions, with no optimiser involved: the rank bound, the
determinant-parity bound over all 120 elements of S5, and the order-requirement
model built from the icosahedral rotation group." \
  work/reference_deltaproduct.py work/exp_beta_reachability.py \
  work/exp_minimal_order.py work/census.py

commit_group "Add the training and measurement harness

Each experiment carries a self-test that runs against synthetic data whose
answer is known in advance, and each test is verified to fail against an
incorrect implementation." \
  work/ state_tracking/

commit_group "Add measured results" results/

commit_group "Add project documentation and progress report" \
  PLAN.md README.md README_upstream.md AUDIT.md GLOSSARY.md HANDOFF.md \
  PROGRESS_REPORT.md report/ progress_review.pptx .gitignore

commit_group "Add remaining project files" .

# ------------------------------------------------------------------- 4. report
say "Result"
git log --pretty='  %h  %an  %s' | tail -n 20
echo
echo "  contributors:"
git shortlog -sne | sed 's/^/    /'
echo
echo "  tracked files: $(git ls-files | wc -l)"
echo "  repo size:     $(git count-objects -vH | awk '/size-pack/{print $2, $3}')"
echo "  largest tracked files:"
git ls-files | xargs -I{} du -k "{}" 2>/dev/null | sort -rn | head -5 | sed 's/^/    /'
echo
cat <<EOF

NEXT — nothing has been pushed, and the old repository is untouched.

  1. Look through it:
       cd "$NEW"
       git log --stat | head -60
       git diff \$(git rev-list --max-parents=0 HEAD)..HEAD --stat | tail -30

  2. Create an EMPTY repo named ${NEW_NAME} on github.com (no README, no
     licence, and do NOT use the Fork button), then:
       git remote add origin https://github.com/Shukla070/${NEW_NAME}.git
       git push -u origin main

  3. Confirm the contributors list on GitHub shows only you, then decide what
     to do with the old repository.
EOF
