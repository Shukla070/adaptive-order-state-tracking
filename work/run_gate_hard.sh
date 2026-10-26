#!/usr/bin/env bash
# Is the accuracy lost by closed gates an artefact of evaluating with hard
# gates after training with soft ones -- and does training with hard gates
# remove it?  (PLAN Part 3e, "The logit wall", point 3; OPEN_WORK item 2.)
# About 11 h on an RTX 4050.
#
# Two arms on the same 20 seeds as gate_wall_seeds.json (no wall, gamma 0.1,
# pre-repair start), interleaved per seed:
#   hard from never  -- the no-wall arm again, now evaluated with soft AND hard
#                       gates; also the first seed-by-seed replication at scale
#   hard from 12000  -- straight-through hard gates for the last 8 000 steps,
#                       so training sees the gates evaluation uses
#
# Launch ONCE, from the repo root, after any other run has finished.
# Full run (20 seeds, ~11 h):
#   nohup bash work/run_gate_hard.sh > results/gate_hard_driver.log 2>&1 &
# Targeted run (the 6 seeds that closed but lost accuracy on 2 Oct, ~3.5 h):
#   SEEDS=12,13,15,25,29,30 OUT=results/gate_hard_6seeds \
#     nohup bash work/run_gate_hard.sh > results/gate_hard_6seeds_driver.log 2>&1 &
# Watch:
#   watch -n 120 "tail -n 6 results/gate_hard_seeds.log"
set -u
cd "$(dirname "$0")/.."
if pgrep -f "exp_gate_train.py" > /dev/null; then
    echo "exp_gate_train.py is already running -- refusing to start a second copy"
    exit 1
fi
SEEDS="${SEEDS:-$(seq -s, 11 30)}"
OUT="${OUT:-results/gate_hard_seeds}"
echo "$(date -u) start  seeds $SEEDS  out $OUT.json"
python -u work/exp_gate_train.py --gate_init hf_default --arms learned \
    --gammas 0.1 --hard_froms 0,12000 --log_dense_until 4000 \
    --seeds "$SEEDS" --fingerprint results/gate_wall_seeds.json \
    --out "$OUT.json" > "$OUT.log" 2>&1
echo "$(date -u) exit $?"
