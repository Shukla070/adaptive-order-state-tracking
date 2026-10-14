#!/usr/bin/env bash
# Restart rule on the best recipe (PLAN, "Restart rule -- pre-registered 14 October").
#
# The best recipe (open start, hard straight-through gates and the budget
# penalty from step 1, first factor always on, gamma 0.1) reached the exact
# allocation in 13 of 20 runs; every failure was visible at step 500 as a mean
# order below 1.20. Here a run below 1.20 at step 500 is stopped and started
# again from a new seed (base + 1000 per attempt), at most 5 attempts; the 5th
# runs to the end unchecked. New base seeds 31-50, so the threshold, chosen on
# seeds 11-30, is tested on runs it has not seen.
#
# About 6 h on an RTX 4050 (20 x ~16.5 min, plus ~0.4 min per restart).
# Launch ONCE, from the repo root:
#   nohup bash work/run_gate_restart.sh > results/gate_restart_driver.log 2>&1 &
# Watch:
#   watch -n 120 "grep -E 'RESTART|seed [0-9]+$|eval ' results/gate_restart.log | tail -n 8"
set -u
cd "$(dirname "$0")/.."
if pgrep -f "exp_gate_train.py|exp_boxes_order.py" > /dev/null; then
    echo "a training run is already active -- refusing to start a second one"
    exit 1
fi
SEEDS="${SEEDS:-$(seq -s, 31 50)}"
OUT="${OUT:-results/gate_restart}"

echo "$(date -u) start  seeds $SEEDS"
python -u work/exp_gate_train.py --arms learned --gammas 0.1 \
    --budget_warmup 0 --budget_ramp 4000 --min_order 1 \
    --gate_init repaired --hard_froms 1 --log_dense_until 4000 \
    --restart_below 1.20 --restart_at 500 --max_attempts 5 \
    --save_dir checkpoints/gate_restart \
    --seeds "$SEEDS" --out "${OUT}.json" > "${OUT}.log" 2>&1
echo "$(date -u) exit $?"
