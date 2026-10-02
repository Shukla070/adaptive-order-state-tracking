#!/usr/bin/env bash
# Does a logit wall stop the halting gate saturating open, and does the gate
# then close reliably?  (PLAN Part 3e, "Rebuilding the earliest configuration";
# OPEN_WORK item 2.)  About 12-13 h on an RTX 4050.
#
# Launch ONCE, from the repo root:
#   nohup bash work/run_gate_wall.sh > results/gate_wall_driver.log 2>&1 &
# Watch:
#   watch -n 120 "tail -n 6 results/gate_wall_*.log"
set -u
cd "$(dirname "$0")/.."
if pgrep -f "exp_gate_train.py" > /dev/null; then
    echo "exp_gate_train.py is already running -- refusing to start a second copy"
    exit 1
fi
COMMON="--gate_init hf_default --arms learned --log_dense_until 4000"

# 1. Anchors (~50 min): seeds whose outcome under this start is known --
#    seed 1 separated 4/4, seeds 2 and 3 0/8. Wall on only.
echo "$(date -u) anchors start"
python -u work/exp_gate_train.py $COMMON --gammas 0.1 --logit_walls 5 \
    --seeds 1,2,3 --fingerprint results/gate_train_p10.json \
    --out results/gate_wall_anchor.json > results/gate_wall_anchor.log 2>&1
echo "$(date -u) anchors exit $?"

# 2. Main comparison (~11 h): 20 new seeds, wall off vs on, interleaved per seed
#    so drift over the night affects both arms alike.
echo "$(date -u) main start"
python -u work/exp_gate_train.py $COMMON --gammas 0.1 --logit_walls 0,5 \
    --seeds "$(seq -s, 11 30)" \
    --out results/gate_wall_seeds.json > results/gate_wall_seeds.log 2>&1
echo "$(date -u) main exit $?"

# 3. Control (~50 min): wall on, no budget. How much does the wall close the
#    gate by itself? (self-test T17d shows it can, with no task loss opposing.)
echo "$(date -u) control start"
python -u work/exp_gate_train.py $COMMON --gammas 0 --logit_walls 5 \
    --seeds 11,12,13 \
    --out results/gate_wall_nobudget.json > results/gate_wall_nobudget.log 2>&1
echo "$(date -u) control exit $?"
