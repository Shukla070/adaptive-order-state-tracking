#!/usr/bin/env bash
# Last per-token attempt before the stopping rule (PLAN Part 7): the budget
# penalty from step 1 instead of after a 4 000-step penalty-free warmup.
# About 11 h on an RTX 4050. Seeds 11-30, gamma 0.1, 20 000 steps, penalty
# ramped linearly from step 1 over 4 000 steps.
#
#   Arm A: pre-repair start (gate almost shut), soft gates -- the 2 October
#          baseline with the warmup removed, nothing else changed.
#   Arm B: open start (repaired gate), hard straight-through gates from step 1
#          -- training uses exactly the gates evaluation uses, so there is no
#          soft/hard mismatch and no late switch.
#
# Both arms keep the first factor always on (--min_order 1): a smoke test on
# 6 Oct without it closed every factor within 250 steps, the layer then wrote
# nothing, and the task could not be learned.
#
# Arms run one after the other (A, then B).
# Launch ONCE, from the repo root:
#   nohup bash work/run_gate_early_penalty.sh > results/gate_early_driver.log 2>&1 &
# Watch:
#   watch -n 120 "tail -n 6 results/gate_early_A.log results/gate_early_B.log"
set -u
cd "$(dirname "$0")/.."
if pgrep -f "exp_gate_train.py" > /dev/null; then
    echo "exp_gate_train.py is already running -- refusing to start a second copy"
    exit 1
fi
SEEDS="${SEEDS:-$(seq -s, 11 30)}"
OUT="${OUT:-results/gate_early}"
COMMON="--arms learned --gammas 0.1 --budget_warmup 0 --budget_ramp 4000 --min_order 1 --log_dense_until 4000 --seeds $SEEDS"

echo "$(date -u) arm A start  seeds $SEEDS"
python -u work/exp_gate_train.py $COMMON --gate_init hf_default \
    --out "${OUT}_A.json" > "${OUT}_A.log" 2>&1
echo "$(date -u) arm A exit $?"

echo "$(date -u) arm B start"
python -u work/exp_gate_train.py $COMMON --gate_init repaired --hard_froms 1 \
    --out "${OUT}_B.json" > "${OUT}_B.log" 2>&1
echo "$(date -u) arm B exit $?"
