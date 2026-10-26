#!/usr/bin/env bash
# Boxes follow-up (PLAN, "Learned gate on Boxes -- follow-up, pre-registered 23 October",
# and its amendment of the same day). Scores are late averages (mean E over the last 4
# quick evaluations).
#
# 1. layers21: fixed orders [2, 1] (2 factors in the first layer, 1 in the second), no
#    gate, 469,120 parameters, seeds 0-7; plus 3 fixed-2 reference runs (seeds 5-7) so
#    the fixed-2 reference has 8 runs. ~8 x 25 min + 3 x 33 min.
# 2. probe2:   the learned gate at gamma 0.001 and 0.003, seeds 0-2. ~6 x 36 min.
# About 8.5 h on an RTX 4050. Results go into results/boxes_gate.json beside the
# earlier stages; every model is saved under checkpoints/boxes_gate/.
#
# Launch ONCE, from the repo root, after the two 300-step smoke tests:
#   nohup bash work/run_boxes_followup.sh > results/boxes_followup_driver.log 2>&1 &
# Watch:
#   watch -n 120 "grep -hE '^>>>|^<<<|quick-eval' results/boxes_followup_*.log | tail -n 8"
set -u
cd "$(dirname "$0")/.."
if pgrep -f "exp_gate_train.py|exp_boxes_order.py|exp_boxes_gate.py" > /dev/null; then
    echo "a training run is already active -- refusing to start a second one"
    exit 1
fi
echo "$(date -u) start layers21"
python -u work/exp_boxes_gate.py --stage layers21 > results/boxes_followup_layers21.log 2>&1
echo "$(date -u) layers21 exit $?"
echo "$(date -u) start probe2"
python -u work/exp_boxes_gate.py --stage probe2 > results/boxes_followup_probe2.log 2>&1
echo "$(date -u) probe2 exit $?"
python work/exp_boxes_gate.py --summary
