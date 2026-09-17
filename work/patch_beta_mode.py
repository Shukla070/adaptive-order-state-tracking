"""
Patch the DeltaProduct layer so beta's parametrisation can be swapped.

WHY
---
Upstream computes the Householder strength as

    beta = sigmoid(W h) * 2          =>  beta in the OPEN interval (0, 2)

with `bias=False` on W. Two consequences, both bad, and together they are the
leading suspect for every learnability failure we have:

  1. beta can never ATTAIN 2. A permutation needs beta exactly 2 (a reflection).
     The layer can approach one and never be one.

  2. d beta/dx = 2*sigma*(1-sigma), which goes to ZERO as beta approaches 2.
     The gradient vanishes exactly at the value the model has to reach:

         beta = 1.00  ->  gradient 0.500     (a useless projection)
         beta = 1.90  ->  gradient 0.095
         beta = 1.99  ->  gradient 0.010
         beta = 1.999 ->  gradient 0.001

     And with bias=False every factor STARTS at beta = 1.0 -- the maximally
     destructive setting, 500x more gradient than where it needs to go.

So the optimiser is initialised at the worst point and has to climb through a
vanishing-gradient region to reach the solution. That is a plausible mechanism
for: fixed2 at p=1.0 (0.1207), A5_c5 n_h=2 (0.1259), A5_c5_dt n_h=2 (0.0443),
A5_c5_dt n_h=4 (0.0429), S5_c5_t1 n_h=4 (0.4721 at 80k steps), S5_c5_t n_h=4
(0.0706), sweep p=0.5 fixed4 (0.7070). Seven cells, provably enough capacity,
none of which trained.

WHAT THIS ADDS
--------------
  m.beta_mode = "sigmoid"   2*sigmoid(x + b)          upstream, the default
               "clamp"      2*clamp(0.25x + b + 0.5, 0, 1)
                            attains 0 and 2 exactly; constant gradient inside
               "ste"        forward sigmoid, backward linear
                            keeps the sigmoid's shape but not its dead gradient
  m.beta_bias               learnable, shape (num_householder, num_heads),
                            initialised to 0. Set it to 4.0 and beta starts at
                            1.96 -- a near-reflection -- instead of 1.0.

Defaults reproduce upstream exactly (mode "sigmoid", bias 0), so this patch is
inert until an experiment opts in. Nothing is plumbed through the config: the
experiment sets the attributes on the built model, which keeps the model builder
untouched.

USAGE
-----
    python work/patch_beta_mode.py            # apply (idempotent)
    python work/patch_beta_mode.py --revert   # restore the backup
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("flash-linear-attention/fla/layers/gated_deltaproduct.py")
BACKUP = Path("flash-linear-attention/fla/layers/gated_deltaproduct.py.orig.beta")
MARK = "PATCH(beta-parametrisation)"

ANCHOR_INIT = """        self.b_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.num_heads, bias=False)
                for _ in range(num_householder)
            ]
        )"""

ADD_INIT = """

        # --- PATCH(beta-parametrisation) ---
        # Upstream: beta = 2*sigmoid(Wh) with bias=False, so beta starts at 1.0
        # (a projection) and the gradient vanishes as it approaches 2.0 (the
        # reflection a permutation actually needs). These two attributes let an
        # experiment swap that out. Defaults are byte-equivalent to upstream.
        self.beta_mode = "sigmoid"
        self.beta_bias = nn.Parameter(
            torch.zeros(num_householder, self.num_heads)
        )
        # --- END PATCH ---"""

ANCHOR_FWD = """            beta = self.b_projs[i](
                hidden_states
            ).sigmoid()  # bs, sequence_length, num_heads"""

ADD_FWD = """            # --- PATCH(beta-parametrisation) ---
            raw = self.b_projs[i](hidden_states) + self.beta_bias[i]
            if self.beta_mode == "clamp":
                # attains both endpoints exactly; gradient 0.25 inside, 0 out
                beta = (raw * 0.25 + 0.5).clamp(0.0, 1.0)
            elif self.beta_mode == "ste":
                # forward value is the sigmoid, backward gradient is the linear
                _lin = (raw * 0.25 + 0.5).clamp(0.0, 1.0)
                _sig = raw.sigmoid()
                beta = _lin + (_sig - _lin).detach()
            else:
                beta = raw.sigmoid()
            # --- END PATCH ---"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--target", default=str(TARGET))
    args = ap.parse_args()

    tgt = Path(args.target)
    bak = Path(str(tgt) + ".orig.beta")

    if not tgt.exists():
        print(f"ERROR: {tgt} not found. Run from the repository root.")
        return 1

    if args.revert:
        if not bak.exists():
            print(f"ERROR: no backup at {bak}")
            return 1
        shutil.copy2(bak, tgt)
        print(f"reverted {tgt} from {bak}")
        return 0

    src = tgt.read_text()
    if MARK in src:
        print("already patched; nothing to do")
        return 0

    for name, anchor in (("__init__ b_projs block", ANCHOR_INIT),
                         ("forward beta computation", ANCHOR_FWD)):
        n = src.count(anchor)
        if n != 1:
            print(f"ERROR: anchor '{name}' found {n} times, expected exactly 1.")
            print("The file has diverged from what this patch expects. Not")
            print("touching it -- inspect the file and update the anchors.")
            return 1

    if not bak.exists():
        shutil.copy2(tgt, bak)
        print(f"backup written: {bak}")

    src = src.replace(ANCHOR_INIT, ANCHOR_INIT + ADD_INIT, 1)
    src = src.replace(ANCHOR_FWD, ADD_FWD, 1)
    tgt.write_text(src)

    check = tgt.read_text()
    ok = (check.count(MARK) == 2
          and "self.beta_mode" in check
          and "self.beta_bias" in check
          and 'elif self.beta_mode == "ste"' in check)
    print(f"patched {tgt}")
    print(f"verification: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("verification failed -- reverting")
        shutil.copy2(bak, tgt)
        return 1

    print()
    print("  defaults unchanged (beta_mode='sigmoid', beta_bias=0), so every")
    print("  existing experiment behaves exactly as before.")
    print("  Revert any time with:  python work/patch_beta_mode.py --revert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
