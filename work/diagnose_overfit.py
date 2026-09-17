"""
Can the model fit S5 AT ALL?

Three S5 runs (1 layer n_h=3, 1 layer n_h=4, 2 layers n_h=4) all landed at
val loss 4.59-4.72 against 4.787 for a model that has learned nothing. We only
ever looked at VALIDATION loss, so we cannot tell these apart:

  (a) under-training -- 7,740 steps with the LR annealed to zero across exactly
      those steps, the same failure as the first S3 run;
  (b) something structural -- d_state too small for a 120-element group, or a
      setup problem specific to S5.

The overfit test separates them. Train on a tiny subset with a CONSTANT
learning rate and no scheduler, and watch TRAINING loss. Any model that can
represent the task should drive training loss toward zero on 512 sequences.

  memorises  -> the setup is sound; S5 needs real training time.
  does not   -> more epochs will not help. Capacity or a bug.

S3 runs first as a positive control: it is known to converge, so if S3 also
fails to memorise, the harness itself is wrong and neither result means
anything.

    python work/diagnose_overfit.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import polars as pl

from fla.models import GatedDeltaProductConfig
from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM

DEV = "cuda"
DTYPE = torch.bfloat16
DATA = Path("state_tracking/data")

N_SEQ = 512          # subset to memorise
BATCH = 128
STEPS = 1500
LR = 1e-3            # CONSTANT. no scheduler -- that is the point.
LOG_EVERY = 100


def load_subset(csv_path: Path, n_seq: int):
    """Read the first n_seq rows. Columns are space-separated token ids."""
    df = pl.read_csv(csv_path, n_rows=n_seq)
    cols = df.columns
    str_cols = [c for c in cols if df[c].dtype == pl.String]
    if len(str_cols) < 2:
        raise SystemExit(f"expected two string columns in {csv_path}, got {cols}")
    icol, tcol = str_cols[0], str_cols[1]

    inp = [[int(x) for x in s.split()] for s in df[icol]]
    tgt = [[int(x) for x in s.split()] for s in df[tcol]]
    L = min(min(len(r) for r in inp), min(len(r) for r in tgt))
    inp = torch.tensor([r[:L] for r in inp], dtype=torch.long)
    tgt = torch.tensor([r[:L] for r in tgt], dtype=torch.long)
    vocab = int(max(inp.max().item(), tgt.max().item())) + 1
    print(f"    columns {cols} -> using '{icol}' / '{tcol}'")
    print(f"    {inp.shape[0]} sequences, length {L}, vocab {vocab}")
    return inp.to(DEV), tgt.to(DEV), vocab


def build(vocab, n_h, n_layers=1):
    conf = GatedDeltaProductConfig(
        hidden_size=128,               # d_state in main.py
        num_hidden_layers=n_layers,
        num_heads=8,
        head_dim=32,
        expand_v=1,
        vocab_size=vocab,
        allow_neg_eigval=True,
        num_householder=n_h,
        fuse_cross_entropy=False,
        max_position_embeddings=2048,
    )
    return GatedDeltaProductForCausalLM(conf).to(DEV, DTYPE)


def overfit(name, csv, n_h, n_layers=1):
    import math
    print(f"\n{'='*70}\n{name}  (n_h={n_h}, layers={n_layers})\n{'='*70}")
    if not csv.exists():
        print(f"    SKIP - {csv} not found")
        return None

    inp, tgt, vocab = load_subset(csv, N_SEQ)
    uniform = math.log(vocab)
    model = build(vocab, n_h, n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)   # constant LR

    print(f"    params {sum(p.numel() for p in model.parameters())/1e3:.0f}k, "
          f"uniform-guess loss = ln({vocab}) = {uniform:.3f}")
    print(f"\n    {'step':>6} {'train loss':>11} {'vs uniform':>11}  {'note':<20}")
    print("    " + "-" * 54)

    t0 = time.perf_counter()
    best = float("inf")
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, inp.shape[0], (BATCH,), device=DEV)
        x, y = inp[idx], tgt[idx]
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        best = min(best, loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            l = loss.item()
            note = ("MEMORISED" if l < 0.05 else
                    "learning"  if l < uniform - 0.5 else
                    "flat")
            print(f"    {step:>6} {l:>11.4f} {uniform - l:>11.3f}  {note:<20}")

    dt = time.perf_counter() - t0
    verdict = ("CAN memorise -> setup sound, needs training time"
               if best < 0.1 else
               "PARTIAL -> learning but slow"
               if best < uniform - 1.0 else
               "CANNOT fit even 512 sequences -> not a time problem")
    print(f"\n    best loss {best:.4f} after {STEPS} steps in {dt:.0f}s")
    print(f"    VERDICT: {verdict}")
    return best


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    print("Overfit diagnostic — constant LR, no scheduler, training loss.\n"
          f"subset={N_SEQ} sequences, batch={BATCH}, steps={STEPS}, lr={LR}")

    s3 = overfit("S3 — POSITIVE CONTROL (known to converge)",
                 DATA / "S3=128.csv", n_h=2)
    s5 = overfit("S5 — THE QUESTION", DATA / "S5=128.csv", n_h=4)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  S3 best loss: {s3:.4f}" if s3 is not None else "  S3: skipped")
    print(f"  S5 best loss: {s5:.4f}" if s5 is not None else "  S5: skipped")
    if s3 is not None and s5 is not None:
        if s3 < 0.1 and s5 < 0.1:
            print("\n  Both memorise. The harness is fine and S5 is a TRAINING TIME\n"
                  "  problem: run it long, with the scheduler off or stretched.")
        elif s3 < 0.1 <= s5:
            print("\n  S3 memorises, S5 does not. Not a time problem -- S5 hits a\n"
                  "  capacity or setup limit. Next: raise d_state, or check whether\n"
                  "  one layer can represent S5 at all.")
        else:
            print("\n  S3 did not memorise either -- this diagnostic is wrong, not the\n"
                  "  model. Fix the harness before drawing any conclusion.")
