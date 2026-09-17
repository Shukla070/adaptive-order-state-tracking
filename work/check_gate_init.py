"""
Does the halting gate's initialisation survive model construction?

WHAT src/gating.py INTENDS
-------------------------
    nn.init.zeros_(self.mlp[-1].weight)          # gate is input-INDEPENDENT
    nn.init.constant_(self.mlp[-1].bias, B)      # gate starts OPEN

Both properties are load-bearing. Zero weight means every token receives the
same order at step 0, so the gate cannot allocate before it has learned
anything. A positive bias means the model begins as fixed-order
DeltaProduct(K), which is the configuration known to converge.

WHY THEY DO NOT SURVIVE ON THEIR OWN
------------------------------------
`GatedDeltaProductForCausalLM` is a HuggingFace model, and `PreTrainedModel`
runs `post_init()` -> `_init_weights` across every submodule AFTER __init__ has
finished. That hook matches `nn.Linear`:

    nn.init.normal_(module.weight, std=initializer_range)
    nn.init.zeros_(module.bias)

The gate's output layer is an nn.Linear, so both properties are overwritten.
The unit tests in src/gating.py construct a HaltingGate directly and therefore
cannot see this: the defect is in the config -> HF model -> post_init path, not
in the gate itself.

THE OBSERVABLE
--------------
With output bias b, lambda = sigmoid(b) and the gates are its cumulative
products, so n_t = sum_{i=1..K} sigmoid(b)^i:

    b = 4.0  ->  3.823   (intended: gate open, model starts at order K)
    b = 0.0  ->  0.938   (what post_init leaves: gate nearly closed)

A training log whose first step reports mean n_t near 0.94 rather than 3.82 is
reporting a gate that did not initialise open.

WHAT THIS FILE CHECKS
---------------------
    A  raw HF construction        EXPECTED TO FAIL -- this is the defect, and
                                  this section is what makes the rest a test
                                  written to fail rather than to pass.
    B  reinit_halting_gates()     must PASS -- the repair, on CPU.
    C  build() end to end         must PASS -- proves the repair is actually
                                  wired into the path training uses. Needs CUDA,
                                  because build() moves the model to the device;
                                  skipped with a warning if CUDA is absent.

Section C is the one that matters. A and B can both pass while training still
runs on an unrepaired model, if build() forgets the call.

RUN
---
    python work/check_gate_init.py
    python work/check_gate_init.py --bias 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

N_SPECIAL = 7
N_GROUP = 120


def expected_n_t(bias: float, K: int, dtype: torch.dtype = torch.float32) -> float:
    """Effective order of an input-independent gate at output bias `bias`.

    Computed in `dtype`, because the answer depends on it. The gate runs in
    bfloat16 once the model reaches the device, and bf16 has an 8-bit mantissa,
    so sigmoid(4.0) rounds to 0.980469 rather than 0.982014 and the cumulative
    product compounds that over K terms:

        float32   ->  3.8233
        bfloat16  ->  3.8125   (exactly 61/16)

    Predicting the wrong one produces a spurious failure of about 0.011, which
    is a hundred times smaller than the 2.886 gap between an open gate and a
    clobbered one, but large enough to trip a tight tolerance.
    """
    lam = torch.sigmoid(torch.tensor(float(bias), dtype=dtype))
    return torch.cumprod(lam.repeat(K), dim=-1).sum().item()


def gate_modules(model):
    out = []
    for m in model.modules():
        if getattr(m, "adaptive_order", False):
            g = getattr(m, "halting_gate", None)
            if g is not None:
                out.append(g)
    return out


def make_config(args):
    from fla.models import GatedDeltaProductConfig
    cfg = GatedDeltaProductConfig(
        hidden_size=args.hidden,
        num_hidden_layers=args.n_layers,
        num_heads=args.n_heads,
        head_dim=args.head_dim,
        expand_v=1,
        vocab_size=N_GROUP + N_SPECIAL,
        allow_neg_eigval=True,
        num_householder=args.max_order,
        fuse_cross_entropy=False,
        max_position_embeddings=2048,
    )
    cfg.adaptive_order = True
    cfg.gate_hard = False
    cfg.gate_init_open_bias = args.bias
    return cfg


def inspect(gates, bias: float, K: int, hidden: int, label: str) -> bool:
    """Report the gate's state. Returns True if it matches the intended init."""
    if not gates:
        print(f"    no halting gate found                       [FAIL]")
        return False

    ok = True
    for i, g in enumerate(gates):
        p = g.mlp[-1]
        # predict in the gate's own dtype: bf16 and fp32 give different answers
        want = expected_n_t(bias, K, p.weight.dtype)
        b_mean = p.bias.detach().float().mean().item()
        w_abs = p.weight.detach().float().abs().sum().item()

        bias_ok = abs(b_mean - bias) < 1e-4
        weight_ok = w_abs < 1e-6

        # the behavioural consequence, evaluated on the gate's own device/dtype
        x = torch.randn(2, 16, hidden, device=p.weight.device, dtype=p.weight.dtype)
        with torch.no_grad():
            _, n_t = g(x)
        n_t = n_t.float()
        got, spread = n_t.mean().item(), n_t.std().item()
        n_ok = abs(got - want) < 5e-3
        flat_ok = spread < 1e-4

        ok = ok and bias_ok and weight_ok and n_ok and flat_ok
        tag = f" (layer {i})" if len(gates) > 1 else ""
        print(f"    output bias mean{tag:<10} {b_mean:+.6f}  want {bias:+.6f}"
              f"   [{'PASS' if bias_ok else 'FAIL'}]")
        print(f"    output weight sum|W|{tag:<6} {w_abs:.6f}  want 0.000000"
              f"   [{'PASS' if weight_ok else 'FAIL'}]")
        dt = str(p.weight.dtype).replace("torch.", "")
        print(f"    mean n_t at init{tag:<10} {got:.4f}    want {want:.4f}"
              f"  ({dt})  [{'PASS' if n_ok else 'FAIL'}]")
        print(f"    n_t spread over tokens{tag:<4} {spread:.2e}  want 0"
              f"          [{'PASS' if flat_ok else 'FAIL'}]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=4.0)
    ap.add_argument("--max_order", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=1)
    args = ap.parse_args()

    from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM
    from exp_matched_compute import build, reinit_halting_gates

    print("=" * 74)
    print("GATE INITIALISATION — does it survive model construction?")
    print("=" * 74)
    print(f"  requested init bias   {args.bias}   intended n_t at init = "
          f"{expected_n_t(args.bias, args.max_order):.4f} (float32) / "
          f"{expected_n_t(args.bias, args.max_order, torch.bfloat16):.4f} (bfloat16)"
          f"  of {args.max_order}")

    # -- A. the defect ------------------------------------------------------
    print("\n  A. raw HuggingFace construction, no repair applied")
    print("     (this SHOULD fail; if it passes, post_init no longer clobbers")
    print("      the gate and the repair in build() is now redundant)")
    torch.manual_seed(0)
    raw = GatedDeltaProductForCausalLM(make_config(args))
    raw_ok = inspect(gate_modules(raw), args.bias, args.max_order, args.hidden, "raw")
    print(f"    -> {'UNEXPECTEDLY CLEAN' if raw_ok else 'clobbered, as expected'}")

    # -- B. the repair, on CPU ---------------------------------------------
    print("\n  B. after reinit_halting_gates()")
    reinit_halting_gates(raw, args.bias)
    b_ok = inspect(gate_modules(raw), args.bias, args.max_order, args.hidden, "fixed")

    # -- C. the path training actually uses --------------------------------
    print("\n  C. via build(), the constructor every experiment calls")
    if not torch.cuda.is_available():
        c_ok = None
        print("    SKIPPED — build() moves the model to CUDA and no GPU is")
        print("    visible. Section C is the one that proves the repair reaches")
        print("    training; re-run this on the GPU machine before trusting it.")
    else:
        m = build(args.max_order, True, args.n_heads, args.head_dim, args.hidden,
                  args.n_layers, N_GROUP + N_SPECIAL, seed=0,
                  gate_init_bias=args.bias)
        c_ok = inspect(gate_modules(m), args.bias, args.max_order, args.hidden, "build")
        del m
        torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    passed = b_ok and (c_ok is not False)
    if passed and c_ok:
        print("PASS — the gate initialises open and input-independent in the path")
        print("       training uses. A run's first log line should read")
        print(f"       'mean n_t {expected_n_t(args.bias, args.max_order, torch.bfloat16):.3f}'"
              f" (bf16; {expected_n_t(args.bias, args.max_order):.3f} in float32).")
    elif passed:
        print("INCOMPLETE — the repair works, but section C did not run, so it is")
        print("       unproven that build() applies it. Re-run where CUDA is visible.")
    else:
        print("FAIL — the gate's initialisation is still not being applied.")
        print("       Check that build() calls reinit_halting_gates() BEFORE")
        print("       moving the model to the device.")
    print("=" * 74)
    return 0 if (passed and c_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
