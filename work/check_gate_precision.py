"""
Can the budget penalty still move a gate whose logits have grown large?

THE CLAIM UNDER TEST
--------------------
The halting gate stopped responding to the budget penalty in 10 of 10 training
runs. Effective order sat at exactly 4.000 and the gate's weight norm stopped
changing to four significant figures by step 2000, before the penalty was even
switched on. Two explanations have already been ruled out by experiment:

  * the gate never becoming input-dependent  -- refuted: final weight norms were
    8 to 13 in every run, including the runs that did close
  * the gate's initialisation                -- refuted: repairing it made the
    outcome worse, 0 of 10 against 2 of 10

The remaining explanation is arithmetic. bfloat16 carries a 7-bit explicit
mantissa, so the largest value it can represent strictly below 1.0 is
0.99609375. Any logit above 6.2364 therefore makes sigmoid(z) round to EXACTLY
1.0, and the local derivative lambda*(1 - lambda) is then exactly zero rather
than merely small. Once that happens no gradient reaches the gate again,
whatever its size. Measured: at z = 6.2 the derivative is 3.89e-03; at z = 6.24
it is 0.

A second, independent failure sits behind it: even with a non-zero gradient, a
bfloat16 weight of magnitude ~1 resolves to about 0.0078, so an update of order
1e-5 disappears when it is added to the weight.

WHAT THIS FILE DOES
-------------------
Reproduces both failures on CPU in seconds, with no GPU and no training run, and
shows that a float32 gate does not suffer either. T1 and T3 are written to FAIL
against the float32 gate -- they assert the bfloat16 gate is broken -- so a
silent regression in either direction is visible.

This is a cheap test standing in front of an expensive experiment. If T2 and T4
do not pass, the diagnosis is wrong and no GPU time should be spent on it.

RUN
---
    python work/check_gate_precision.py           # CPU, a few seconds
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from gating import HaltingGate, budget_loss                         # noqa: E402

B, T, K, D = 4, 32, 4, 128
SAT_BIAS = 10.0      # past the bf16 cliff at 6.2364; the runs reached 8-13
OK_BIAS = 1.0        # inside the responsive region, as a contrast


def make_gate(dtype: torch.dtype, bias: float, fp32_halting: bool) -> HaltingGate:
    """A gate whose logits start constant at `bias`, in the given dtype.

    `fp32_halting=False` reproduces the behaviour that was running during the
    training runs: the sigmoid and the cumulative product evaluated in the
    layer's own dtype. `True` is the repair. Both must be exercised here, or the
    comparison is the repair measured against itself.
    """
    torch.manual_seed(0)
    g = HaltingGate(hidden_size=D, max_order=K, init_open_bias=bias,
                    fp32_halting=fp32_halting)
    with torch.no_grad():
        torch.nn.init.zeros_(g.mlp[-1].weight)
        torch.nn.init.constant_(g.mlp[-1].bias, bias)
    return g.to(dtype)


def grad_norm_after_budget(dtype: torch.dtype, bias: float,
                           fp32_halting: bool) -> tuple[float, float]:
    """Push one budget-loss gradient through the gate. Returns (|grad|, n_t)."""
    g = make_gate(dtype, bias, fp32_halting)
    x = torch.randn(B, T, D, dtype=dtype)
    _, n_t = g(x)
    budget_loss(n_t).backward()
    gn = sum(p.grad.abs().sum().item() for p in g.parameters() if p.grad is not None)
    return gn, n_t.float().mean().item()


def descend(dtype: torch.dtype, bias: float, fp32_halting: bool,
            steps: int = 300, lr: float = 1e-2):
    """Optimise the budget penalty alone. Returns (n_t before, n_t after)."""
    g = make_gate(dtype, bias, fp32_halting)
    x = torch.randn(B, T, D, dtype=dtype)
    opt = torch.optim.AdamW(g.parameters(), lr=lr)
    with torch.no_grad():
        before = g(x)[1].float().mean().item()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, n_t = g(x)
        budget_loss(n_t).backward()
        opt.step()
    with torch.no_grad():
        after = g(x)[1].float().mean().item()
    return before, after


def main() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    print("=" * 76)
    print("GATE PRECISION — can the budget penalty move a saturated gate?")
    print("=" * 76)

    # found by construction, not assumed
    import math
    lim = max(v for v in (torch.tensor(1.0 - 2.0 ** -k, dtype=torch.bfloat16)
                          .to(torch.float64).item() for k in range(6, 14)) if v < 1.0)
    cliff = math.log(((lim + 1) / 2) / (1 - (lim + 1) / 2))
    print(f"  largest bfloat16 value below 1.0: {lim:.8f}")
    print(f"  sigmoid rounds to exactly 1.0 above logit {cliff:.4f}, "
          f"where the derivative becomes exactly 0")
    print(f"  saturating bias used below: {SAT_BIAS}   responsive bias: {OK_BIAS}\n")

    # ---- the defect ------------------------------------------------------
    print("  saturated logits (bias = %.1f)" % SAT_BIAS)
    g_bf, n_bf = grad_norm_after_budget(torch.bfloat16, SAT_BIAS, False)
    g_fp, n_fp = grad_norm_after_budget(torch.float32, SAT_BIAS, True)
    print(f"    bfloat16 gate: |grad| = {g_bf:.3e}   n_t = {n_bf:.4f}")
    print(f"    float32  gate: |grad| = {g_fp:.3e}   n_t = {n_fp:.4f}")

    check("T1  bfloat16 gate receives EXACTLY zero gradient when saturated",
          g_bf == 0.0, f"|grad| = {g_bf:.3e}")
    check("T2  float32 gate still receives a usable gradient",
          g_fp > 0.0, f"|grad| = {g_fp:.3e}")

    # ---- the behavioural consequence -------------------------------------
    print("\n  300 steps of budget-only descent from the same saturated start")
    b_bf, a_bf = descend(torch.bfloat16, SAT_BIAS, False)
    b_fp, a_fp = descend(torch.float32, SAT_BIAS, True)
    print(f"    bfloat16 gate: n_t {b_bf:.4f} -> {a_bf:.4f}   (of {K})")
    print(f"    float32  gate: n_t {b_fp:.4f} -> {a_fp:.4f}   (of {K})")

    check("T3  bfloat16 gate does not move at all",
          abs(a_bf - b_bf) < 1e-6, f"moved {abs(a_bf - b_bf):.3e}")
    check("T4  float32 gate closes materially",
          a_fp < b_fp - 1.0, f"{b_fp:.3f} -> {a_fp:.3f}")

    # ---- the contrast: this is specific to saturation ---------------------
    print(f"\n  responsive logits (bias = {OK_BIAS}) — both dtypes should work")
    gb2, _ = grad_norm_after_budget(torch.bfloat16, OK_BIAS, False)
    gf2, _ = grad_norm_after_budget(torch.float32, OK_BIAS, True)
    bb, ab = descend(torch.bfloat16, OK_BIAS, False)
    bf_, af = descend(torch.float32, OK_BIAS, True)
    print(f"    bfloat16 gate: |grad| = {gb2:.3e}   n_t {bb:.4f} -> {ab:.4f}")
    print(f"    float32  gate: |grad| = {gf2:.3e}   n_t {bf_:.4f} -> {af:.4f}")
    check("T5  away from saturation bfloat16 is fine — the defect is the cliff, "
          "not bfloat16 as such", gb2 > 0.0 and ab < bb - 0.5,
          f"|grad| = {gb2:.3e}, {bb:.3f} -> {ab:.3f}")

    # ---- the contract the kernel depends on -------------------------------
    print("\n  interface")
    gf = make_gate(torch.float32, OK_BIAS, True)
    x_bf = torch.randn(B, T, D, dtype=torch.bfloat16)
    gates, n_t = gf(x_bf)
    check("T6  a float32 gate given bfloat16 input returns bfloat16 gates "
          "(the chunked kernel asserts non-float32)",
          gates.dtype == torch.bfloat16, f"got {gates.dtype}")
    mono = bool((gates[..., :-1].float() - gates[..., 1:].float() >= -1e-6).all())
    ranged = bool((gates.float() >= 0).all() and (gates.float() <= 1).all())
    check("T7  monotonicity and range survive the cast", mono and ranged,
          f"monotone={mono}, in range={ranged}")

    print("\n" + "=" * 76)
    if ok:
        print("DIAGNOSIS CONFIRMED — bfloat16 saturation freezes the gate, and a")
        print("float32 gate does not freeze. Proceeding to a GPU run is justified.")
    else:
        print("DIAGNOSIS NOT SUPPORTED — do not spend GPU time on this change.")
        print("Read which check failed: T1/T3 failing means bfloat16 is not the")
        print("cause; T2/T4 failing means float32 does not fix it either.")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
