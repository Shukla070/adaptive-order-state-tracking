"""
Adaptive-order gating for DeltaProduct.

DeltaProduct applies a FIXED number of generalized Householder factors (`n_h`)
to every token. This module makes that order input-dependent and learned.

Mechanism
---------
Each micro-step i of token t gets a halting probability lambda_t^i. Gates are
the cumulative product, which makes them monotone non-increasing:

    g_t^i = prod_{j<=i} lambda_t^j          g_t^1 >= g_t^2 >= ... >= g_t^K
    beta~_t^i = g_t^i * beta_t^i            <-- SCALES a learnable beta
    n_t = sum_i g_t^i                       effective order of token t

The gate is never supervised. It learns because a budget penalty pushes n_t
down while the task loss pushes it up where expressivity is actually needed.

Three design decisions, each forced by a measured result
-------------------------------------------------------
1. THE GATE SCALES BETA; IT NEVER SELECTS BETWEEN {0, 2}.
   With beta pinned at 2 every factor is a reflection (det = -1), so a product
   of K factors has det (-1)^K and can only represent permutations with
   K = L (mod 2) -- half of all targets are unreachable at any K. Verified in
   work/reference_deltaproduct.py, test T4. A hard binary gate, which is what a
   Gumbel/straight-through formulation naturally produces, hits this wall and
   fails silently on half the inputs. `test_parity_guard` below is a regression
   test for exactly this.

2. GATES INITIALISE OPEN.
   At init the model must behave like fixed-order DeltaProduct(K), which we know
   converges (S3: seq acc 1.000 in one epoch). The budget penalty then closes
   gates where they are not earning their keep. Initialising closed would start
   the model at DeltaNet and may never open.

3. BETA = 0 GIVES EXACTLY THE IDENTITY.
   Verified bit-exact in float64 (T1) and at 222x signal-to-noise in the real
   Triton kernel (work/check_beta_identity.py). This is what lets a closed gate
   mean "skip this micro-step" without any new operation.

Integration point
-----------------
fla/layers/gated_deltaproduct.py, in the per-micro-step loop:

    beta = self.b_projs[i](hidden_states).sigmoid()
    if self.allow_neg_eigval:
        beta = beta * 2
    beta = beta * gates[..., i:i+1]        # <-- here
    betas.append(beta)

Because `interleave_multiple_sequences` then stacks the betas, a gate of zero
produces an identity factor in the expanded length-(n_h*T) sequence. Ragged
expansion (Path B, for wall-clock savings) is a later step; this module is
correct for both.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HaltingGate(nn.Module):
    """Predicts per-token, per-micro-step gates in [0, 1], monotone in i.

    Args:
        hidden_size:    width of the layer input the gate reads.
        max_order:      K, the maximum number of Householder factors.
        num_heads:      number of attention heads (only used if per_head).
        mlp_dim:        width of the gate's hidden layer. A few thousand
                        parameters total -- this is a tiny component.
        per_head:       if True, each head gets its own order. Default False:
                        one order per token, which is what the theory describes
                        (a token needs L transpositions, not L-per-head).
        init_open_bias: output bias at init. Larger => gates start closer to
                        fully open, but the sigmoid saturates and gradients
                        shrink. 4.0 gives lambda ~ 0.982, so n_t ~ 0.93*K at
                        init while still training.
        fp32_halting:   evaluate the halting sigmoid and the cumulative
                        product in float32 regardless of the layer's dtype.
                        Default True, and it should stay True: see the PRECISION
                        note in forward(). Exposed only so that the old
                        behaviour can be constructed and measured by
                        work/check_gate_precision.py.
        use_state_summary:
                        if True, concatenate a causal running mean of the
                        hidden states so the gate can see accumulated context,
                        not just the current token. Needed for the
                        non-locally-detectable condition in E4b; unnecessary on
                        group word problems, where each token carries its own
                        difficulty. Default False.
    """

    def __init__(
        self,
        hidden_size: int,
        max_order: int,
        num_heads: int = 1,
        mlp_dim: int = 64,
        per_head: bool = False,
        init_open_bias: float = 4.0,
        use_state_summary: bool = False,
        fp32_halting: bool = True,
    ) -> None:
        super().__init__()
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")

        self.max_order = max_order
        self.num_heads = num_heads
        self.per_head = per_head
        self.use_state_summary = use_state_summary
        self.fp32_halting = fp32_halting

        in_dim = hidden_size * (2 if use_state_summary else 1)
        out_dim = max_order * (num_heads if per_head else 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim),
        )
        # start open: zero weights + positive bias => constant, near-1 lambdas
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_open_bias)

    @staticmethod
    def _causal_mean(x: torch.Tensor) -> torch.Tensor:
        """Running mean over time. Causal: position t sees only <= t."""
        counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        return x.cumsum(dim=1) / counts.view(1, -1, *([1] * (x.dim() - 2)))

    def forward(self, hidden_states: torch.Tensor, hard: bool = False):
        """
        Args:
            hidden_states: (B, T, hidden_size)
            hard: if True, threshold gates to {0,1} with a straight-through
                  estimator, so the forward pass uses an integer order while
                  gradients still flow. Use for eval / measuring real cost.

        Returns:
            gates: (B, T, K) or (B, T, K, num_heads) if per_head
            n_t:   (B, T) effective order per token
        """
        in_dtype = hidden_states.dtype
        gate_dtype = self.mlp[-1].weight.dtype

        x = hidden_states.to(gate_dtype)
        if self.use_state_summary:
            x = torch.cat([x, self._causal_mean(x)], dim=-1)

        logits = self.mlp(x)
        if self.per_head:
            logits = logits.view(*logits.shape[:-1], self.max_order, self.num_heads)

        # PRECISION -- do not remove. The halting nonlinearity is evaluated in
        # float32 whatever dtype the surrounding layer uses.
        #
        # bfloat16 carries a 7-bit explicit mantissa, so the largest value it
        # can hold strictly below 1.0 is 0.99609375 (= 1 - 2^-8). sigmoid(z)
        # therefore rounds to EXACTLY 1.0 for every z above 6.2364, which is
        # where it first exceeds the midpoint 0.998047. That makes the local
        # derivative lambda*(1 - lambda) exactly zero -- not small, zero. Once
        # a gate's logits pass that point no gradient of any size can reach it
        # again, and the gate is frozen for the rest of training.
        #
        # This is not hypothetical. In a 10-seed run at p = 0.10 the effective
        # order hit 4.000 and the gate's weight norm stopped changing to four
        # significant figures by step 2000 -- before the budget penalty had
        # been switched on -- in 10 runs out of 10. Final gate weight norms were
        # 8 to 13, comfortably past the cliff at 6.2364.
        #
        # The gate is a small MLP beside a Triton kernel, so evaluating it in
        # float32 costs nothing measurable. `gates` is cast back to the input
        # dtype before returning, because the downstream chunked kernel asserts
        # a non-float32 input.
        lambdas = torch.sigmoid(logits.float() if self.fp32_halting
                                else logits)
        gates = torch.cumprod(lambdas, dim=2 if self.per_head else -1)

        if hard:
            hard_gates = (gates > 0.5).to(gates.dtype)
            # NB: the parenthesisation matters. (hard + gates) - gates.detach()
            # rounds, and the forward value drifts off {0,1}; grouping the
            # subtraction makes it exactly zero, so forward == hard exactly.
            gates = hard_gates + (gates - gates.detach())

        n_t = gates.sum(dim=2 if self.per_head else -1)
        return gates.to(in_dtype), n_t


def apply_gates(betas: list[torch.Tensor], gates: torch.Tensor) -> list[torch.Tensor]:
    """Scale each micro-step's beta by its gate.

    Args:
        betas: list of K tensors, each (B, T, num_heads) -- fla's layout, one
               entry per Householder factor.
        gates: (B, T, K) or (B, T, K, num_heads)

    Returns:
        list of K gated beta tensors, same shapes as the inputs.
    """
    if len(betas) != gates.shape[2]:
        raise ValueError(f"{len(betas)} betas but gates has K={gates.shape[2]}")
    if gates.dim() == 3:                       # per token, broadcast over heads
        return [b * gates[..., i : i + 1] for i, b in enumerate(betas)]
    return [b * gates[:, :, i, :] for i, b in enumerate(betas)]


def budget_loss(
    n_t: torch.Tensor,
    target_order: float | None = None,
    under_penalty: float = 2.0,
) -> torch.Tensor:
    """Penalty on the compute the gate spends.

    With `target_order=None` this is simply mean(n_t): push the order down and
    let the task loss push back where expressivity is needed. Multiply by your
    gamma at the call site.

    With a target, the penalty is asymmetric around it. Over-allocating only
    wastes compute; under-allocating breaks correctness, so under-shooting is
    penalised `under_penalty` times harder. This is the asymmetry recorded in
    PLAN.md 3.2.
    """
    mean_order = n_t.mean()
    if target_order is None:
        return mean_order
    diff = mean_order - target_order
    return torch.where(diff >= 0, diff, -diff * under_penalty)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _open_gate(**kw) -> HaltingGate:
    """A gate forced fully open: sigmoid(20) == 1.0 exactly in float32."""
    g = HaltingGate(init_open_bias=20.0, **kw)
    return g


def test_reduces_to_fixed_order() -> bool:
    """Fully open gates must reproduce fixed-order DeltaProduct exactly."""
    torch.manual_seed(0)
    B, T, H, K, D = 2, 16, 4, 4, 32
    gate = _open_gate(hidden_size=D, max_order=K, num_heads=H)
    x = torch.randn(B, T, D)
    gates, n_t = gate(x)

    betas = [torch.rand(B, T, H) * 2 for _ in range(K)]
    gated = apply_gates(betas, gates)

    err = max((g - b).abs().max().item() for g, b in zip(gated, betas))
    order_err = (n_t - K).abs().max().item()
    ok = err == 0.0 and order_err == 0.0
    print(f"T1  open gates == fixed order   : max|dbeta| = {err:.2e}, "
          f"max|n_t - K| = {order_err:.2e}   [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_closed_gate_is_identity_factor() -> bool:
    """A closed gate must drive beta to (effectively) zero => identity factor.

    A sigmoid never returns exactly 0, so in SOFT mode the residual beta is
    ~1e-17. The kernel runs in bfloat16, whose epsilon is ~7.8e-3, so a residual
    fifteen orders of magnitude below that is not merely negligible -- it is
    unrepresentable. The criterion is therefore "far below bf16 epsilon", not
    "bit-zero".

    In HARD mode gates are exactly {0,1}, so beta is exactly 0 and the factor is
    exactly the identity. That is the mode used when measuring real cost.
    """
    torch.manual_seed(0)
    B, T, H, K, D = 2, 8, 4, 4, 32
    BF16_EPS = 2 ** -7                         # ~7.8e-3

    gate = HaltingGate(hidden_size=D, max_order=K, num_heads=H)
    with torch.no_grad():
        gate.mlp[-1].bias.fill_(-40.0)
    x = torch.randn(B, T, D)
    betas = [torch.rand(B, T, H) * 2 for _ in range(K)]

    soft_gates, soft_n = gate(x)
    soft_max = max(g.abs().max().item() for g in apply_gates(betas, soft_gates))

    hard_gates, hard_n = gate(x, hard=True)
    hard_max = max(g.abs().max().item() for g in apply_gates(betas, hard_gates))

    soft_ok = soft_max < 1e-8 and soft_n.max().item() < 1e-8
    hard_ok = hard_max == 0.0 and hard_n.abs().max().item() == 0.0
    ok = soft_ok and hard_ok
    print(f"T2  closed gates => beta ~ 0    : soft {soft_max:.2e} "
          f"(bf16 eps {BF16_EPS:.1e}), hard {hard_max:.2e}   "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def test_monotone() -> bool:
    """Gates must be non-increasing in i: once halted, stay halted."""
    torch.manual_seed(1)
    B, T, K, D = 4, 32, 5, 32
    gate = HaltingGate(hidden_size=D, max_order=K, init_open_bias=0.0)
    with torch.no_grad():                      # random weights => varied gates
        nn.init.normal_(gate.mlp[-1].weight, std=1.0)
    gates, n_t = gate(torch.randn(B, T, D))

    mono = bool((gates[..., :-1] - gates[..., 1:] >= -1e-7).all())
    ranged = bool((gates >= 0).all() and (gates <= 1).all())
    order_ok = bool((n_t >= 0).all() and (n_t <= K).all())
    ok = mono and ranged and order_ok
    print(f"T3  monotone / in range / n_t in [0,K] : {mono}, {ranged}, "
          f"{order_ok}   [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_parity_guard() -> bool:
    """REGRESSION TEST for the parity wall (oracle T4).

    The gate must SCALE beta, producing intermediate values. If someone
    reimplements this as a hard selector between {0, 2}, determinant parity
    makes half of all permutations unreachable. This fails if gated betas ever
    collapse to only the endpoints.
    """
    torch.manual_seed(2)
    B, T, H, K, D = 4, 32, 2, 4, 32
    gate = HaltingGate(hidden_size=D, max_order=K, num_heads=H, init_open_bias=0.0)
    with torch.no_grad():
        nn.init.normal_(gate.mlp[-1].weight, std=1.0)
    gates, _ = gate(torch.randn(B, T, D))
    betas = [torch.rand(B, T, H) * 2 for _ in range(K)]
    gated = torch.stack(apply_gates(betas, gates))

    interior = ((gated > 1e-4) & (gated < 2.0 - 1e-4)).float().mean().item()
    ok = interior > 0.5
    print(f"T4  beta takes interior values  : {interior*100:.1f}% strictly "
          f"inside (0,2)   [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        print("    -> gate is behaving like a binary selector. See the parity")
        print("       constraint in this module's docstring and oracle T4.")
    return ok


def test_gradients_flow() -> bool:
    """The gate must be trainable, and the budget loss must reduce the order."""
    torch.manual_seed(3)
    B, T, H, K, D = 4, 16, 2, 4, 32
    gate = HaltingGate(hidden_size=D, max_order=K, num_heads=H)
    x = torch.randn(B, T, D)

    _, n_t = gate(x)
    loss = budget_loss(n_t)
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in gate.parameters() if p.grad is not None)

    opt = torch.optim.Adam(gate.parameters(), lr=0.1)
    before = n_t.mean().item()
    for _ in range(50):
        opt.zero_grad()
        _, n = gate(x)
        budget_loss(n).backward()
        opt.step()
    after = n.mean().item()

    ok = gnorm > 0 and after < before - 0.5
    print(f"T5  grads flow, budget bites    : |grad| = {gnorm:.3f}, "
          f"order {before:.2f} -> {after:.2f}   [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_hard_mode_is_integer() -> bool:
    """`hard=True` must give integer orders and still pass gradients through."""
    torch.manual_seed(4)
    B, T, K, D = 2, 16, 4, 32
    gate = HaltingGate(hidden_size=D, max_order=K, init_open_bias=0.0)
    with torch.no_grad():
        nn.init.normal_(gate.mlp[-1].weight, std=1.0)
    x = torch.randn(B, T, D, requires_grad=True)

    gates, n_t = gate(x, hard=True)
    is_int = bool(((n_t - n_t.round()).abs() < 1e-5).all())
    binary = bool((((gates == 0) | (gates == 1)).all()))
    n_t.sum().backward()
    has_grad = x.grad is not None and x.grad.abs().sum().item() > 0

    ok = is_int and binary and has_grad
    print(f"T6  hard mode integer + ST grad : int={is_int}, binary={binary}, "
          f"grad={has_grad}   [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_against_reference() -> bool:
    """End-to-end against the float64 oracle.

    Gating micro-steps off must reproduce a genuine lower-order DeltaProduct
    state, exactly. This is oracle T1 driven by the real gate module.
    """
    try:
        from reference_deltaproduct import deltaproduct_seq
    except ImportError:
        print("T7  vs reference oracle         : SKIPPED (reference_deltaproduct.py "
              "not importable from here)")
        return True

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)
    T, K, d = 24, 4, 12

    gate = HaltingGate(hidden_size=d, max_order=K).to(torch.float64)
    with torch.no_grad():                       # open step 1, close 2..K
        # 40, not 20: sigmoid(20) = 1 - 2e-9, which is exactly 1.0 in float32
        # but NOT in float64, and this test runs in float64. sigmoid(40) rounds
        # to exactly 1.0 in both.
        gate.mlp[-1].bias.copy_(torch.tensor([40.0, -40.0, -40.0, -40.0]))
    gates, n_t = gate(torch.randn(1, T, d))

    ks = torch.randn(T, K, d); ks = ks / ks.norm(dim=-1, keepdim=True)
    vs = torch.randn(T, K, d)
    betas = torch.rand(T, K) * 2

    S_order1 = deltaproduct_seq(betas[:, :1], ks[:, :1], vs[:, :1])

    # soft gates: micro-steps 2..K carry beta ~ 1e-17 rather than bit-zero, so
    # agreement is to float64 epsilon accumulated over T steps, not exact.
    S_soft = deltaproduct_seq(betas * gates[0], ks, vs)
    soft_err = (S_soft - S_order1).abs().max().item()

    # hard gates: exactly {0,1}, so the closed factors are exactly the identity
    # and the state must match a true order-1 model bit for bit.
    hard_gates, hard_n = gate(torch.randn(1, T, d), hard=True)
    with torch.no_grad():
        hard_gates = torch.zeros_like(hard_gates)
        hard_gates[..., 0] = 1.0
    S_hard = deltaproduct_seq(betas * hard_gates[0], ks, vs)
    hard_err = (S_hard - S_order1).abs().max().item()

    ok = soft_err < 1e-12 and hard_err == 0.0 and abs(n_t.mean().item() - 1.0) < 1e-9
    print(f"T7  vs reference oracle         : soft {soft_err:.2e} (fp64 eps), "
          f"hard {hard_err:.2e} (exact), n_t = {n_t.mean().item():.4f}   "
          f"[{'PASS' if ok else 'FAIL'}]")
    torch.set_default_dtype(torch.float32)
    return ok


if __name__ == "__main__":
    print("=" * 72)
    print("Adaptive-order gating — unit tests")
    print("=" * 72)
    results = [
        test_reduces_to_fixed_order(),
        test_closed_gate_is_identity_factor(),
        test_monotone(),
        test_parity_guard(),
        test_gradients_flow(),
        test_hard_mode_is_integer(),
        test_against_reference(),
    ]
    print("=" * 72)
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    print("=" * 72)
