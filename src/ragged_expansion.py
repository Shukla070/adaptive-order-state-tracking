"""
Ragged expansion for the adaptive-order DeltaProduct layer ("Path B").

WHAT IT DOES
------------
The layer as built (Path A) expands every token into K micro-steps, one per
Householder factor, and runs the delta-rule kernel over a sequence of length
K*T. A factor the halting gate closes has beta = 0, which is an exact identity
on the state, so Path A still pays for it.

Ragged expansion drops the closed micro-steps before the kernel runs. Token t
contributes max(n_t, 1) micro-steps instead of K, the sequences of a batch are
packed end to end, and `cu_seqlens` marks where each one starts so the kernel
resets its state between them. The output is read at each token's last kept
micro-step instead of at micro-step K.

WHY IT IS EXACT (with hard gates)
---------------------------------
Path A, for one token with gates g_1..g_K in {0,1}, monotone (a prefix of ones):

  step 1      : decay exp(g_decay), then factor 1   (beta_1 * g_1)
  step i > 1  : no decay,           then factor i   (beta_i * g_i)
  step K      : query applied here, output read here

A closed factor (g_i = 0) has beta = 0, so the step is S -> S: identity. All
closed factors come after all open ones, so steps n_t+1..K are identities and
the state read at step K equals the state after step n_t. Placing the query at
step n_t (instead of K) and deleting steps n_t+1..K therefore gives the same
output. Step 1 is always kept (even if n_t = 0) because it carries the decay;
with n_t = 0 its beta is already 0, so it applies only the decay, exactly as
Path A does.

Requirements, all checked at runtime: gates binary, monotone non-increasing
along the factor axis, one gate vector per token (not per head), no recurrent
cache (inference from the start of the sequence), no padding mask.

WHAT IT DOES NOT DO
-------------------
- Training. Closed factors are not computed, so a straight-through estimator
  gets no gradient to reopen them. Train with Path A; infer with Path B.
- It does not skip the per-factor key/value/beta projections; those are
  computed for every factor before packing. Only the recurrence is shortened.
- Packing (version 2, 11 October): index_copy per factor into preallocated
  buffers instead of stacking all K factors and selecting with boolean masks.
  The first version made four full copies of K-factor tensors and one host
  synchronisation per mask; it was correct but slow (see PLAN, ragged timing).

Run `python src/ragged_expansion.py` for the CPU self-test (float64, no GPU needed). It
checks the packing and indexing against an independent Path A computation
with a reference kernel, and checks that four deliberately broken variants
fail. The real kernel is checked by `work/check_ragged.py` on the GPU.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# the plan: which micro-steps to keep, where sequences start, where to read
# ---------------------------------------------------------------------------

def validate_gates(gates: torch.Tensor) -> torch.Tensor:
    """Check gates are (B, T, K), binary and monotone; return n_t as int64 (B, T).

    Raises instead of silently producing a wrong answer: soft gates, per-head
    gates or non-prefix gates would make the ragged output differ from Path A.
    """
    if gates.dim() != 3:
        raise ValueError(
            f"ragged expansion needs one gate vector per token, shape (B, T, K); "
            f"got {tuple(gates.shape)}. Per-head gates are not supported.")
    gf = gates.detach().float()
    if not torch.all((gf == 0) | (gf == 1)):
        raise ValueError("ragged expansion needs hard gates (exactly 0 or 1); "
                         "set the layer's gates to hard before using Path B.")
    if gf.shape[-1] > 1 and torch.any(gf[..., 1:] > gf[..., :-1]):
        raise ValueError("gates must be monotone non-increasing along the factor "
                         "axis (a prefix of ones); a closed factor followed by an "
                         "open one cannot be dropped.")
    return gf.sum(-1).round().long()


def ragged_plan(n_t: torch.Tensor, K: int):
    """From per-token orders, build everything the packing needs.

    Returns
      keep        (B, T, K) bool   micro-steps kept (prefix of length max(n_t,1))
      last        (B, T)    long   index of the last kept micro-step, max(n_t,1)-1
      cu_seqlens  (B+1,)    long   packed sequence boundaries
      read_pos    (B, T)    long   packed position at which to read each token
    """
    B, T = n_t.shape
    n_eff = n_t.clamp(min=1, max=K)
    ar = torch.arange(K, device=n_t.device)
    keep = ar.view(1, 1, K) < n_eff.unsqueeze(-1)
    last = n_eff - 1
    counts = keep.view(B, -1).sum(-1)
    cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=n_t.device)
    cu_seqlens[1:] = torch.cumsum(counts, 0)
    flat_keep = keep.reshape(-1).long()
    packed_index = torch.cumsum(flat_keep, 0) - 1          # valid where kept
    flat_last = (torch.arange(B * T, device=n_t.device) * K
                 + last.reshape(-1))
    read_pos = packed_index[flat_last].view(B, T)
    return keep, last, cu_seqlens, read_pos


# ---------------------------------------------------------------------------
# packing and the ragged core
# ---------------------------------------------------------------------------

def _stack_factors(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """List of K tensors (B, T, ...) -> (B, T, K, ...)."""
    return torch.stack(list(xs), dim=2)


def ragged_core(q: torch.Tensor,
                ks: Sequence[torch.Tensor],
                vs: Sequence[torch.Tensor],
                betas: Sequence[torch.Tensor],
                g: Optional[torch.Tensor],
                gates: torch.Tensor,
                kernel: Callable,
                _mutant: str = "") -> torch.Tensor:
    """The recurrence of one layer, with closed micro-steps removed.

    q            (B, T, H, Dk)       one query per token
    ks, vs       K x (B, T, H, D*)   per-factor keys and values
    betas        K x (B, T, H)       per-factor beta, ALREADY multiplied by the gate
    g            (B, T, H) or None   per-token log-decay (applied at micro-step 1)
    gates        (B, T, K)           hard, monotone
    kernel       callable(q, k, v, g, beta, cu_seqlens) -> o, all with batch 1
    returns      (B, T, H, Dv)

    Packing is done by index_copy into preallocated buffers, one factor at a
    time, with no boolean masks: token j's micro-step i goes to packed position
    start[j] + i when i < max(n_j, 1), and otherwise to a scratch row of its
    own (index N + j, discarded). One host synchronisation (to size the buffers); no copy of all K
    factors stacked together; the query is written only at each token's last
    kept step.

    `_mutant` exists only for the self-tests that must fail.
    """
    B, T, H, Dk = q.shape
    K = len(ks)
    BT = B * T
    dev = q.device
    n_t = validate_gates(gates)
    n_eff = n_t.clamp(min=1, max=K).reshape(-1)                  # (BT,)
    start = torch.cumsum(n_eff, 0) - n_eff                      # exclusive cumsum
    read = start + n_eff - 1                                     # last kept step
    per_seq = n_eff.view(B, T).sum(-1)
    cu = torch.zeros(B + 1, dtype=torch.long, device=dev)
    cu[1:] = torch.cumsum(per_seq, 0)
    N = int(cu[-1])                                              # the one sync
    if _mutant == "no_cu_seqlens":
        cu = torch.tensor([0, N], dtype=torch.long, device=dev)

    # closed micro-steps are written to a private scratch row per token
    # (N + j), so no two writes collide and the result is deterministic
    scratch = N + torch.arange(BT, device=dev)

    def pack(factors, width_shape):
        buf = factors[0].new_empty((N + BT,) + width_shape)
        for i, f in enumerate(factors):
            dest = torch.where(n_eff > i, start + i, scratch)
            buf.index_copy_(0, dest, f.reshape((BT,) + width_shape))
        return buf[:N].unsqueeze(0)

    pk = pack(ks, (H, ks[0].shape[-1]))
    pv = pack(vs, (H, vs[0].shape[-1]))
    pb = pack(betas, (H,))

    q_dest = read if _mutant != "query_at_first" else start
    pq = q.new_zeros((N, H, Dk))
    pq.index_copy_(0, q_dest, q.reshape(BT, H, Dk))
    pq = pq.unsqueeze(0)

    pg = None
    if g is not None:
        g_dest = start if _mutant != "decay_at_last" else read
        pg = g.new_zeros((N, H))
        pg.index_copy_(0, g_dest, g.reshape(BT, H))
        pg = pg.unsqueeze(0)

    o = kernel(pq, pk, pv, pg, pb, cu)                           # (1, N, H, Dv)
    rp = read if _mutant != "read_at_first" else start
    return o[0].index_select(0, rp).view(B, T, H, -1)


# ---------------------------------------------------------------------------
# Path A, written independently, for comparison
# ---------------------------------------------------------------------------

def path_a_core(q, ks, vs, betas, g, kernel_batched):
    """What the layer computes today: interleave all K factors, query at
    micro-step K, decay at micro-step 1, read every K-th output.

    kernel_batched(q, k, v, g, beta) -> o, batch dimension B, no cu_seqlens.
    """
    B, T, H, _ = q.shape
    K = len(ks)

    def inter(xs):  # K x (B, T, ...) -> (B, T*K, ...), token-major
        return torch.stack(list(xs), dim=2).reshape(B, T * K, *xs[0].shape[2:])

    qi = inter([torch.zeros_like(q)] * (K - 1) + [q])
    ki, vi, bi = inter(ks), inter(vs), inter(betas)
    gi = None if g is None else inter([g] + [torch.zeros_like(g)] * (K - 1))
    o = kernel_batched(qi, ki, vi, gi, bi)
    return o[:, K - 1::K]


# ---------------------------------------------------------------------------
# stacked ("fused") variants: all K factors in one tensor, (B, T, K, ...)
# ---------------------------------------------------------------------------
#
# Profiling (results/ragged_profile.json, 12 Oct) showed two costs that are
# artefacts of computing each factor separately: upstream's interleaving
# (torch.stack of K tensors, ~9 ms of 61 at batch 32 x 2048 for fixed n_h = 4)
# and the per-factor packing copies of version 2 (~6.8 ms). If the K factor
# projections are computed by ONE matmul with the K weight matrices
# concatenated, the result (B, T, K*C) viewed as (B, T, K, H, D) is already
# token-major, factor-minor -- i.e. already interleaved. Path A then needs no
# copy at all, and ragged expansion needs only one index_select of the kept
# rows. The same fused projection is offered to the fixed-order baseline, so a
# speedup measured against it is not just "we skipped upstream's copies".

def path_a_stacked(q, k_all, v_all, b_all, g, kernel_batched):
    """Path A from stacked factors. q (B,T,H,Dk); k_all (B,T,K,H,Dk);
    v_all (B,T,K,H,Dv); b_all (B,T,K,H) gated; g (B,T,H) or None."""
    B, T, K, H, Dk = k_all.shape
    qi = q.new_zeros(B, T, K, H, Dk)
    qi[:, :, K - 1] = q
    gi = None
    if g is not None:
        gi = g.new_zeros(B, T, K, H)
        gi[:, :, 0] = g
        gi = gi.reshape(B, T * K, H)
    o = kernel_batched(qi.reshape(B, T * K, H, Dk), k_all.reshape(B, T * K, H, Dk),
                       v_all.reshape(B, T * K, H, -1), gi, b_all.reshape(B, T * K, H))
    return o[:, K - 1::K]


def ragged_stacked(q, k_all, v_all, b_all, g, gates, kernel, _mutant: str = ""):
    """Ragged expansion from stacked factors: gather only the kept rows.

    Kept micro-steps, flattened token-major and factor-minor, are already in
    packed order, so packed position = rank among kept rows, and one
    index_select per tensor does all the packing.
    """
    B, T, K, H, Dk = k_all.shape
    BT = B * T
    dev = q.device
    n_t = validate_gates(gates)
    n_eff = n_t.clamp(min=1, max=K).reshape(-1)
    keep = torch.arange(K, device=dev).view(1, K) < n_eff.view(BT, 1)
    kept = keep.reshape(-1).nonzero().squeeze(1)           # the one host sync
    N = kept.numel()
    start = torch.cumsum(n_eff, 0) - n_eff
    read = start + n_eff - 1
    cu = torch.zeros(B + 1, dtype=torch.long, device=dev)
    cu[1:] = torch.cumsum(n_eff.view(B, T).sum(-1), 0)
    if _mutant == "no_cu_seqlens":
        cu = torch.tensor([0, N], dtype=torch.long, device=dev)

    pk = k_all.reshape(BT * K, H, Dk).index_select(0, kept).unsqueeze(0)
    pv = v_all.reshape(BT * K, H, -1).index_select(0, kept).unsqueeze(0)
    pb = b_all.reshape(BT * K, H).index_select(0, kept).unsqueeze(0)
    pq = q.new_zeros((N, H, Dk))
    pq.index_copy_(0, read if _mutant != "query_at_first" else start, q.reshape(BT, H, Dk))
    pg = None
    if g is not None:
        pg = g.new_zeros((N, H))
        pg.index_copy_(0, start if _mutant != "decay_at_last" else read, g.reshape(BT, H))
        pg = pg.unsqueeze(0)
    o = kernel(pq.unsqueeze(0), pk, pv, pg, pb, cu)
    rp = read if _mutant != "read_at_first" else start
    return o[0].index_select(0, rp).view(B, T, H, -1)


def fused_factor_inputs(layer, h, gates=None):
    """All K factors' keys, values and betas of a GatedDeltaProduct layer from
    one matmul (and one depthwise convolution) each, with the weights of the K
    factors concatenated. Mathematically identical to the per-factor loop in
    the layer; numerically equal up to bf16 GEMM tiling.

    Returns q (B,T,H,Dk), k_all (B,T,K,H,Dk), v_all (B,T,K,H,Dv),
    b_all (B,T,K,H) with the gate applied, g (B,T,H) or None.
    """
    import torch.nn.functional as F
    K, H = layer.num_householder, layer.num_heads
    B, T, _ = h.shape
    cat = lambda mods: torch.cat([m.weight for m in mods], 0)
    k = F.linear(h, cat(layer.k_projs))
    v = F.linear(h, cat(layer.v_projs))
    if layer.use_short_conv:
        k = _fused_short_conv(layer.k_conv1ds, k)
        v = _fused_short_conv(layer.v_conv1ds, v)
        q, _ = layer.q_conv1ds[0](x=layer.q_proj(h), mask=None, cache=None,
                                  output_final_state=False)
    else:
        k, v = F.silu(k), F.silu(v)
        q = F.silu(layer.q_proj(h))
    raw = F.linear(h, cat(layer.b_projs)).view(B, T, K, H) + layer.beta_bias
    mode = getattr(layer, "beta_mode", "sigmoid")
    if mode == "clamp":
        beta = (raw * 0.25 + 0.5).clamp(0.0, 1.0)
    elif mode == "ste":
        lin = (raw * 0.25 + 0.5).clamp(0.0, 1.0)
        beta = lin + (raw.sigmoid() - lin).detach()
    else:
        beta = raw.sigmoid()
    if layer.allow_neg_eigval:
        beta = beta * 2
    if gates is not None:
        if gates.dim() != 3:
            raise ValueError("fused path supports one gate vector per token only")
        beta = beta * gates.to(beta.dtype).unsqueeze(-1)
    g = None
    if layer.use_forget_gate:
        g = -layer.A_log.float().exp() * F.softplus(layer.a_proj(h).float() + layer.dt_bias)
    return (q.reshape(B, T, H, -1), k.reshape(B, T, K, H, -1),
            v.reshape(B, T, K, H, -1), beta, g)


def _fused_short_conv(mods, x):
    """K depthwise causal convolutions as one, on concatenated channels.
    Depthwise convolution acts on each channel independently, so this is exact."""
    import torch.nn.functional as F
    from einops import rearrange
    m0 = mods[0]
    weight = torch.cat([m.weight for m in mods], 0)            # (K*C, 1, W)
    bias = None if m0.bias is None else torch.cat([m.bias for m in mods], 0)
    W = weight.shape[-1]
    if getattr(m0, "use_fast_conv1d", False):
        from fla.modules import convolution as _fconv
        y = _fconv.causal_conv1d_fn(x=rearrange(x, "b t d -> b d t"),
                                    weight=rearrange(weight, "d 1 w -> d w"),
                                    bias=bias, activation=m0.activation, seq_idx=None)
        return rearrange(y, "b d t -> b t d")
    y = F.conv1d(rearrange(x, "b t d -> b d t"), weight, bias, padding=W - 1,
                 groups=weight.shape[0])[..., :x.shape[1]]
    if m0.activation is not None:
        y = F.silu(y)
    return rearrange(y, "b d t -> b t d")


# ---------------------------------------------------------------------------
# reference kernel (float64, sequential) — for the CPU self-test only
# ---------------------------------------------------------------------------

def reference_gated_delta(q, k, v, g, beta, cu_seqlens=None, l2norm=True):
    """Sequential gated delta rule, same convention as fla's chunk kernels:

        S <- S * exp(g)
        S <- S + beta * k (v - S^T k)^T
        o  = S^T q * Dk^-0.5

    Shapes (B, L, H, D); with cu_seqlens, B must be 1 and the state resets at
    each boundary. Slow and exact; used to test indexing, not speed.
    """
    B, L, H, Dk = q.shape
    Dv = v.shape[-1]
    scale = Dk ** -0.5
    if l2norm:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    o = torch.zeros(B, L, H, Dv, dtype=q.dtype, device=q.device)
    if cu_seqlens is None:
        bounds = [(b, 0, L) for b in range(B)]
    else:
        assert B == 1, "cu_seqlens requires batch 1"
        c = cu_seqlens.tolist()
        bounds = [(0, c[i], c[i + 1]) for i in range(len(c) - 1)]
    for b, s, e in bounds:
        S = torch.zeros(H, Dk, Dv, dtype=q.dtype, device=q.device)
        for t in range(s, e):
            if g is not None:
                S = S * torch.exp(g[b, t]).view(H, 1, 1)
            kt, vt, bt = k[b, t], v[b, t], beta[b, t]
            v_old = torch.einsum("hk,hkv->hv", kt, S)
            S = S + bt.view(H, 1, 1) * torch.einsum("hk,hv->hkv", kt, vt - v_old)
            o[b, t] = torch.einsum("hk,hkv->hv", q[b, t], S) * scale
    return o


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _random_case(B=3, T=11, K=4, H=2, Dk=5, Dv=3, p_high=0.3, seed=0,
                 with_decay=True, allow_zero=True):
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = torch.randn(B, T, H, Dk, generator=gen, dtype=dt)
    ks = [torch.randn(B, T, H, Dk, generator=gen, dtype=dt) for _ in range(K)]
    vs = [torch.randn(B, T, H, Dv, generator=gen, dtype=dt) for _ in range(K)]
    raw_b = [2 * torch.rand(B, T, H, generator=gen, dtype=dt) for _ in range(K)]
    g = (-torch.rand(B, T, H, generator=gen, dtype=dt)) if with_decay else None
    # per-token order: mostly 1, some K, a few in between, optionally 0
    u = torch.rand(B, T, generator=gen)
    n = torch.where(u < p_high, torch.full_like(u, K), torch.ones_like(u))
    mid = torch.rand(B, T, generator=gen) < 0.2
    n = torch.where(mid, torch.randint(1, K + 1, (B, T), generator=gen).float(), n)
    if allow_zero:
        n = torch.where(torch.rand(B, T, generator=gen) < 0.05, torch.zeros_like(n), n)
    gates = (torch.arange(K).view(1, 1, K) < n.unsqueeze(-1)).to(dt)
    betas = [raw_b[i] * gates[..., i].unsqueeze(-1) for i in range(K)]
    return q, ks, vs, betas, g, gates


def selftest() -> bool:
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    kern_packed = lambda q, k, v, g, b, cu: reference_gated_delta(q, k, v, g, b, cu)
    kern_batched = lambda q, k, v, g, b: reference_gated_delta(q, k, v, g, b, None)

    print("ragged expansion self-test (float64, CPU, reference kernel)")

    # R1: the plan on a hand-made case
    n = torch.tensor([[1, 4, 0, 2], [3, 1, 1, 4]])
    keep, last, cu, rp = ragged_plan(n, 4)
    check("R1 plan: kept counts, boundaries and read positions by hand",
          keep.sum().item() == 1 + 4 + 1 + 2 + 3 + 1 + 1 + 4
          and cu.tolist() == [0, 8, 17]
          and rp.tolist() == [[0, 4, 5, 7], [10, 11, 12, 16]],
          f"cu {cu.tolist()}, read {rp.tolist()}")

    # R2: equality with Path A on random cases, with and without decay
    worst = 0.0
    for seed in range(6):
        for decay in (True, False):
            q, ks, vs, betas, g, gates = _random_case(seed=seed, with_decay=decay)
            a = path_a_core(q, ks, vs, betas, g, kern_batched)
            r = ragged_core(q, ks, vs, betas, g, gates, kern_packed)
            worst = max(worst, (a - r).abs().max().item())
    check("R2 ragged == Path A, 12 random cases (orders 0-4, with/without decay)",
          worst < 1e-12, f"max |diff| {worst:.2e}")

    # R3: all tokens at full order -> ragged is Path A with nothing removed
    q, ks, vs, betas, g, gates = _random_case(seed=7, p_high=1.0, allow_zero=False)
    gates = torch.ones_like(gates)
    a = path_a_core(q, ks, vs, betas, g, kern_batched)
    r = ragged_core(q, ks, vs, betas, g, gates, kern_packed)
    check("R3 all gates open: identical to Path A", (a - r).abs().max().item() < 1e-12)

    # R4: the saving is what it should be
    q, ks, vs, betas, g, gates = _random_case(seed=3)
    n_t = validate_gates(gates)
    keep, _, cu, _ = ragged_plan(n_t, 4)
    check("R4 packed length = sum of max(n_t, 1)",
          int(cu[-1]) == int(n_t.clamp(min=1).sum()),
          f"{int(cu[-1])} micro-steps instead of {4 * n_t.numel()}")

    # R5-R8: deliberately broken variants must FAIL the R2 comparison
    q, ks, vs, betas, g, gates = _random_case(seed=11)
    a = path_a_core(q, ks, vs, betas, g, kern_batched)
    for mut, label in [("no_cu_seqlens", "R5 state leaks across sequences"),
                       ("query_at_first", "R6 query on the first kept step"),
                       ("read_at_first", "R7 output read at the first kept step"),
                       ("decay_at_last", "R8 decay on the last kept step")]:
        r = ragged_core(q, ks, vs, betas, g, gates, kern_packed, _mutant=mut)
        d = (a - r).abs().max().item()
        check(f"{label}: mutant is detected", d > 1e-3, f"max |diff| {d:.2e}")

    # R12-R14: stacked ("fused") variants against the per-factor Path A
    worst_a = worst_r = 0.0
    for seed in range(6):
        for decay in (True, False):
            q, ks, vs, betas, g, gates = _random_case(seed=seed, with_decay=decay)
            ref = path_a_core(q, ks, vs, betas, g, kern_batched)
            k_all, v_all = torch.stack(ks, 2), torch.stack(vs, 2)
            b_all = torch.stack(betas, 2)
            a = path_a_stacked(q, k_all, v_all, b_all, g, kern_batched)
            r = ragged_stacked(q, k_all, v_all, b_all, g, gates, kern_packed)
            worst_a = max(worst_a, (ref - a).abs().max().item())
            worst_r = max(worst_r, (ref - r).abs().max().item())
    check("R12 stacked Path A == per-factor Path A, 12 cases", worst_a < 1e-12,
          f"max |diff| {worst_a:.2e}")
    check("R13 stacked ragged == per-factor Path A, 12 cases", worst_r < 1e-12,
          f"max |diff| {worst_r:.2e}")
    q, ks, vs, betas, g, gates = _random_case(seed=11)
    ref = path_a_core(q, ks, vs, betas, g, kern_batched)
    k_all, v_all, b_all = torch.stack(ks, 2), torch.stack(vs, 2), torch.stack(betas, 2)
    for mut in ("no_cu_seqlens", "query_at_first", "read_at_first", "decay_at_last"):
        r = ragged_stacked(q, k_all, v_all, b_all, g, gates, kern_packed, _mutant=mut)
        d = (ref - r).abs().max().item()
        check(f"R14 stacked mutant '{mut}' is detected", d > 1e-3, f"max |diff| {d:.2e}")

    # R9-R11: invalid gates are refused, not silently mis-packed
    for name, bad in [("R9 soft gates refused", torch.full((1, 2, 4), 0.7)),
                      ("R10 non-prefix gates refused",
                       torch.tensor([[[1., 0., 1., 0.]]])),
                      ("R11 per-head gates refused", torch.ones(1, 2, 4, 3))]:
        try:
            validate_gates(bad)
            raised = False
        except ValueError:
            raised = True
        check(name, raised)

    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
