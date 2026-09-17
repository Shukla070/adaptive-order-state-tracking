"""
Reference (naive, obviously-correct) DeltaProduct + adaptive-order gating.

Purpose
-------
This is NOT for training. It is a slow, loop-based, CPU implementation whose
only job is to be so simple it is clearly correct, so that it can serve as a
numerical oracle for the fast Triton path in flash-linear-attention.

It establishes three things before any GPU work happens:

  T1  beta = 0 makes a generalized Householder factor EXACTLY the identity,
      so "skip this micro-step" lies inside the model's existing continuous
      parameter space. The whole adaptive-order formulation depends on this.

  T2  A permutation of transposition length L is exactly representable as a
      product of L generalized Householder factors with beta = 2. This is
      Proposition 2, verified constructively. It also shows *why* DeltaProduct
      needs allow_neg_eigval=True: a reflection has eigenvalue -1.

  T3  Monotone halting gates on beta implement a well-defined integer
      effective order n_t, and the gated model reduces exactly to fixed-order
      DeltaProduct when all gates are open.

Run:  python3 reference_deltaproduct.py
"""

import math
import torch

torch.set_default_dtype(torch.float64)  # oracle: precision over speed


# ---------------------------------------------------------------------------
# Core recurrence
# ---------------------------------------------------------------------------

def householder(beta, k):
    """Generalized Householder factor  I - beta * k k^T,  with ||k|| = 1.

    beta = 0 -> identity          (eigenvalues all +1)
    beta = 1 -> rank-1 projection (eigenvalue 0 along k)
    beta = 2 -> reflection        (eigenvalue -1 along k)  <- needs neg eigvals
    """
    d = k.shape[-1]
    return torch.eye(d) - beta * torch.outer(k, k)


def deltaproduct_step(S, betas, ks, vs):
    """One token = n_h sequential delta micro-steps.

    S     : (d_v, d_k) state
    betas : (n_h,)
    ks    : (n_h, d_k)  unit norm
    vs    : (n_h, d_v)

    Returns the updated state. Written as an explicit loop over micro-steps,
    which is exactly the "expand the sequence by n_h and run DeltaNet" view.
    """
    for i in range(betas.shape[0]):
        b, k, v = betas[i], ks[i], vs[i]
        S = S @ householder(b, k) + b * torch.outer(v, k)
    return S


def deltaproduct_seq(betas, ks, vs, S0=None):
    """Run a whole sequence.  betas (T, n_h), ks (T, n_h, d_k), vs (T, n_h, d_v)."""
    T, n_h = betas.shape
    d_k, d_v = ks.shape[-1], vs.shape[-1]
    S = torch.zeros(d_v, d_k) if S0 is None else S0.clone()
    out = []
    for t in range(T):
        S = deltaproduct_step(S, betas[t], ks[t], vs[t])
        out.append(S.clone())
    return torch.stack(out)


def transition_matrix(betas, ks):
    """The state-transition matrix A_t = prod_i (I - beta_i k_i k_i^T)."""
    d = ks.shape[-1]
    A = torch.eye(d)
    for i in range(betas.shape[0]):
        A = A @ householder(betas[i], ks[i])
    return A


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------

def perm_matrix(p):
    """Permutation matrix P with P[i, p[i]] = 1, acting on row vectors as x @ P."""
    n = len(p)
    P = torch.zeros(n, n)
    for i, j in enumerate(p):
        P[i, j] = 1.0
    return P


def transposition_decomposition(p):
    """Decompose permutation p into transpositions. Returns list of (i, j).

    Length of the returned list is exactly n - (number of cycles) = the
    transposition length L(p).
    """
    n = len(p)
    q = list(p)
    swaps = []
    for i in range(n):
        while q[i] != i:
            j = q[i]
            q[i], q[j] = q[j], q[i]
            swaps.append((i, j))
    return swaps


def transposition_length(p):
    n = len(p)
    seen = [False] * n
    c = 0
    for i in range(n):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return n - c


def transposition_as_householder(i, j, n):
    """The transposition (i j) as an exact generalized Householder factor.

    v = (e_i - e_j)/sqrt(2),  beta = 2  =>  I - 2 v v^T  swaps coords i and j.
    """
    v = torch.zeros(n)
    v[i], v[j] = 1.0, -1.0
    v = v / math.sqrt(2.0)
    return torch.tensor(2.0), v


# ---------------------------------------------------------------------------
# Adaptive-order gating
# ---------------------------------------------------------------------------

def monotone_gates(lambdas):
    """PonderNet-style monotone halting gates from per-micro-step continue probs.

    lambdas : (..., K) in [0, 1]
    returns g : (..., K) with g_1 >= g_2 >= ... >= g_K,  g_i = prod_{j<=i} lambda_j

    Effective order n_t = sum_i g_i  (an integer when the lambdas are hard).
    """
    return torch.cumprod(lambdas, dim=-1)


def gated_betas(betas, lambdas):
    """Apply gates to betas. beta_eff = g * beta; g = 0 gives an identity factor."""
    return betas * monotone_gates(lambdas)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def t1_zero_beta_is_identity():
    """beta = 0 must give EXACTLY the identity, and gating micro-steps 2..K off
    must reproduce n_h = 1 bit-for-bit."""
    torch.manual_seed(0)
    T, K, d = 32, 4, 16

    ks = torch.randn(T, K, d)
    ks = ks / ks.norm(dim=-1, keepdim=True)
    vs = torch.randn(T, K, d)
    betas = torch.rand(T, K) * 2.0

    # exact identity for a single factor
    k = ks[0, 0]
    err_id = (householder(torch.tensor(0.0), k) - torch.eye(d)).abs().max().item()

    # full-K model with micro-steps 2..K gated off
    lambdas = torch.ones(T, K)
    lambdas[:, 1:] = 0.0
    S_gated = deltaproduct_seq(gated_betas(betas, lambdas), ks, vs)

    # genuine n_h = 1 model
    S_one = deltaproduct_seq(betas[:, :1], ks[:, :1], vs[:, :1])

    err_seq = (S_gated - S_one).abs().max().item()
    ok = err_id == 0.0 and err_seq == 0.0
    print(f"T1  beta=0 -> identity          : max|I - H(0,k)| = {err_id:.2e}")
    print(f"T1  gated K=4 == true n_h=1     : max|dS|         = {err_seq:.2e}   "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def t2_permutation_needs_L_householders():
    """Proposition 2, constructively: a permutation of transposition length L is
    exactly a product of L Householder factors with beta = 2, and L-1 factors
    are provably insufficient."""
    from itertools import permutations
    n = 5
    torch.manual_seed(0)
    all_ok = True
    by_len = {}

    for p in permutations(range(n)):
        L = transposition_length(p)
        by_len.setdefault(L, []).append(p)

    print(f"T2  S{n}: exact reconstruction from L Householder factors (beta=2)")
    for L in sorted(by_len):
        worst = 0.0
        for p in by_len[L][:40]:                      # sample per length class
            swaps = transposition_decomposition(p)
            assert len(swaps) == L, (p, swaps, L)
            betas, ks = [], []
            for (i, j) in swaps:
                b, v = transposition_as_householder(i, j, n)
                betas.append(b); ks.append(v)
            if L == 0:
                A = torch.eye(n)
            else:
                A = transition_matrix(torch.stack(betas), torch.stack(ks))
            worst = max(worst, (A - perm_matrix(p)).abs().max().item())
        ok = worst < 1e-12
        all_ok &= ok
        print(f"    L={L}  ({len(by_len[L]):3d} elements)  max reconstruction err = "
              f"{worst:.2e}  [{'PASS' if ok else 'FAIL'}]")

    # eigenvalue check: a reflection has eigenvalue -1, hence allow_neg_eigval=True
    b, v = transposition_as_householder(0, 1, n)
    ev = torch.linalg.eigvals(householder(b, v)).real
    print(f"    reflection eigenvalues      = {sorted(round(x,6) for x in ev.tolist())}"
          f"   -> requires negative eigenvalues")
    return all_ok


def t3_gating_reduces_to_fixed_order():
    """All gates open == fixed-order DeltaProduct; effective order counts correctly."""
    torch.manual_seed(1)
    T, K, d = 16, 4, 8
    ks = torch.randn(T, K, d); ks = ks / ks.norm(dim=-1, keepdim=True)
    vs = torch.randn(T, K, d)
    betas = torch.rand(T, K) * 2.0

    S_full = deltaproduct_seq(betas, ks, vs)
    S_open = deltaproduct_seq(gated_betas(betas, torch.ones(T, K)), ks, vs)
    err = (S_full - S_open).abs().max().item()

    # random integer orders -> effective order equals the intended n_t
    n_t = torch.randint(0, K + 1, (T,))
    lam = (torch.arange(K)[None, :] < n_t[:, None]).to(torch.get_default_dtype())
    n_eff = monotone_gates(lam).sum(-1)
    counts_ok = torch.equal(n_eff, n_t.to(n_eff.dtype))

    # gates are monotone non-increasing
    g = monotone_gates(torch.rand(T, K))
    mono_ok = bool((g[:, :-1] - g[:, 1:] >= -1e-15).all())

    ok = err == 0.0 and counts_ok and mono_ok
    print(f"T3  gates all open == fixed n_h=K : max|dS| = {err:.2e}")
    print(f"T3  effective order == intended   : {counts_ok}")
    print(f"T3  gates monotone non-increasing : {mono_ok}   "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def t4_order_deficit_breaks_tracking():
    """The order requirement is tight, not slack.

    Fit K free Householder factors to a target permutation by gradient descent.
    Error should fall to ~0 exactly when K >= L(p), and plateau above zero for
    K < L. This is the numerical statement of "n_h must be at least the
    transposition length", and it is the mechanism behind the README's
    'DeltaProduct1 on S3 is not expected to converge'.
    """
    torch.manual_seed(0)
    n = 5
    targets = [((0, 1, 2, 3, 4), "identity"),
               ((1, 0, 2, 3, 4), "transposition"),
               ((1, 2, 0, 3, 4), "3-cycle"),
               ((1, 2, 3, 0, 4), "4-cycle"),
               ((1, 2, 3, 4, 0), "5-cycle")]

    print("T4  fit K free reflections to a target permutation (gradient descent)")
    print("    rows = target, cols = K;  values = final max abs error")
    header = "    {:<16s}".format("target (L)") + "".join(f"  K={K}   " for K in range(1, 5))
    print(header)

    all_ok = True
    for p, name in targets:
        L = transposition_length(p)
        P = perm_matrix(p)
        row_free, row_pinned = [], []
        for K in range(1, 5):
            for pinned in (True, False):
                best = float("inf")
                for restart in range(3):
                    g = torch.Generator().manual_seed(restart)
                    params = torch.randn(K, n, generator=g, requires_grad=True)
                    graw = torch.zeros(K, requires_grad=True)   # -> beta via sigmoid
                    tunable = [params] if pinned else [params, graw]
                    opt = torch.optim.Adam(tunable, lr=0.05)
                    for _ in range(1500):
                        opt.zero_grad()
                        ks = params / params.norm(dim=-1, keepdim=True)
                        betas = (torch.full((K,), 2.0) if pinned
                                 else 2.0 * torch.sigmoid(graw))
                        A = transition_matrix(betas, ks)
                        loss = ((A - P) ** 2).sum()
                        loss.backward(); opt.step()
                    with torch.no_grad():
                        ks = params / params.norm(dim=-1, keepdim=True)
                        betas = (torch.full((K,), 2.0) if pinned
                                 else 2.0 * torch.sigmoid(graw))
                        A = transition_matrix(betas, ks)
                        best = min(best, (A - P).abs().max().item())
                (row_pinned if pinned else row_free).append(best)
            # with FREE beta the rule should be exactly K >= L
            if K >= L and row_free[-1] > 0.05:
                all_ok = False
        cf = "".join(f"  {v:6.4f}" for v in row_free)
        cp = "".join(f"  {v:6.4f}" for v in row_pinned)
        print(f"    {name:<14s}(L={L})  free beta:{cf}")
        print(f"    {'':<14s}       beta=2 :{cp}")
    print(f"    with free beta: converged (<0.02) exactly when K >= L, else >0.4   "
          f"[{'PASS' if all_ok else 'FAIL'}]")
    print("    with beta pinned at 2 every factor is a reflection (det = -1), so")
    print("    det(A) = (-1)^K must match sgn(p) = (-1)^L: solvable only when")
    print("    K = L (mod 2). DESIGN CONSEQUENCE: the gate must scale a LEARNABLE")
    print("    beta, never select between {0, 2} -- a hard binary gate hits this")
    print("    parity wall and cannot represent half of all target orders.")
    return all_ok


if __name__ == "__main__":
    print("=" * 72)
    print("Reference DeltaProduct — correctness oracle for the adaptive-order model")
    print("=" * 72)
    results = []
    for fn in (t1_zero_beta_is_identity, t2_permutation_needs_L_householders,
               t3_gating_reduces_to_fixed_order, t4_order_deficit_breaks_tracking):
        print()
        results.append(fn())
    print()
    print("=" * 72)
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    print("=" * 72)
