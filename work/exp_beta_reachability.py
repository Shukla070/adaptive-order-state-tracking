"""
FREE-BETA REACHABILITY  --  what a product of K generalized Householders can be.

WHY THIS EXISTS
---------------
Everything in this project rests on one inherited assumption:

    to apply permutation g, a DeltaProduct layer needs L(g) Householder factors

where L(g) = n - (number of cycles) is the transposition length. DeltaProduct
assumes it, our oracle is built on it, and our halting gate is trained to
predict it. But our own runs keep contradicting it: A5 trains at n_h=2 when the
theory wants 4, and fixed n_h=3 reaches 54% sequence accuracy on a task whose
hard tokens have L=4.

So: is K >= L actually forced, or is it an artifact of pinning beta = 2?

WHAT BETA IS
------------
A generalized Householder factor is

    H(beta, k) = I - beta * k k^T ,   ||k|| = 1,   beta in [0, 2]

It touches only the direction k and leaves the orthogonal complement alone,
scaling the k-component by (1 - beta):

    beta = 0  ->  identity            (do nothing)
    beta = 1  ->  projection          (erase the k component)   <- the delta rule
    beta = 2  ->  reflection          (flip the k component)    <- orthogonal

beta is the strength of the delta-rule write. beta = 2 is the interesting end:
a transposition (i j) IS a reflection, H(2, (e_i - e_j)/sqrt2), so a permutation
of length L is a product of L reflections. That is where K = L comes from, and
why DeltaProduct needs allow_neg_eigval (beta up to 2, eigenvalue 1-beta = -1).

THE PARITY WALL (ours, from the oracle's test T4)
-------------------------------------------------
det H(beta, k) = 1 - beta, so with beta pinned at 2 a product of K factors has
det = (-1)^K, while det(P) = sgn(p) = (-1)^L. Pinned beta therefore reaches p
only when K = L (mod 2). A halting gate that switches a factor off flips the
determinant and lands in the wrong coset -- which is why the gate must scale a
LEARNABLE beta rather than mask a fixed one.

Free beta removes that obstruction. The question this file answers is whether it
removes the K >= L bound as well.

THE PREDICTION
--------------
It does not, and the reason is rank, not orthogonality:

    A = prod_i H(beta_i, k_i)  =>  A - I has column space inside span{k_1..k_K}
    =>  rank(A - I) <= K

and for a permutation matrix, rank(P - I) = n - cycles(p) = L(p) exactly. So
K >= L is forced for EVERY beta, free or pinned. Free beta buys the parity
coset, nothing more.

E4 pushes on the obvious escape hatch: the model works in 32 dimensions, not 5,
and is free to pick its own embedding of S5. Does a cleverer representation make
5-cycles cheaper? Character theory says no -- for a 5-cycle,

    rank(rho(g) - I) = 4 * (number of non-linear irrep constituents of rho)

so it is 4 in the smallest faithful representations and a larger multiple of 4
in every bigger one. Widening the state cannot help and usually hurts. E4 checks
that numerically over a family of constructed representations.

RUN
---
    python work/exp_beta_reachability.py               # E1-E4, ~10 min on CPU
    python work/exp_beta_reachability.py --quick       # coarse, ~1 min
    python work/exp_beta_reachability.py --only E2
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)          # this whole file is float64 CPU
DEV = "cpu"
N = 5


# ---------------------------------------------------------------------------
# group machinery
# ---------------------------------------------------------------------------

ELEMENTS = list(itertools.permutations(range(N)))          # 120 of them


def n_cycles(p) -> int:
    seen = [False] * len(p)
    c = 0
    for i in range(len(p)):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return c


def ell(p) -> int:
    """Transposition length = n - cycles. Also equals rank(P - I)."""
    return len(p) - n_cycles(p)


def perm_matrix(p) -> torch.Tensor:
    """P e_i = e_{p(i)}, so P[p[i], i] = 1."""
    P = torch.zeros(len(p), len(p))
    for i, pi in enumerate(p):
        P[pi, i] = 1.0
    return P


ELL = [ell(p) for p in ELEMENTS]


# ---------------------------------------------------------------------------
# the factorisation
# ---------------------------------------------------------------------------

def householder(beta: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    d = k.shape[-1]
    return torch.eye(d, dtype=k.dtype) - beta * torch.outer(k, k)


def build(betas: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """Ordered product H_K ... H_2 H_1 of K generalized Householder factors."""
    d = ks.shape[-1]
    A = torch.eye(d, dtype=ks.dtype)
    for i in range(ks.shape[0]):
        A = householder(betas[i], ks[i]) @ A
    return A


def fit(target: torch.Tensor, K: int, mode: str = "free",
        restarts: int = 12, adam_steps: int = 600, lbfgs_steps: int = 120,
        seed: int = 0) -> dict:
    """Best achievable ||prod H_i - target||_F over (beta_i, k_i).

    k is parametrized by a raw vector and normalized, so ||k|| = 1 is exact
    rather than penalized.

    beta modes:
      "free"   beta is an unconstrained real. This is the mode for the
               reachability question, because the architecture's box [0,2] is a
               separate constraint and we want to know which obstruction binds.
               Feasibility is REPORTED, not imposed: beta_min / beta_max come
               back with the residual so the caller can see whether the solution
               a free optimizer found is one the real layer could express.
      "box"    beta = 2*sigmoid(raw), the architecture's actual reach. Note this
               is the OPEN interval (0,2) -- it approaches a reflection but never
               attains one, so exact targets floor out around 1e-5 here. That is
               a property of the parametrization, not of the problem, which is
               exactly why "free" is the right mode for reachability and "box"
               is the right mode for measuring approximation quality.
      "pinned" beta = 2 exactly, pure reflections. This is the regime the parity
               wall lives in.

    Adam finds the basin, LBFGS drives it to machine precision. Without the
    LBFGS stage the residual floors out around 1e-4 and every K looks like a
    partial success, which would have made this whole test unreadable.
    """
    d = target.shape[0]
    if K == 0:
        e = (torch.eye(d) - target).norm().item()
        return {"residual": e, "beta_min": None, "beta_max": None,
                "feasible": True}

    best = {"residual": float("inf"), "beta_min": None, "beta_max": None,
            "feasible": False}
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 1000 + r)
        kraw = torch.randn(K, d, generator=g, requires_grad=True)
        params = [kraw]
        braw = None
        if mode == "free":
            braw = (torch.ones(K) + 0.5 * torch.randn(K, generator=g)
                    ).requires_grad_(True)
            params.append(braw)
        elif mode == "box":
            braw = (0.5 * torch.randn(K, generator=g)).requires_grad_(True)
            params.append(braw)

        def betas():
            if mode == "free":
                return braw
            if mode == "box":
                return 2.0 * torch.sigmoid(braw)
            return torch.full((K,), 2.0, dtype=kraw.dtype)

        def residual():
            k = kraw / kraw.norm(dim=-1, keepdim=True).clamp_min(1e-300)
            return (build(betas(), k) - target).norm()

        opt = torch.optim.Adam(params, lr=0.05)
        for _ in range(adam_steps):
            opt.zero_grad()
            loss = residual()
            loss.backward()
            opt.step()

        opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs_steps,
                                 tolerance_grad=1e-16, tolerance_change=1e-18,
                                 history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss = residual()
            loss.backward()
            return loss

        opt2.step(closure)
        with torch.no_grad():
            e = residual().item()
            if e < best["residual"]:
                b = betas()
                best = {"residual": e,
                        "beta_min": b.min().item(),
                        "beta_max": b.max().item(),
                        "feasible": bool((b >= -1e-9).all()
                                         and (b <= 2.0 + 1e-9).all())}
        if best["residual"] < 1e-12:
            break
    return best


# ---------------------------------------------------------------------------
# representations, for E4
# ---------------------------------------------------------------------------

def rep_permutation(p):
    return perm_matrix(p)


def rep_standard(p):
    """The 4-dim irrep: permutation rep restricted to the sum-zero subspace."""
    P = perm_matrix(p)
    # orthonormal basis of {x : sum x = 0}
    B = torch.zeros(N, N - 1)
    for j in range(N - 1):
        v = torch.zeros(N)
        v[: j + 1] = 1.0
        v[j + 1] = -(j + 1.0)
        B[:, j] = v / v.norm()
    return B.T @ P @ B


def rep_direct_sum(reps):
    def f(p):
        blocks = [r(p) for r in reps]
        d = sum(b.shape[0] for b in blocks)
        M = torch.zeros(d, d)
        o = 0
        for b in blocks:
            s = b.shape[0]
            M[o:o + s, o:o + s] = b
            o += s
        return M
    return f


def rep_tensor(r1, r2):
    return lambda p: torch.kron(r1(p), r2(p))


def rep_regular(p):
    """|G| x |G|: left multiplication by p on the group itself."""
    idx = {e: i for i, e in enumerate(ELEMENTS)}
    M = torch.zeros(len(ELEMENTS), len(ELEMENTS))
    for j, e in enumerate(ELEMENTS):
        prod = tuple(p[e[i]] for i in range(N))      # p o e
        M[idx[prod], j] = 1.0
    return M


def rep_padded(r, extra):
    def f(p):
        b = r(p)
        d = b.shape[0] + extra
        M = torch.eye(d)
        M[: b.shape[0], : b.shape[0]] = b
        return M
    return f


def is_faithful(r) -> bool:
    ident = tuple(range(N))
    d = r(ident).shape[0]
    I = torch.eye(d)
    for p in ELEMENTS:
        if p != ident and (r(p) - I).abs().max() < 1e-9:
            return False
    return True


def mat_rank(M, tol=1e-9) -> int:
    return int((torch.linalg.svdvals(M) > tol).sum().item())


# ---------------------------------------------------------------------------
# E1 -- the rank bound
# ---------------------------------------------------------------------------

def E1(args) -> dict:
    print("=" * 78)
    print("E1  rank(A - I) <= K for any beta,  and  rank(P - I) = L(p)")
    print("=" * 78)

    ok_bound = True
    worst = []
    g = torch.Generator().manual_seed(11)
    for d in (5, 8, 16):
        for K in range(0, 7):
            for _ in range(40):
                ks = torch.randn(K, d, generator=g)
                ks = ks / ks.norm(dim=-1, keepdim=True)
                betas = 2.0 * torch.rand(K, generator=g)       # free in [0,2)
                A = build(betas, ks)
                r = mat_rank(A - torch.eye(d))
                if r > K:
                    ok_bound = False
                    worst.append((d, K, r))
    print(f"  random free-beta products, d in 5/8/16, K in 0..6, 40 draws each")
    print(f"    rank(A - I) <= K always : {ok_bound}   "
          f"[{'PASS' if ok_bound else 'FAIL'}]")
    if worst:
        print(f"    violations: {worst[:5]}")

    ok_perm = True
    for p, L in zip(ELEMENTS, ELL):
        if mat_rank(perm_matrix(p) - torch.eye(N)) != L:
            ok_perm = False
    print(f"    rank(P - I) == L(p) for all 120 : {ok_perm}   "
          f"[{'PASS' if ok_perm else 'FAIL'}]")
    print()
    print("  => K >= L(p) is forced for every beta. Free beta cannot beat it.")
    print()
    return {"rank_bound": ok_bound, "rank_equals_ell": ok_perm}


# ---------------------------------------------------------------------------
# E2 / E3 -- reachability by optimization, free vs pinned beta
# ---------------------------------------------------------------------------

TOL = 1e-8


# ---------------------------------------------------------------------------
# E2 / E3 -- reachability, by construction rather than by optimizer
# ---------------------------------------------------------------------------
#
# An earlier version of this file answered E2 by gradient descent. That was a
# mistake, and an instructive one: with beta = 2*sigmoid(raw) the parametrization
# cannot ATTAIN beta = 2, so exact reflections floored out near 1e-5; and with
# beta unconstrained the optimizer kept collapsing to A = I (residual exactly
# ||P - I||, the giveaway). Both are properties of the harness, not of the
# problem, and either one would have been reported as a result.
#
# So reachability is settled the right way instead:
#   achievable  -- by explicit construction, checked to machine precision,
#                  for all 120 elements at every K. No optimizer involved.
#   impossible  -- by the rank bound of E1: rank(A - I) <= K < L = rank(P - I).
# Optimization is kept only for E5, where the question is how good the best
# APPROXIMATION is, and a good number is all that is wanted.

def transposition_factor(i: int, j: int, d: int):
    """The transposition (i j) as a Householder reflection: beta = 2 exactly."""
    v = torch.zeros(d)
    v[i], v[j] = 1.0, -1.0
    return 2.0, v / math.sqrt(2.0)


def decompose(p) -> list[tuple[int, int]]:
    """p as a product of exactly L(p) transpositions.

    Left-multiplying P by the matrix of (x y) swaps rows x and y, i.e. sends
    p to (x y) o p. Repeatedly clearing position i gives T_m ... T_1 P = I, so
    P = T_1 ... T_m, and m = L(p) because each step raises the cycle count by
    one. The caller reverses the list to match build()'s ordering.
    """
    a = list(p)
    out = []
    for i in range(len(a)):
        if a[i] != i:
            j = a.index(i)
            out.append((a[i], i))
            a[i], a[j] = a[j], a[i]
    return out


def construct(p, K: int, pinned: bool):
    """Realize P as a product of exactly K generalized Householder factors.

    L reflections do the work. The remaining K - L factors must compose to the
    identity, and there are only two ways to build an identity out of these:

        a PAIR of identical reflections   H(2,k) H(2,k) = I     (costs 2, beta=2)
        a SINGLE closed factor            H(0,k)       = I      (costs 1, beta=0)

    Pinned beta = 2 has only the first, so it can only pad by even amounts --
    that IS the parity wall, stated constructively. A gate that can drive beta
    to 0 unlocks the odd padding, and with it every K >= L.

    Returns (betas, ks) or None when the padding cannot be built.
    """
    d = len(p)
    L = ell(p)
    if K < L:
        return None
    facs = [transposition_factor(i, j, d) for (i, j) in decompose(p)]
    facs = facs[::-1]                      # build() applies index 0 first
    pad = K - L
    if pinned:
        if pad % 2:
            return None                    # no identity factor available
        filler = torch.zeros(d)
        filler[0] = 1.0
        facs = [(2.0, filler)] * pad + facs
    else:
        extra = []
        if pad % 2:
            z = torch.zeros(d)
            z[0] = 1.0
            extra.append((0.0, z))         # one closed factor
            pad -= 1
        filler = torch.zeros(d)
        filler[0] = 1.0
        extra = [(2.0, filler)] * pad + extra
        facs = extra + facs
    betas = torch.tensor([b for b, _ in facs])
    ks = torch.stack([k for _, k in facs]) if facs else torch.zeros(0, d)
    return betas, ks


def reach_table(pinned: bool, args) -> dict:
    """Every one of the 120 elements, at every K in 0..4, by construction."""
    grid = {L: [None] * N for L in range(N)}
    worst = {L: [0.0] * N for L in range(N)}
    for p, L in zip(ELEMENTS, ELL):
        P = perm_matrix(p)
        for K in range(N):
            c = construct(p, K, pinned)
            if c is None:
                grid[L][K] = False
            else:
                err = (build(*c) - P).norm().item()
                grid[L][K] = err < 1e-12
                worst[L][K] = max(worst[L][K], err)

    label = ("PINNED beta = 2 (pure reflections, no gate)" if pinned
             else "LEARNABLE beta in [0,2] (a gate can close a factor)")
    print(f"  {label}")
    print(f"    . = reached exactly by construction     x = not reachable")
    print(f"    {'':>6}" + "".join(f"{'K=' + str(K):>8}" for K in range(N)))
    for L in range(N):
        cells = ["." if grid[L][K] else "x" for K in range(N)]
        print(f"    L={L}   " + "".join(f"{c:>8}" for c in cells))
    w = max(max(r) for r in worst.values())
    print(f"    worst reconstruction error over all 120 elements: {w:.2e}")
    return {str(L): grid[L] for L in grid}


def E2(args) -> dict:
    print("=" * 78)
    print("E2/E3  exact reachability -- learnable beta vs beta pinned at 2")
    print("=" * 78)
    free = reach_table(False, args)
    print()
    pinned = reach_table(True, args)
    print()

    free_ok = all(free[str(L)][K] == (K >= L) for L in range(N) for K in range(N))
    parity_ok = all(pinned[str(L)][K] == (K >= L and (K - L) % 2 == 0)
                    for L in range(N) for K in range(N))
    print(f"  learnable beta reaches p iff K >= L        : {free_ok}   "
          f"[{'PASS' if free_ok else 'FAIL'}]")
    print(f"  pinned beta reaches p iff K >= L AND       : {parity_ok}   "
          f"[{'PASS' if parity_ok else 'FAIL'}]")
# ---------------------------------------------------------------------------
# E4 -- can a different representation make 5-cycles cheaper?
# ---------------------------------------------------------------------------

def E4(args) -> dict:
    print("=" * 78)
    print("E4  does a wider / different representation lower the required order?")
    print("=" * 78)

    five = next(p for p, e in zip(ELEMENTS, ELL) if e == 4)
    trans = next(p for p, e in zip(ELEMENTS, ELL) if e == 1)

    std = rep_standard
    perm = rep_permutation
    cands = [
        ("standard (4)",            std),
        ("permutation (5)",         perm),
        ("perm padded to 8",        rep_padded(perm, 3)),
        ("perm padded to 32",       rep_padded(perm, 27)),
        ("std + std (8)",           rep_direct_sum([std, std])),
        ("perm + perm (10)",        rep_direct_sum([perm, perm])),
        ("std (x) std (16)",        rep_tensor(std, std)),
        ("perm (x) perm (25)",      rep_tensor(perm, perm)),
        ("std (x) std + std (20)",  rep_direct_sum([rep_tensor(std, std), std])),
        ("regular (120)",           rep_regular),
    ]

    print(f"  {'representation':<24} {'dim':>5} {'faithful':>9} "
          f"{'rank(g5 - I)':>13} {'rank(t - I)':>12} {'mult of 4':>10}")
    print("  " + "-" * 76)
    rows = []
    all_mult4 = True
    min_rank = math.inf
    for name, r in cands:
        d = r(tuple(range(N))).shape[0]
        faith = is_faithful(r)
        r5 = mat_rank(r(five) - torch.eye(d))
        rt = mat_rank(r(trans) - torch.eye(d))
        m4 = (r5 % 4 == 0) and r5 >= 4
        if faith:
            all_mult4 &= m4
            min_rank = min(min_rank, r5)
        rows.append(dict(name=name, dim=d, faithful=faith,
                         rank_5cycle=r5, rank_transposition=rt))
        print(f"  {name:<24} {d:>5} {str(faith):>9} {r5:>13} {rt:>12} "
              f"{str(m4):>10}")

    print()
    print(f"  every faithful rep has rank(5-cycle - I) = 4 * m, m >= 1 : "
          f"{all_mult4}   [{'PASS' if all_mult4 else 'FAIL'}]")
    print(f"  minimum over faithful reps tested                        : "
          f"{min_rank}")
    print()
    print("  Character theory says why. For g of order m,")
    print("      dim Fix(g) = (1/m) sum_j chi(g^j),")
    print("  and for a 5-cycle in S5 this collapses to")
    print("      rank(rho(g) - I) = 4 * (# non-linear irrep constituents).")
    print("  Faithfulness forces at least one (the only normal subgroups of S5")
    print("  are 1, A5, S5, so a rep with no non-linear part kills A5). Hence 4")
    print("  is the floor, in ANY dimension -- and the regular rep pays 96.")
    print()
    print("  => the model's 32-dim head cannot buy a cheaper 5-cycle by picking")
    print("     a smarter embedding. Widening the state does not help.")
    print()
    return {"reps": rows, "all_multiple_of_4": all_mult4, "min_rank": min_rank}


# ---------------------------------------------------------------------------
# E5 -- how good is the BEST approximation below L?
# ---------------------------------------------------------------------------

def E5(args) -> dict:
    """How good is the BEST approximation strictly below L?

    E1/E2 settle that K < L cannot be exact. But 'not exact' is not the same as
    'useless': if the best order-3 map were within a hair of a 5-cycle, a trained
    network at n_h=3 would look like it works and then decay with length, which
    is exactly the fixed-3 behaviour we are trying to explain. So measure the
    floor.

    K >= L is not optimized here -- E2 already constructs those exactly, and
    reporting an optimizer's 1e-5 floor next to a true 0 would be misleading.
    """
    print("=" * 78)
    print("E5  best approximation strictly below L  (the fixed n_h=3 question)")
    print("=" * 78)

    out = {}
    for L_target, label in ((4, "5-cycle"), (3, "4-cycle")):
        targets = [p for p, e in zip(ELEMENTS, ELL) if e == L_target]
        p0 = targets[0]
        P = perm_matrix(p0)
        denom = P.norm().item()
        rows = []
        print(f"  target: {label}, L={L_target}, ||P||_F = {denom:.4f}")
        print(f"    {'K':>3} {'residual':>12} {'relative':>10} {'source':>14}")
        for K in range(0, N):
            if K >= L_target:
                c = construct(p0, K, pinned=False)
                err = (build(*c) - P).norm().item()
                src = "construction"
            else:
                r = fit(P, K, "box", restarts=max(args.restarts, 16),
                        adam_steps=args.adam, lbfgs_steps=args.lbfgs,
                        seed=900 + 31 * L_target + K)
                err = r["residual"]
                src = "optimized"
            rows.append(dict(K=K, residual=err, relative=err / denom, source=src))
            print(f"    {K:>3} {err:>12.3e} {err / denom:>10.3f} {src:>14}")
        out[label] = rows
        print()

    k3 = next(r for r in out["5-cycle"] if r["K"] == 3)
    print(f"  The best a 3-factor layer can do against a 5-cycle is "
          f"{k3['relative']*100:.0f}% relative error.")
    print(f"  That is not a near miss. And it compounds: the task is a RUNNING")
    print(f"  product, so a per-token error of that size cannot survive 128")
    print(f"  steps. Whatever fixed n_h=3 is doing to reach 0.54 sequence")
    print(f"  accuracy, it is NOT approximating each 5-cycle with its three")
    print(f"  Householder factors.")
    print()
    print(f"  Which leaves three candidates, none of which this file can settle:")
    print(f"    (a) the additive v k^T channel -- DeltaNet's state update is")
    print(f"        S_t = A_t S_{{t-1}} + v_t k_t^T, and only the A_t half is")
    print(f"        constrained by any of the above;")
    print(f"    (b) 12 heads coding the group jointly, so no single head ever")
    print(f"        represents the permutation;")
    print(f"    (c) it is not tracking at all, just getting most tokens right.")
    print(f"        Tempting -- 0.95 per hard token over ~12.8 hard tokens is")
    print(f"        0.95^12.8 = 0.52, close to the observed 0.54 -- but that")
    print(f"        same model predicts token accuracy 0.746, and the observed")
    print(f"        value is 0.948. So (c) does not fit either, on its own.")
    print(f"  Neither naive failure model reproduces BOTH numbers, which means")
    print(f"  the errors are clustered in some structured way. Measure that")
    print(f"  directly: work/exp_error_forensics.py")
    print()
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--adam", type=int, default=600)
    ap.add_argument("--lbfgs", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="results/beta_reachability.json")
    args = ap.parse_args()
    if args.quick:
        args.restarts, args.adam, args.lbfgs = 3, 200, 60

    torch.manual_seed(0)
    t0 = time.perf_counter()
    print()
    print("#" * 78)
    print("#  FREE-BETA REACHABILITY   (float64, CPU)")
    print("#" * 78)
    print()

    stages = {"E1": E1, "E2": E2, "E4": E4, "E5": E5}
    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(stages)
    out = {}
    for name in want:
        if name not in stages:
            raise SystemExit(f"unknown stage {name}; choose from {list(stages)}")
        out[name] = stages[name](args)

    dt = time.perf_counter() - t0
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if "E1" in out and "E2" in out:
        print("  1. K >= L(g) is forced by rank, not by parity or orthogonality.")
        print("     No choice of beta beats it.")
        print("  2. Pinning beta = 2 imposes an EXTRA constraint, K = L (mod 2).")
        print("     Free beta removes that one. Our gate needs free beta; that")
        print("     is now proved, not assumed.")
    if "E4" in out:
        print("  3. No faithful representation of S5 makes a 5-cycle cheaper")
        print("     than 4. Width does not help.")
    if "E5" in out:
        print("  4. The best order-3 approximation to a 5-cycle is far from")
        print("     exact, so fixed n_h=3's partial success needs a different")
        print("     explanation -- the additive v k^T channel, or multi-head")
        print("     distributed coding. That is the next thing to measure.")
    print(f"\n  elapsed {dt:.0f}s")
    print("=" * 78)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"written: {p}\n")


if __name__ == "__main__":
    main()
