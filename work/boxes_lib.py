"""Shared library for the Boxes experiments.

Verbatim copy of the code cells (sections 1-7) of
experiments/hybrid_delta_attn.ipynb, branch nlp_experiment, commit abd1349,
written by Shashwat Chaturvedi. Nothing below this docstring is edited, so the
notebook's self-test (selftest()) still certifies it. The DeltaProduct here is a
pure-PyTorch float32 implementation (no decay gate, beta in (0, 2)), not the
flash-linear-attention kernel used by the group-task experiments.

The order sweep that uses it is work/exp_boxes_order.py.
"""

import argparse
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:  # a GPU that this PyTorch build has no kernels for (e.g. P100) fails here, not 20 minutes in
        (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
        return "cuda"
    except Exception as e:  # noqa: BLE001
        print(f"CUDA is present but unusable with this PyTorch build ({e}).\nChoose the T4 accelerator. Falling back to CPU.")
        return "cpu"


DEVICE = pick_device()
print(f"torch {torch.__version__} | device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
if DEVICE == "cpu":
    print("NO GPU: everything works but is many times slower. Kaggle: Settings -> Accelerator -> GPU T4.")

# =============================================================================
# 1. Delta rule and DeltaProduct, pure PyTorch
# =============================================================================
#
#   S_t = (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T ,   o_t = S_t^T q_t
#
# DeltaProduct applies n_h of these updates per token (one Householder-like
# factor each) and reads out after the last. That is the same as running the
# plain delta rule on a sequence n_h times longer and keeping every n_h-th output,
# which is how it is implemented here.


def delta_rule_sequential(q, k, v, beta):
    """Reference: one token at a time. q,k: (B,H,T,dk)  v: (B,H,T,dv)  beta: (B,H,T)."""
    B, H, T, dk = q.shape
    S = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None, None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S - bt * torch.einsum("bhk,bhv->bhkv", kt, pred) + bt * torch.einsum("bhk,bhv->bhkv", kt, vt)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, 2)


def delta_rule_chunked(q, k, v, beta, chunk=32):
    """Same maths, chunk-parallel. Inside a chunk, with S0 the state entering it,
    S_{t-1} = S0 + sum_{j<t} k_j u_j^T and u_t = b_t (v_t - S_{t-1}^T k_t), so
    (I + tril(diag(b) K K^T, -1)) U = diag(b) (V - K S0): one triangular solve."""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:  # beta = 0 and k = 0 make the padded steps identity; outputs are cut off
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta = F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k = q.reshape(B, H, n, chunk, dk), k.reshape(B, H, n, chunk, dk)
    v, beta = v.reshape(B, H, n, chunk, dv), beta.reshape(B, H, n, chunk, 1)
    kb, vb = k * beta, v * beta
    A = torch.tril(kb @ k.transpose(-1, -2), -1)
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    WU = torch.linalg.solve_triangular(eye + A, torch.cat([kb, vb], -1), upper=False)
    W, U0 = WU[..., :dk], WU[..., dk:]
    QK = torch.tril(q @ k.transpose(-1, -2))  # includes the diagonal: o_t sees S_t
    S = q.new_zeros(B, H, dk, dv)
    outs = []
    for i in range(n):
        u = U0[:, :, i] - W[:, :, i] @ S
        outs.append(q[:, :, i] @ S + QK[:, :, i] @ u)
        S = S + k[:, :, i].transpose(-1, -2) @ u
    return torch.stack(outs, 2).reshape(B, H, n * chunk, dv)[:, :, :T]


def deltaproduct(q, k, v, beta, chunk=32):
    """q: (B,H,T,dk)  k: (B,H,T,n_h,dk)  v: (B,H,T,n_h,dv)  beta: (B,H,T,n_h)."""
    B, H, T, n_h, dk = k.shape
    dv = v.shape[-1]
    q_e = F.pad(q.unsqueeze(3), (0, 0, n_h - 1, 0)).reshape(B, H, T * n_h, dk)  # q only in the last slot
    o = delta_rule_chunked(
        q_e, k.reshape(B, H, T * n_h, dk), v.reshape(B, H, T * n_h, dv), beta.reshape(B, H, T * n_h), chunk
    )
    return o[:, :, n_h - 1 :: n_h]


def deltaproduct_reference(q, k, v, beta):
    """Independent loop-per-factor implementation, used only to test `deltaproduct`."""
    B, H, T, n_h, dk = k.shape
    S = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        for i in range(n_h):
            kt, vt, bt = k[:, :, t, i], v[:, :, t, i], beta[:, :, t, i, None, None]
            pred = torch.einsum("bhk,bhkv->bhv", kt, S)
            S = S - bt * torch.einsum("bhk,bhv->bhkv", kt, pred) + bt * torch.einsum("bhk,bhv->bhkv", kt, vt)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, 2)


# =============================================================================
# 2. Model
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, *shape, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(*shape)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class ShortConv(nn.Module):
    """Causal depthwise conv + SiLU over time. (B,T,C) -> (B,T,C)."""

    def __init__(self, ch, k=4):
        super().__init__()
        self.k, self.conv = k, nn.Conv1d(ch, ch, k, groups=ch, bias=False)

    def forward(self, x):
        return F.silu(self.conv(F.pad(x.transpose(1, 2), (self.k - 1, 0)))).transpose(1, 2)


def rope(x, base=10000.0):
    """x: (B,H,T,dh). Rotary embedding with positions 0..T-1."""
    T, half = x.shape[2], x.shape[-1] // 2
    inv = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class DeltaProductBranch(nn.Module):
    def __init__(self, d, H, dh, n_h, chunk, conv_k):
        super().__init__()
        self.H, self.dh, self.n_h, self.chunk = H, dh, n_h, chunk
        self.q, self.k = nn.Linear(d, H * dh, bias=False), nn.Linear(d, n_h * H * dh, bias=False)
        self.v, self.b = nn.Linear(d, n_h * H * dh, bias=False), nn.Linear(d, n_h * H, bias=False)
        self.cq, self.ck, self.cv = ShortConv(H * dh, conv_k), ShortConv(n_h * H * dh, conv_k), ShortConv(n_h * H * dh, conv_k)

    def forward(self, x):
        B, T, _ = x.shape
        H, dh, n_h = self.H, self.dh, self.n_h
        q = self.cq(self.q(x)).reshape(B, T, H, dh).transpose(1, 2)
        k = self.ck(self.k(x)).reshape(B, T, n_h, H, dh).permute(0, 3, 1, 2, 4)
        v = self.cv(self.v(x)).reshape(B, T, n_h, H, dh).permute(0, 3, 1, 2, 4)
        beta = 2 * torch.sigmoid(self.b(x)).reshape(B, T, n_h, H).permute(0, 3, 1, 2)  # (0,2): reflections reachable
        o = deltaproduct(F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, beta, self.chunk)
        return o.transpose(1, 2)  # (B,T,H,dh)


class CausalAttention(nn.Module):
    def __init__(self, d, H, dh, conv_k, causal=True):
        super().__init__()
        self.H, self.dh, self.causal = H, dh, causal  # causal=False exists ONLY as a test-the-test control
        self.q, self.k, self.v = (nn.Linear(d, H * dh, bias=False) for _ in range(3))
        self.cq, self.ck, self.cv = (ShortConv(H * dh, conv_k) for _ in range(3))

    def forward(self, x):
        B, T, _ = x.shape

        def split(t):
            return t.reshape(B, T, self.H, self.dh).transpose(1, 2)

        q, k, v = rope(split(self.cq(self.q(x)))), rope(split(self.ck(self.k(x)))), split(self.cv(self.v(x)))
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal).transpose(1, 2)  # (B,T,H,dh)


HYBRID_MODES = ("scalar", "head", "channel", "concat", "token")


class Mixer(nn.Module):
    def __init__(self, d, H, dh, variant, mode, n_h, chunk, conv_k, attn_causal=True):
        super().__init__()
        self.variant, self.mode = variant, mode
        self.record, self.last_gate = False, None
        if variant in ("delta", "hybrid"):
            self.delta, self.nd = DeltaProductBranch(d, H, dh, n_h, chunk, conv_k), RMSNorm(H, dh)
        if variant in ("attn", "hybrid"):
            self.attn, self.na = CausalAttention(d, H, dh, conv_k, attn_causal), RMSNorm(H, dh)
        width = H * dh
        if variant == "hybrid":
            assert mode in HYBRID_MODES, mode
            if mode == "concat":
                width = 2 * H * dh
            elif mode == "token":
                self.gate = nn.Linear(d, H)
                nn.init.normal_(self.gate.weight, std=0.02)
                nn.init.zeros_(self.gate.bias)
            else:  # alpha = sigmoid(a) = 0.5 at init: neither branch favoured
                self.alpha = nn.Parameter(torch.zeros({"scalar": (), "head": (H, 1), "channel": (H, dh)}[mode]))
        self.out = nn.Linear(width, d, bias=False)

    def forward(self, x):
        if self.variant == "delta":
            return self.out(self.nd(self.delta(x)).flatten(2))
        if self.variant == "attn":
            return self.out(self.na(self.attn(x)).flatten(2))
        od, oa = self.nd(self.delta(x)), self.na(self.attn(x))  # normalise first: scales differ
        if self.mode == "concat":
            o = torch.cat([od.flatten(2), oa.flatten(2)], -1)
        else:
            if self.mode == "token":
                g = torch.sigmoid(self.gate(x))  # (B,T,H)
                if self.record:
                    self.last_gate = g.detach()
                a = g.unsqueeze(-1)
            else:
                a = torch.sigmoid(self.alpha)
            o = (a * od + (1 - a) * oa).flatten(2)
        return self.out(o)


class Block(nn.Module):
    def __init__(self, d, H, dh, variant, mode, n_h, d_ff, chunk, conv_k, attn_causal):
        super().__init__()
        self.n1, self.mix = RMSNorm(d), Mixer(d, H, dh, variant, mode, n_h, chunk, conv_k, attn_causal)
        self.mlp = None
        if d_ff:
            self.n2 = RMSNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.SiLU(), nn.Linear(d_ff, d, bias=False))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x)) if self.mlp is not None else x


class LM(nn.Module):
    def __init__(self, vocab, d, layers, H, dh, variant, mode, n_h, d_ff, chunk=32, conv_k=4, attn_causal=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d, H, dh, variant, mode, n_h, d_ff, chunk, conv_k, attn_causal) for _ in range(layers)
        )
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    def gate_mixers(self):
        return [b.mix for b in self.blocks if b.mix.variant == "hybrid" and b.mix.mode == "token"]

    def alphas(self):
        return [torch.sigmoid(b.mix.alpha).tolist() for b in self.blocks if hasattr(b.mix, "alpha")]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(vocab, a, variant, mode, d_ff, attn_causal=True):
    return LM(vocab, a.d, a.layers, a.heads, a.head_dim, variant, mode, a.n_h, d_ff, a.chunk, a.conv_k, attn_causal)


def match_d_ff(vocab, a, variant, mode, target):
    """MLP width that brings this model's parameter count closest to `target`."""
    p1, p2 = n_params(build(vocab, a, variant, mode, 1)), n_params(build(vocab, a, variant, mode, 2))
    slope = p2 - p1
    return max(1, round((target - (p1 - slope)) / slope))


def parse_model(name, default_mode):
    if name in ("delta", "attn"):
        return name, None
    if name == "hybrid":
        return "hybrid", default_mode
    if name.startswith("hybrid:") and name.split(":")[1] in HYBRID_MODES:
        return "hybrid", name.split(":")[1]
    raise ValueError(f"unknown model {name!r}; use delta | attn | hybrid | hybrid:{'|'.join(HYBRID_MODES)}")


def model_name(variant, mode):
    return variant if mode is None else f"{variant}:{mode}"


# =============================================================================
# 3. Tasks.  Every task yields Batch(x, y, mask, kinds, group):
#      mask   positions that are scored / trained on
#      kinds  what each INPUT token is (-1 = padding); used by the gate diagnostics
#      group  which evaluation condition a position/row belongs to
# =============================================================================

Batch = namedtuple("Batch", "x y mask kinds group")


class PermGroupTask:
    """Running product in S_n. Input g_t, target p_t = g_t o p_{t-1}. Pure state tracking."""

    metric = "token"
    kind_names = ["token"]

    def __init__(self, group="S3", train_len=128, eval_len=512, n_eval=1000):
        n = int(group[1:])
        self.perms = list(itertools.permutations(range(n)))
        idx = {p: i for i, p in enumerate(self.perms)}
        self.table = torch.tensor([[idx[tuple(a[b[i]] for i in range(n))] for b in self.perms] for a in self.perms])
        self.identity = idx[tuple(range(n))]
        self.vocab_size, self.train_len, self.eval_len, self.n_eval = len(self.perms), train_len, eval_len, n_eval
        edges = list(range(0, eval_len, train_len)) + [eval_len]
        self.group_names = [f"pos[{a}:{b})" for a, b in zip(edges[:-1], edges[1:])]
        self.train_len_desc = f"{group}, train length {train_len}, eval length {eval_len}"

    def _make(self, bs, T, gen):
        g = torch.randint(0, self.vocab_size, (bs, T), generator=gen)
        p, ys = torch.full((bs,), self.identity), torch.empty_like(g)
        for t in range(T):
            p = self.table[g[:, t], p]
            ys[:, t] = p
        return g, ys

    def train_batch(self, bs, gen):
        x, y = self._make(bs, self.train_len, gen)
        return Batch(x, y, torch.ones_like(x, dtype=torch.bool), torch.zeros_like(x), torch.zeros_like(x))

    def eval_batches(self, n=None, bs=100):
        n = n or self.n_eval
        gen = torch.Generator().manual_seed(1234)  # same eval set for every model and seed
        for _ in range(math.ceil(n / bs)):
            x, y = self._make(min(bs, n), self.eval_len, gen)
            grp = (torch.arange(self.eval_len) // self.train_len).clamp(max=len(self.group_names) - 1)
            yield Batch(x, y, torch.ones_like(x, dtype=torch.bool), torch.zeros_like(x), grp.expand_as(x).contiguous())


class RecallTask:
    """Multi-query associative recall: k1 v1 ... kN vN  q1 ... qN  -> v(q_i) at each query. Pure recall."""

    metric = "token"
    kind_names = ["key", "value", "query"]

    def __init__(self, n_keys=128, n_vals=128, train_pairs=16, eval_pairs=(16, 32, 64), n_eval=2000):
        assert n_keys >= max(eval_pairs) and n_keys >= train_pairs
        self.n_keys, self.n_vals, self.train_pairs, self.eval_pairs, self.n_eval = n_keys, n_vals, train_pairs, eval_pairs, n_eval
        self.vocab_size = n_keys + n_vals
        self.group_names = [f"N={n}" for n in eval_pairs]
        self.train_len_desc = f"train {train_pairs} pairs, eval {list(eval_pairs)} pairs"

    def _make(self, bs, N, gen, grp=0):
        keys = torch.rand(bs, self.n_keys, generator=gen).argsort(1)[:, :N]  # distinct within a row
        vals = torch.randint(0, self.n_vals, (bs, N), generator=gen) + self.n_keys
        order = torch.rand(bs, N, generator=gen).argsort(1)
        qk, qv = keys.gather(1, order), vals.gather(1, order)
        x = torch.cat([torch.stack([keys, vals], 2).reshape(bs, 2 * N), qk], 1)
        y, mask = torch.zeros_like(x), torch.zeros_like(x, dtype=torch.bool)
        y[:, 2 * N :], mask[:, 2 * N :] = qv, True
        kinds = torch.cat([torch.arange(2 * N) % 2, torch.full((N,), 2)]).expand_as(x).contiguous()
        return Batch(x, y, mask, kinds, torch.full_like(x, grp))

    def train_batch(self, bs, gen):
        return self._make(bs, self.train_pairs, gen)

    def eval_batches(self, n=None, bs=250):
        n = n or self.n_eval
        gen = torch.Generator().manual_seed(1234)
        for gi, N in enumerate(self.eval_pairs):
            for _ in range(math.ceil(n / bs)):
                yield self._make(min(bs, n), N, gen, gi)


BOXES_URL = "https://github.com/sebschu/entity-tracking-lms/raw/main/data/boxes-dataset-v1.zip"
BOXES_PWD = b"iamnotaLM"  # published by the dataset authors in their README
BOXES_SPLIT = "t5_boxes_nso_exp2_max3"
BOXES_MAX_BUCKET = 5  # buckets ops=0..4 and ops>=5 (ops 6-8 are too rare to stand alone)


def _tok(text):
    return re.findall(r"[A-Za-z0-9]+|[.,]", text)


def _boxes_locate(root, split_file):
    for dp, _, fs in os.walk(root):
        if os.path.basename(dp) == BOXES_SPLIT and split_file in fs:
            return os.path.join(dp, split_file)
    return None


def _boxes_fetch(data_dir, zip_path=None):
    """Return paths of train/test jsonl, extracting (and if needed downloading) only what is needed."""
    files = {s: _boxes_locate(data_dir, f"{s}-t5.jsonl") for s in ("train", "test")}
    if all(files.values()):
        return files
    os.makedirs(data_dir, exist_ok=True)
    zp = zip_path or os.path.join(data_dir, "boxes-dataset-v1.zip")
    if not os.path.exists(zp):
        print(f"downloading Boxes dataset (28 MB) -> {zp}", flush=True)
        try:
            urllib.request.urlretrieve(BOXES_URL, zp)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"download failed ({e}). On Kaggle turn Internet ON, or add boxes-dataset-v1.zip as a "
                             f"Dataset and pass --boxes_zip /kaggle/input/<name>/boxes-dataset-v1.zip")
    with zipfile.ZipFile(zp) as z:
        for s in ("train", "test"):
            member = f"boxes-dataset-v1/{BOXES_SPLIT}/{s}-t5.jsonl"
            print(f"extracting {member}", flush=True)
            z.extract(member, data_dir, pwd=BOXES_PWD)
    return {s: _boxes_locate(data_dir, f"{s}-t5.jsonl") for s in ("train", "test")}


class BoxesTask:
    """Kim & Schuster (2023). Decoder-only framing of their masked-answer format:
    text = <description> <operations> Box k   ->  answer  'contains the a and the b' | 'is empty'  then '.'
    Loss/score on the answer tokens only. Exact match under teacher forcing equals greedy-decoding exact
    match (if every argmax is right, greedy follows the same path)."""

    metric = "exact"
    kind_names = ["description", "operation", "query", "answer"]
    PAD, UNK = 0, 1

    def __init__(self, data_dir, zip_path=None, max_train_ops=2, train_n=None, eval_per_bucket=2000, seed=1234):
        paths = _boxes_fetch(data_dir, zip_path)
        rng = random.Random(seed)
        self.vocab = {"<pad>": 0, "<unk>": 1}
        train_rows = [r for r in _read_jsonl(paths["train"]) if r["numops"] <= max_train_ops]
        if train_n and train_n < len(train_rows):
            train_rows = rng.sample(train_rows, train_n)
        self.train = [self.encode(r, grow=True) for r in train_rows]
        del train_rows
        self.dot = self.vocab["."]
        by_bucket = defaultdict(list)
        for r in _read_jsonl(paths["test"]):
            by_bucket[min(r["numops"], BOXES_MAX_BUCKET)].append(r)
        self.eval_sets, self.n_unk = {}, 0
        for b in sorted(by_bucket):
            rows = by_bucket[b] if len(by_bucket[b]) <= eval_per_bucket else rng.sample(by_bucket[b], eval_per_bucket)
            self.eval_sets[b] = [self.encode(r, grow=False) for r in rows]
            self.n_unk += sum(int((e[0] == self.UNK).sum()) for e in self.eval_sets[b])
        self.vocab_size = len(self.vocab)
        names = [f"ops={b}" if b < BOXES_MAX_BUCKET else f"ops>={b}" for b in sorted(by_bucket)]
        self.group_names = names
        self.bucket_of_group = sorted(by_bucket)
        self.train_len_desc = f"train on ops<={max_train_ops} ({len(self.train)} ex), eval on ops 0..8 (ops>{max_train_ops} = extrapolation)"
        print(f"boxes: vocab {self.vocab_size}, train {len(self.train)}, eval {[len(v) for v in self.eval_sets.values()]}, "
              f"UNK tokens in eval: {self.n_unk}", flush=True)

    def encode(self, row, grow):
        prompt = _tok(row["sentence_masked"].split("<extra_id_0>")[0])
        answer = _tok(row["masked_content"].replace("<extra_id_0>", "")) + ["."]
        ids = np.array([self._id(w, grow) for w in prompt + answer], dtype=np.int32)
        return ids, len(prompt), min(row["numops"], BOXES_MAX_BUCKET)

    def _id(self, w, grow):
        if grow:
            return self.vocab.setdefault(w, len(self.vocab))
        return self.vocab.get(w, self.UNK)

    def _collate(self, exs, group_of):
        L = max(len(e[0]) for e in exs) - 1
        B = len(exs)
        x, y = torch.zeros(B, L, dtype=torch.long), torch.zeros(B, L, dtype=torch.long)
        mask, kinds, grp = torch.zeros(B, L, dtype=torch.bool), torch.full((B, L), -1), torch.zeros(B, L, dtype=torch.long)
        for i, (ids, P, bucket) in enumerate(exs):
            n = len(ids) - 1
            t = torch.from_numpy(ids).long()
            x[i, :n], y[i, :n] = t[:-1], t[1:]
            mask[i, P - 1 : n] = True  # y[P-1] = first answer token
            dots = np.flatnonzero(ids[:P] == self.dot)
            k = np.empty(len(ids), dtype=np.int64)
            k[: dots[0] + 1], k[dots[0] + 1 : dots[-1] + 1], k[dots[-1] + 1 : P], k[P:] = 0, 1, 2, 3
            kinds[i, :n] = torch.from_numpy(k[:-1])
            grp[i, :] = group_of(bucket)
        return Batch(x, y, mask, kinds, grp)

    def train_batch(self, bs, gen):
        idx = torch.randint(0, len(self.train), (bs,), generator=gen).tolist()
        return self._collate([self.train[i] for i in idx], lambda b: 0)

    def eval_batches(self, n=None, bs=128):
        for gi, b in enumerate(self.bucket_of_group):
            exs = self.eval_sets[b][: n or len(self.eval_sets[b])]
            for i in range(0, len(exs), bs):
                yield self._collate(exs[i : i + bs], lambda _b, gi=gi: gi)

    def decode(self, ids):
        inv = {i: w for w, i in self.vocab.items()}
        return " ".join(inv[int(i)] for i in ids)


def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


# =============================================================================
# 4. Scoring, gate diagnostics, training
# =============================================================================


def score(logits, b, metric, G):
    """-> (right[G], total[G]). 'token': per scored position. 'exact': per row, all scored positions right."""
    ok = logits.argmax(-1) == b.y
    right, total = torch.zeros(G), torch.zeros(G)
    if metric == "token":
        for g in range(G):
            sel = b.mask & (b.group == g)
            right[g], total[g] = (ok & sel).sum().item(), sel.sum().item()
    else:
        row_ok = (ok | ~b.mask).all(1)
        row_g = b.group.max(1).values
        for g in range(G):
            sel = row_g == g
            right[g], total[g] = (row_ok & sel).sum().item(), sel.sum().item()
    return right, total


def eta2(g, labels):
    """Fraction of the variance of g (N,H) explained by a categorical label (N,). Mean over heads."""
    g = g.double()
    N = g.shape[0]
    C = int(labels.max()) + 1
    cnt = torch.bincount(labels, minlength=C).double()
    means = torch.zeros(C, g.shape[1], dtype=torch.double).index_add_(0, labels, g) / cnt.clamp(min=1)[:, None]
    between = (cnt[:, None] * (means - g.mean(0)) ** 2).sum(0) / N
    total = g.var(0, unbiased=False)
    return torch.where(total > 1e-12, between / total.clamp(min=1e-12), torch.zeros_like(total)).mean().item()


class GateStats:
    """Where does the soft router lean on each branch? g = weight on DeltaProduct (1-g on attention)."""

    N_POS = 5

    def __init__(self, kind_names):
        self.kind_names, self.buf = kind_names, defaultdict(list)

    def add(self, mixers, b):
        valid = b.kinds >= 0
        rows = valid.sum(1, keepdim=True).clamp(min=1)
        pos = torch.arange(b.x.shape[1])[None].expand_as(b.x)
        pb = ((pos.float() / rows.float().cpu()) * self.N_POS).long().clamp(max=self.N_POS - 1)
        for li, m in enumerate(mixers):
            self.buf[li].append((m.last_gate[valid.to(m.last_gate.device)].cpu(), b.kinds[valid], b.x[valid], pb[valid]))

    def summary(self):
        out = {}
        for li, parts in self.buf.items():
            g, kinds, toks, pb = (torch.cat(t) for t in zip(*parts))
            by_kind = {self.kind_names[k]: g[kinds == k].mean().item() for k in range(len(self.kind_names)) if (kinds == k).any()}
            by_pos = [g[pb == p].mean().item() if (pb == p).any() else None for p in range(self.N_POS)]
            e = {"kind": eta2(g, kinds), "token": eta2(g, toks), "pos": eta2(g, pb)}
            out[f"layer{li}"] = {"mean": g.mean().item(), "std": g.std().item(), "by_kind": by_kind, "by_pos": by_pos,
                                 "eta2": e, "label": gate_label(g.std().item(), e)}
        return out


def gate_label(std, e):
    """Token-id eta2 is deliberately NOT used: layer 0 sees only the token embedding, so its gate is a function of
    the token id by construction (eta2 = 1) and would make every run look 'structured'."""
    if std < 0.05:
        return "near-constant"
    if max(e["kind"], e.get("pos", 0.0)) > 0.25:
        return "structured"
    return "varies, but not by kind or position"


@torch.no_grad()
def evaluate(model, task, device, n=None, record=False):
    model.eval()
    G = len(task.group_names)
    right, total = torch.zeros(G), torch.zeros(G)
    mixers = model.gate_mixers() if record else []
    gs = GateStats(task.kind_names) if mixers else None
    for m in mixers:
        m.record = True
    for b in task.eval_batches(n):
        bd = Batch(*[t.to(device) for t in b])
        r, t = score(model(bd.x), bd, task.metric, G)
        right, total = right + r, total + t
        if gs:
            gs.add(mixers, b)
    for m in mixers:
        m.record = False
    acc = {name: (right[i] / total[i]).item() if total[i] > 0 else float("nan") for i, name in enumerate(task.group_names)}
    model.train()
    return acc, (gs.summary() if gs else None)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_one(a, task, variant, mode, d_ff, seed, device, steps, bs):
    set_seed(seed)
    model = build(task.vocab_size, a, variant, mode, d_ff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)

    def lr_at(s):  # constant after warmup by default: a decaying LR can suppress a late plateau escape
        w = min(1.0, (s + 1) / a.warmup)
        return w * (0.5 * (1 + math.cos(math.pi * s / steps)) if a.lr_schedule == "cosine" else 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator().manual_seed(seed * 7919 + 1)
    t0, curve_loss, curve_eval, run, truncated, diverged = time.time(), [], [], [], False, False
    name = model_name(variant, mode)
    for step in range(1, steps + 1):
        b = Batch(*[t.to(device) for t in task.train_batch(bs, gen)])
        loss = F.cross_entropy(model(b.x)[b.mask], b.y[b.mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if not math.isfinite(loss.item()):
            diverged = True
            print(f"  [{name} seed{seed}] loss not finite at step {step}; stopping this run", flush=True)
            break
        run.append(loss.item())
        if step % a.log_every == 0:
            curve_loss.append((step, sum(run) / len(run)))
            print(f"  [{name} seed{seed}] step {step:6d}  loss {curve_loss[-1][1]:.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
            run = []
        if a.eval_every and step % a.eval_every == 0:
            acc, _ = evaluate(model, task, device, n=a.quick_n)
            curve_eval.append((step, acc))
            print(f"  [{name} seed{seed}]   quick-eval " + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()), flush=True)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            truncated = True
            print(f"  [{name} seed{seed}] stopped at step {step}: --max_minutes reached (result marked truncated)", flush=True)
            break
    final, gate = evaluate(model, task, device, record=True)
    return {"model": name, "seed": seed, "params": n_params(model), "d_ff": d_ff, "steps": step, "truncated": truncated,
            "diverged": diverged, "minutes": (time.time() - t0) / 60, "final": final, "curve_loss": curve_loss,
            "curve_eval": curve_eval, "gate": gate, "alpha": model.alphas() or None}


# =============================================================================
# 5. Summary and the Stage-1 verdict
# =============================================================================


def median(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def verdict(med, hybrid, margin):
    """med: {model: {condition: median metric}}. Hybrid vs the BEST single branch, condition by condition.
    Four outcomes, in this order of precedence:
      INTERFERES     hybrid is worse than the best single branch somewhere
      SYNERGY        hybrid beats the best single branch somewhere (better than either alone)
      COMPLEMENTARY  the two branches have different decisive winners on different conditions, and the hybrid
                     matches the best of them on every one: one model is good at both. This is the plan's success case.
      NO GAIN        one branch already covers every condition, so mixing adds nothing"""
    notes, worse, better, winners = [], [], [], {}
    for c in med[hybrid]:
        singles = {m: med[m][c] for m in ("delta", "attn") if m in med and med[m].get(c) == med[m].get(c)}
        if not singles:
            continue
        best_m = max(singles, key=singles.get)
        diff = med[hybrid][c] - singles[best_m]
        sym = ">" if diff > margin else "<" if diff < -margin else "~"
        notes.append(f"{c}: hybrid {med[hybrid][c]:.3f} {sym} best single ({best_m} {singles[best_m]:.3f})")
        (better if sym == ">" else worse if sym == "<" else []).append(c)
        if len(singles) == 2 and abs(singles["delta"] - singles["attn"]) > margin:
            winners[c] = best_m
    if max(v for m in med for v in med[m].values() if v == v) < 0.10:
        return "UNINFORMATIVE: no model learned anything above 0.10", notes
    if worse:
        return f"INTERFERES: hybrid is worse than the best single branch on {worse}", notes
    if better:
        return f"SYNERGY: hybrid beats the best single branch on {better} and is never worse", notes
    if len(set(winners.values())) >= 2:
        wins = {m: [c for c, w in winners.items() if w == m] for m in set(winners.values())}
        return f"COMPLEMENTARY: {wins}; hybrid matches the best branch on every condition", notes
    return "NO GAIN: one branch already covers every condition, so mixing adds nothing here", notes


def fmt_seeds(vals):
    return " ".join(f"{v:.3f}" for v in sorted(vals))


def print_summary(results, tag, margin):
    runs = defaultdict(list)
    for r in results["runs"].values():
        if r["tag"] == tag:
            runs[r["model"]].append(r)
    if not runs:
        return
    models = sorted(runs, key=lambda m: (m.split(":")[0] != "delta", m.split(":")[0] != "attn", m))
    conds = list(next(iter(runs.values()))[0]["final"])
    print(f"\n{'=' * 78}\nSUMMARY  {tag}\n{'=' * 78}")
    for m in models:
        rs = runs[m]
        flags = [f"seed{r['seed']}:" + ("truncated" if r["truncated"] else "") + ("DIVERGED" if r["diverged"] else "") for r in rs if r["truncated"] or r["diverged"]]
        print(f"  {m:<16} {len(rs)} seeds  params {rs[0]['params']:,}  d_ff {rs[0]['d_ff']}  " + " ".join(flags))
        if len(rs) < 3:
            print("     ! fewer than 3 seeds: cannot tell a lucky run from a reliable one")
    print("\n  per-seed values, sorted, then median. Never read the median alone on a hard cell.\n")
    w = max(len(c) for c in conds) + 2
    for m in models:
        print(f"  {m}")
        for c in conds:
            vals = [r["final"][c] for r in runs[m]]
            print(f"     {c:<{w}} [{fmt_seeds(vals)}]   median {median(vals):.3f}")
    med = {m: {c: median([r["final"][c] for r in runs[m]]) for c in conds} for m in models}
    for h in [m for m in models if m.startswith("hybrid")]:
        if "delta" in med or "attn" in med:
            label, notes = verdict(med, h, margin)
            print(f"\n  VERDICT for {h} (margin {margin}; indicative, judge it against the per-seed spread above):\n     {label}")
            for n in notes:
                print(f"       {n}")
    for m in models:
        rs = [r for r in runs[m] if r.get("alpha")]
        if rs:
            print(f"\n  learned alpha (weight on DeltaProduct, 1-alpha on attention), {m}, per seed, per layer:")
            for r in rs:
                print(f"     seed{r['seed']}: " + " | ".join(json.dumps(np.round(np.array(a), 3).tolist()) for a in r["alpha"]))
    for m in [m for m in models if m.endswith(":token")]:
        print(f"\n  SOFT-ROUTER DIAGNOSTIC for {m}: g = weight on DeltaProduct. This is the evidence for whether a hard router is worth building.")
        print("     eta2 = share of g's variance explained by that label (0 = none, 1 = all).")
        for r in runs[m]:
            for layer, s in (r["gate"] or {}).items():
                bk = "  ".join(f"{k} {v:.2f}" for k, v in s["by_kind"].items())
                bp = " ".join("  -  " if v is None else f"{v:.2f}" for v in s["by_pos"])
                print(f"     seed{r['seed']} {layer}: mean {s['mean']:.3f} std {s['std']:.3f}  eta2 kind {s['eta2']['kind']:.2f} "
                      f"token {s['eta2']['token']:.2f} pos {s['eta2']['pos']:.2f}  -> {gate_label(s['std'], s['eta2'])}")
                print(f"           by kind: {bk}\n           by relative position (5 bins): {bp}")
        print("     thresholds are heuristics: std<0.05 = near-constant; eta2(kind or pos)>0.25 = structured.")
        print("     eta2 token is information only: layer 0 sees just the token embedding, so it is ~1 there by construction.")
    print("\n  Both branches always run: nothing above is a compute saving.\n")


# =============================================================================
# 6. Self-test: every check has a stated way to fail
# =============================================================================


def selftest(device="cpu"):
    """Runs on CPU always, and repeats the device-dependent checks on `device` (solve_triangular, fused attention and
    the training step are separate code paths on a GPU and cannot be assumed to match the CPU ones)."""
    ok_all, torch_dtype = True, torch.get_default_dtype()
    devices = list(dict.fromkeys(["cpu", device]))

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    torch.manual_seed(0)
    print("delta rule")
    B, H, T, dk, dv = 2, 3, 45, 8, 6  # T not a multiple of the chunk: exercises padding
    q = torch.randn(B, H, T, dk, dtype=torch.double)
    k = F.normalize(torch.randn(B, H, T, dk, dtype=torch.double), dim=-1)
    v = torch.randn(B, H, T, dv, dtype=torch.double)
    beta = torch.rand(B, H, T, dtype=torch.double) * 2
    ref, out = delta_rule_sequential(q, k, v, beta), delta_rule_chunked(q, k, v, beta, chunk=16)
    check("chunked == sequential, float64, T=45 chunk=16", (out - ref).abs().max() < 1e-9, f"max err {(out - ref).abs().max():.1e}")
    out32 = delta_rule_chunked(*(x.float() for x in (q, k, v, beta)), chunk=16)
    check("float32 chunked within 1e-4 of float64 reference", (out32.double() - ref).abs().max() < 1e-4, f"max err {(out32.double() - ref).abs().max():.1e}")
    k1 = F.normalize(torch.randn(1, 1, 1, dk, dtype=torch.double), dim=-1).expand(1, 1, 64, dk).contiguous()  # worst case: all keys equal, beta=2
    q1, v1, b1 = torch.randn(1, 1, 64, dk, dtype=torch.double), torch.randn(1, 1, 64, dv, dtype=torch.double), torch.full((1, 1, 64), 2.0, dtype=torch.double)
    r1, o1 = delta_rule_sequential(q1, k1, v1, b1), delta_rule_chunked(q1, k1, v1, b1, chunk=32)
    o1f = delta_rule_chunked(q1.float(), k1.float(), v1.float(), b1.float(), chunk=32)
    check("adversarial case (identical keys, beta=2) matches", (o1 - r1).abs().max() < 1e-9 and (o1f.double() - r1).abs().max() < 1e-3)
    gs = []
    for fn, kw in ((delta_rule_sequential, {}), (delta_rule_chunked, {"chunk": 16})):
        xs = [t.clone().requires_grad_() for t in (q, k, v, beta)]
        fn(*xs, **kw).pow(2).sum().backward()
        gs.append([x.grad for x in xs])
    check("gradients agree (q,k,v,beta)", all((a - b).abs().max() < 1e-8 for a, b in zip(*gs)))
    kk = F.normalize(torch.randn(B, H, T, 3, dk, dtype=torch.double), dim=-1)
    vv, bb = torch.randn(B, H, T, 3, dv, dtype=torch.double), torch.rand(B, H, T, 3, dtype=torch.double) * 2
    d1, d2 = deltaproduct(q, kk, vv, bb, chunk=16), deltaproduct_reference(q, kk, vv, bb)
    check("DeltaProduct (n_h=3) == explicit product of per-token factors", (d1 - d2).abs().max() < 1e-9, f"max err {(d1 - d2).abs().max():.1e}")
    R = torch.randn(1, 1, 10, 16)
    rr = rope(R)
    check("rope preserves norm", (rr.norm(dim=-1) - R.norm(dim=-1)).abs().max() < 1e-5)
    v0 = torch.randn(1, 1, 1, 16).expand(1, 1, 24, 16).contiguous()
    rv = rope(v0)[0, 0]
    sc = rv @ rv.T
    check("rope scores depend only on relative offset", abs(sc[3, 1] - sc[15, 13]) < 1e-4 and abs(sc[3, 1] - sc[3, 2]) > 1e-3)

    print("causality (every variant; plus a leaky control that the test MUST catch)")

    class A:  # tiny config
        d, layers, heads, head_dim, n_h, chunk, conv_k = 32, 2, 2, 16, 2, 8, 4

    x = torch.randint(0, 20, (2, 24))
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 20, (2, 12))
    for dev in devices:
        for var, mode in [("delta", None), ("attn", None)] + [("hybrid", m) for m in HYBRID_MODES]:
            m = build(20, A, var, mode, 64).to(dev).eval()
            with torch.no_grad():
                leak = (m(x.to(dev))[:, :12] - m(x2.to(dev))[:, :12]).abs().max().item()
            check(f"causal on {dev}: {model_name(var, mode)}", leak < 1e-5, f"leak {leak:.1e}")
        m = build(20, A, "attn", None, 64, attn_causal=False).to(dev).eval()
        with torch.no_grad():
            leak = (m(x.to(dev))[:, :12] - m(x2.to(dev))[:, :12]).abs().max().item()
        check(f"leak detector fires on non-causal attention on {dev} (control)", leak > 1e-4, f"leak {leak:.1e}")

    if device != "cpu":
        print(f"delta rule and training step on {device} (compared with the float64 CPU reference)")
        for chunk in (16, 32):
            og = delta_rule_chunked(*(x.float().to(device) for x in (q, k, v, beta)), chunk=chunk)
            check(f"{device}: float32 chunked (chunk={chunk}) within 1e-3 of the CPU float64 reference",
                  (og.double().cpu() - ref).abs().max() < 1e-3, f"max err {(og.double().cpu() - ref).abs().max():.1e}")
        og1 = delta_rule_chunked(*(x.float().to(device) for x in (q1, k1, v1, b1)), chunk=32)
        check(f"{device}: adversarial case", (og1.double().cpu() - r1).abs().max() < 1e-3)
        for var, mode in [("delta", None), ("attn", None), ("hybrid", "scalar"), ("hybrid", "token")]:
            m = build(20, A, var, mode, 64).to(device)
            F.cross_entropy(m(x.to(device)).reshape(-1, 20), x.to(device).reshape(-1)).backward()
            check(f"{device}: forward/backward finite: {model_name(var, mode)}",
                  all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    print("parameter matching")
    tgt = n_params(build(50, A, "hybrid", "scalar", 64))
    for var, mode in [("delta", None), ("attn", None), ("hybrid", "concat"), ("hybrid", "token")]:
        p = n_params(build(50, A, var, mode, match_d_ff(50, A, var, mode, tgt)))
        check(f"matched: {model_name(var, mode)}", abs(p - tgt) / tgt < 0.01, f"{p} vs {tgt}")

    print("tasks")
    t = PermGroupTask("S5")
    tb = t.table
    tri = torch.randint(0, 120, (3, 200))
    check("S5 has 120 elements; table associative", t.vocab_size == 120 and (tb[tb[tri[0], tri[1]], tri[2]] == tb[tri[0], tb[tri[1], tri[2]]]).all())
    check("S5 identity is neutral; every row is a permutation", (tb[t.identity] == torch.arange(120)).all() and all(sorted(r.tolist()) == list(range(120)) for r in tb))
    check("S5 is non-abelian (a table that is accidentally commutative would pass everything above)", (tb != tb.T).any())
    t3 = PermGroupTask("S3", train_len=20, eval_len=40)
    bt = t3.train_batch(4, torch.Generator().manual_seed(0))
    okp = True
    for row in range(4):
        p = tuple(range(3))
        for s in range(20):
            g = t3.perms[bt.x[row, s]]
            p = tuple(g[p[i]] for i in range(3))
            okp &= t3.perms[bt.y[row, s]] == p
    check("running product == independent tuple composition", okp)
    eb = next(iter(t3.eval_batches(4, bs=4)))
    check("S3 eval windows [0:20) [20:40)", t3.group_names == ["pos[0:20)", "pos[20:40)"] and eb.group[0, 19] == 0 and eb.group[0, 20] == 1)

    r = RecallTask(train_pairs=8, eval_pairs=(8, 16))
    bt = r.train_batch(16, torch.Generator().manual_seed(0))
    okr = True
    for row in range(16):
        keys, vals = bt.x[row, 0:16:2].tolist(), bt.x[row, 1:16:2].tolist()
        okr &= len(set(keys)) == 8 and all(kk < 128 for kk in keys) and all(vv >= 128 for vv in vals)
        for pos in range(16, 24):
            okr &= bt.y[row, pos].item() == vals[keys.index(bt.x[row, pos].item())]
    check("recall: keys distinct, queries are stored keys, target = stored value", okr)
    check("recall: only query positions scored", bt.mask[:, :16].sum() == 0 and bt.mask[:, 16:].all())

    print("Boxes pipeline on hand-written rows in the real file format")
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "boxes-dataset-v1", BOXES_SPLIT)
        os.makedirs(d)
        desc = ("Box 0 contains the branch and the cheese, Box 1 contains the egg, Box 2 contains the disk and the tie, "
                "Box 3 contains the book, Box 4 contains the bag, Box 5 contains the bell, Box 6 contains the bone.")

        def row(ops, box, ans, nops):
            body = f"{desc} {ops} Box {box} ." if ops else f"{desc} Box {box} ."
            return {"sentence": body.replace(" .", "."), "sentence_masked": body.replace(f"Box {box} .", f"Box {box} <extra_id_0> ."),
                    "masked_content": f"<extra_id_0> {ans}", "sample_id": 1, "numops": nops}

        rows = [row("", 0, "contains the branch and the cheese", 0),
                row("Remove the egg from Box 1. Put the bill into Box 6.", 1, "is empty", 2),
                row("Move the tie from Box 2 to Box 1. Remove the bone from Box 6. Put the cup into Box 4. Remove the bag from Box 4.", 1,
                    "contains the egg and the tie", 4)]
        train_rows = rows[:2] + [dict(rows[2], numops=2)]  # train must contain every word the test rows use
        for s, rs in (("train", train_rows), ("test", rows)):
            with open(os.path.join(d, f"{s}-t5.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(r_) for r_ in rs))
        bx = BoxesTask(td, max_train_ops=2, eval_per_bucket=10)
        unk_ids, _, _ = bx.encode(dict(rows[0], sentence_masked=rows[0]["sentence_masked"].replace("Box 0 contains", "Box 0 zebra")), grow=False)
        check("unseen word maps to <unk> (control for the UNK counter)", (unk_ids == bx.UNK).sum() == 1)
        b = bx._collate(bx.train, lambda _b: 0)
        dec = bx.decode(b.y[0][b.mask[0]])
        check("answer tokens are exactly the scored ones", dec == "contains the branch and the cheese .", repr(dec))
        check("'is empty' answer", bx.decode(b.y[1][b.mask[1]]) == "is empty .")
        prompt_ids = b.x[0][(b.kinds[0] >= 0) & (b.kinds[0] < 3)]
        check("prompt round-trips the description", bx.decode(prompt_ids).startswith("Box 0 contains the branch and the cheese , Box 1") and bx.decode(prompt_ids).endswith("Box 0"), bx.decode(prompt_ids)[-30:])
        k0 = b.kinds[0][b.kinds[0] >= 0].tolist()
        # description ends with '.', query is 'Box','0', then the 6 answer INPUT tokens (the final '.' is only ever a target)
        check("kinds for numops=0: ... 0 | 2 2 | 3 x6, no operation segment", k0[-9:] == [0, 2, 2, 3, 3, 3, 3, 3, 3] and 1 not in k0, str(k0[-9:]))
        k1 = b.kinds[1][b.kinds[1] >= 0].tolist()
        check("kinds for numops=2 contain an operation segment between description and query", 1 in k1 and k1.index(1) > k1.index(0) and k1.index(2) > max(i for i, v in enumerate(k1) if v == 1))
        check("padding is marked -1 and never scored", (b.kinds[0][b.kinds[0] < 0].numel() == 0) or (~b.mask[0][b.kinds[0] < 0]).all())
        check("ops=4 goes to its own eval bucket; no UNK tokens", bx.group_names == ["ops=0", "ops=2", "ops=4"] and bx.n_unk == 0, str(bx.group_names))

    print("scoring")
    Gb = Batch(torch.zeros(2, 4, dtype=torch.long), torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.bool),
               torch.zeros(2, 4, dtype=torch.long), torch.tensor([[0] * 4, [1] * 4]))
    lg = F.one_hot(torch.tensor([[9, 2, 3, 4], [9, 2, 0, 4]]), 10).float()  # row 1 wrong at one scored position
    r_, t_ = score(lg, Gb, "exact", 2)
    check("exact match: one wrong token fails the row", r_.tolist() == [1, 0] and t_.tolist() == [1, 1])
    r_, t_ = score(lg, Gb, "token", 2)
    check("token accuracy: 3/3 and 2/3, unscored position ignored", r_.tolist() == [3, 2] and t_.tolist() == [3, 3])

    print("verdict and gate diagnostics")
    mk = lambda d, a, h: {"delta": d, "attn": a, "hybrid:scalar": h}  # noqa: E731
    V = lambda d, a, h: verdict(mk(d, a, h), "hybrid:scalar", 0.03)[0].split(":")[0]  # noqa: E731
    check("verdict COMPLEMENTARY (each branch wins one task, hybrid gets both)", V({"s": 0.9, "r": 0.2}, {"s": 0.2, "r": 0.9}, {"s": 0.9, "r": 0.9}) == "COMPLEMENTARY")
    check("verdict SYNERGY (hybrid beats both)", V({"s": 0.9, "r": 0.2}, {"s": 0.2, "r": 0.9}, {"s": 0.99, "r": 0.9}) == "SYNERGY")
    check("verdict NO GAIN (one branch dominates, hybrid only matches it)", V({"s": 0.9, "r": 0.9}, {"s": 0.2, "r": 0.2}, {"s": 0.9, "r": 0.9}) == "NO GAIN")
    check("verdict INTERFERES (hybrid loses to a single branch)", V({"s": 0.9, "r": 0.2}, {"s": 0.2, "r": 0.9}, {"s": 0.5, "r": 0.9}) == "INTERFERES")
    check("verdict: hybrid that matches only ONE branch's task is not called complementary", V({"s": 0.9, "r": 0.2}, {"s": 0.2, "r": 0.9}, {"s": 0.9, "r": 0.2}) == "INTERFERES")
    check("verdict UNINFORMATIVE when nothing trains", verdict(mk({"s": 0.01}, {"s": 0.02}, {"s": 0.01}), "hybrid:scalar", 0.03)[0].startswith("UNINFORMATIVE"))
    kinds = torch.randint(0, 3, (20000,))
    gk = torch.stack([torch.tensor([0.9, 0.1, 0.5])[kinds]] * 2, 1) + 0.01 * torch.randn(20000, 2)
    gc = 0.5 + 0.001 * torch.randn(20000, 2)
    gn = 0.5 + 0.2 * torch.randn(20000, 2)
    check("eta2 ~1 when g depends on kind", eta2(gk, kinds) > 0.95, f"{eta2(gk, kinds):.3f}")
    check("eta2 ~0 when g is noise independent of kind", eta2(gn, kinds) < 0.01, f"{eta2(gn, kinds):.4f}")
    check("labels: constant / structured", gate_label(0.001, {"kind": 0.0}) == "near-constant" and gate_label(0.3, {"kind": 0.9}) == "structured")
    check("labels: token-id dependence alone is NOT called structured (it is a tautology in layer 0)",
          gate_label(0.3, {"kind": 0.0, "pos": 0.0, "token": 1.0}).startswith("varies"))

    print("training plumbing (a few CPU steps, then JSON round-trip)")
    A.n_h, A.lr, A.wd, A.warmup, A.lr_schedule, A.log_every, A.eval_every, A.quick_n, A.max_minutes = 2, 3e-3, 0.0, 5, "constant", 100, 0, 8, 0
    res = run_one(A, t3, "hybrid", "token", 64, 0, device, 6, 8)
    check("run_one returns gate stats and is JSON-serialisable", res["gate"] is not None and json.loads(json.dumps(res))["steps"] == 6)
    A.d, A.heads, A.head_dim, A.layers = 32, 2, 16, 1
    for var, mode in [("delta", None), ("attn", None), ("hybrid", "scalar"), ("hybrid", "token")]:
        rt = RecallTask(n_keys=16, n_vals=16, train_pairs=3, eval_pairs=(3,))
        set_seed(0)
        model = build(rt.vocab_size, A, var, mode, 64).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        b = Batch(*[t.to(device) for t in rt.train_batch(16, torch.Generator().manual_seed(0))])  # ONE fixed batch: a model that cannot memorise it has broken plumbing
        first = None
        for _ in range(250):
            loss = F.cross_entropy(model(b.x)[b.mask], b.y[b.mask])
            first = first or loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        check(f"overfits one fixed batch on {device}: {model_name(var, mode)}", loss.item() < 0.2 * first, f"{first:.2f} -> {loss.item():.3f}")

    print("\nSELFTEST", "PASSED" if ok_all else "FAILED")
    torch.set_default_dtype(torch_dtype)
    return ok_all


# =============================================================================
# 7. Driver: defaults, saving, and one function that runs a task
# =============================================================================
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK_DIR = "/kaggle/working" if ON_KAGGLE else os.path.abspath("hybrid_runs")  # results + Boxes data live here

DEFAULTS = dict(
    models="delta,attn,hybrid", hybrid_mode="scalar", seeds="0,1,2", quick=False, group="S3",
    steps=None, batch_size=None,                      # None = per-task default (TASK_DEFAULTS below)
    lr=1e-3, wd=0.01, warmup=200, lr_schedule="constant",  # constant: a decaying LR can suppress a late plateau escape
    d=128, layers=2, heads=4, head_dim=32, n_h=None,  # n_h None = n-1 for S_n, else 2
    chunk=32, conv_k=4, d_ff=0,                       # d_ff 0 = 2*d for the reference hybrid; others are matched to its params
    margin=0.03, n_eval=1000, eval_per_bucket=2000, max_train_ops=2, train_n=None,
    boxes_dir=os.path.join(WORK_DIR, "boxes_data"), boxes_zip=None,
    log_every=500, eval_every=2000, quick_n=200, max_minutes=0,
    out=os.path.join(WORK_DIR, "hybrid_results.json"), device=DEVICE,
)


def make_args(**overrides):
    unknown = set(overrides) - set(DEFAULTS)
    assert not unknown, f"unknown setting(s): {sorted(unknown)}; valid: {sorted(DEFAULTS)}"
    return argparse.Namespace(**{**DEFAULTS, **overrides})


TASK_DEFAULTS = {"s5": {"steps": 20000, "bs": 64}, "recall": {"steps": 6000, "bs": 128}, "boxes": {"steps": 8000, "bs": 64}}


SIG_KEYS = ["d", "layers", "heads", "head_dim", "n_h", "chunk", "conv_k", "d_ff", "lr", "wd", "warmup", "lr_schedule", "steps", "batch_size", "group",
            "max_train_ops", "train_n", "eval_per_bucket", "n_eval"]


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def save_results(path, res):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)


def run_task(a, name):
    dflt = TASK_DEFAULTS[name]
    a = argparse.Namespace(**vars(a))
    a.steps = a.steps or (300 if a.quick else dflt["steps"] if not (name == "s5" and a.group == "S3") else 6000)
    a.batch_size = a.batch_size or dflt["bs"]
    a.n_h = a.n_h or (int(a.group[1:]) - 1 if name == "s5" else 2)
    a.d_ff = a.d_ff or 2 * a.d
    if a.quick:
        a.n_eval, a.eval_per_bucket, a.eval_every, a.log_every, a.train_n = 100, 100, 0, 100, a.train_n or 2000
    seeds = [int(s) for s in a.seeds.split(",")][: 1 if a.quick else None]
    tag = f"{name}-{a.group}" if name == "s5" else name
    if a.quick:
        tag += "-QUICK"
    specs = [parse_model(m, a.hybrid_mode) for m in a.models.split(",")]
    results = load_results(a.out)
    sig = json.dumps({k: getattr(a, k, None) for k in SIG_KEYS}, sort_keys=True)
    todo = [(v, m, s) for v, m in specs for s in seeds
            if results["runs"].get(f"{tag}|{model_name(v, m)}|seed{s}", {}).get("sig") != sig]
    if todo:
        print(f"\n### task {tag}: {len(todo)} run(s) to do, {len(specs) * len(seeds) - len(todo)} already saved  (device {a.device})", flush=True)
        if name == "s5":
            task = PermGroupTask(a.group, n_eval=a.n_eval)
        elif name == "recall":
            task = RecallTask(n_eval=a.n_eval)
        else:
            task = BoxesTask(a.boxes_dir, a.boxes_zip, a.max_train_ops, a.train_n, a.eval_per_bucket)
        print(f"    {task.train_len_desc}", flush=True)
        ref_v, ref_m = next(((v, m) for v, m in specs if v == "hybrid"), ("hybrid", a.hybrid_mode))
        target = n_params(build(task.vocab_size, a, ref_v, ref_m, a.d_ff))
        d_ffs = {(v, m): (a.d_ff if (v, m) == (ref_v, ref_m) else match_d_ff(task.vocab_size, a, v, m, target)) for v, m in specs}
        for (v, m), f in d_ffs.items():
            print(f"    {model_name(v, m):<16} d_ff {f:<5} params {n_params(build(task.vocab_size, a, v, m, f)):,}  (target {target:,})", flush=True)
        for v, m, s in todo:
            key = f"{tag}|{model_name(v, m)}|seed{s}"
            print(f"\n>>> {key}", flush=True)
            r = run_one(a, task, v, m, d_ffs[(v, m)], s, a.device, a.steps, a.batch_size)
            r.update(tag=tag, sig=sig)
            results["runs"][key] = r
            save_results(a.out, results)
            print(f"<<< {key}  " + "  ".join(f"{k} {x:.3f}" for k, x in r["final"].items()) + f"   ({r['minutes']:.1f} min)", flush=True)
    print_summary(results, tag, a.margin)
    if a.quick:
        print("QUICK RUN: 300 steps. These numbers say the pipeline works, nothing about the models.")
