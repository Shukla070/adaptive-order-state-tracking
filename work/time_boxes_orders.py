"""Controlled timing on Boxes: the [2, 1] model (2 factors in the first layer, 1 in the
second) against fixed 2, with fixed 1 as the floor and the gated model as a control.

Pre-registered in PLAN.md ("Timing [2, 1] against fixed 2", 24 October) before any run.

Why: [2, 1] matched fixed 2 under the pre-registered rule (24 Oct) and its whole
training runs took 23.6 min against 33.4 min. Whole runs include evaluation and
logging; this measures the model alone, on the same GPU, in the same process.

Models, all at 469,120 parameters (MLP width matched, as in training):
  fixed2   [2, 2]   d_ff 455   (trained checkpoint if given)
  layers21 [2, 1]   d_ff 522   (trained checkpoint if given)
  fixed1   [1, 1]   d_ff matched (untrained: timing does not depend on the weights --
                    the recurrence has no data-dependent control flow)
  gated    the learned gate at gamma 0.003, if its checkpoint is given: a CONTROL.
           Its order is ~1.4 per token, but closed factors are still computed and
           multiplied by zero, so it should cost about as much as fixed 2.

Workloads (real Boxes data; the SAME batches for every model):
  infer64   inference, batch 64: 12 test batches, 2 per bucket (0..5+ operations)
  infer1    single-example latency: 24 test examples, 4 per bucket
  train64   one training step, batch 64 (forward, loss, backward, clipping, AdamW):
            8 training batches
Protocol: 3 warm-up passes per model and workload, then 30 rounds; within a round the
models run in a random order (seeded) so drift in a laptop GPU's clock is shared.
Per round, ratio = time([2, 1]) / time(fixed 2). Reported: median ratio and a bootstrap
95% interval of that median (2 000 resamples of rounds).

Decision (per workload), written before any run
  MEASURED SAVING   the interval lies entirely below 0.95 -> state the median saving
  NO CLAIM          otherwise (a difference under ~5% could come from the harness)
  Control: the gated model's median ratio to fixed 2 is reported; a value near 1 shows
  the gate's per-token saving is notional without the faster path.
Scope: this is the pure-PyTorch recurrence (boxes_lib.deltaproduct) used in every
Boxes run, float32, on whatever GPU runs it. Not the Triton kernels used on S5.

Usage
  python work/time_boxes_orders.py --selftest
  python -u work/time_boxes_orders.py --ckpt_dir checkpoints/boxes_gate > results/boxes_timing.log 2>&1
  (the GPU must be idle: the script refuses to start while a training script is running)
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
import boxes_lib as bl  # noqa: E402
import exp_boxes_gate as eg  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ORDERS = {"fixed2": (2, 2), "layers21": (2, 1), "fixed1": (1, 1)}
CKPT = {"fixed2": "fixed22_g0.0_seed5.pt", "layers21": "fixed21_g0.0_seed0.pt", "gated": "learned_g0.003_seed0.pt"}
WARMUP, ROUNDS, BOOT, SEED = 3, 30, 2000, 1234
INFER_PER_BUCKET, LAT_PER_BUCKET, TRAIN_BATCHES, BS = 2, 4, 8, 64
WORKLOADS = ("infer64", "infer1", "train64")
CLAIM_BELOW = 0.95   # claim a saving only if the 95% interval of the ratio lies below 0.95


# ----------------------------------------------------------------------------- models

def match_d_ff(vocab, a, orders, target=eg.TARGET_PARAMS):
    """MLP width bringing a model with these per-layer orders closest to `target`."""
    p1 = bl.n_params(eg.build_model(vocab, a, "fixed21", 1, orders=orders))
    p2 = bl.n_params(eg.build_model(vocab, a, "fixed21", 2, orders=orders))
    slope = p2 - p1
    return max(1, round((target - (p1 - slope)) / slope))


def build(name, vocab, a, ckpt_dir=None, device="cpu"):
    """-> (model, info). Loads the trained checkpoint when there is one, and refuses a
    checkpoint whose layer orders differ from what the name says."""
    path = os.path.join(ckpt_dir, CKPT[name]) if ckpt_dir and name in CKPT else None
    if name == "gated":
        if not path or not os.path.exists(path):
            return None, None
        ck = torch.load(path, map_location="cpu")
        m = eg.build_model(vocab, a, "learned", ck["d_ff"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device), {"orders": None, "d_ff": ck["d_ff"], "checkpoint": path, "params": bl.n_params(m)}
    orders = ORDERS[name]
    d_ff = match_d_ff(vocab, a, orders)
    m = eg.build_model(vocab, a, "fixed21", d_ff, orders=orders)
    loaded = None
    if path and os.path.exists(path):
        ck = torch.load(path, map_location="cpu")
        if tuple(ck.get("layer_orders", ())) != orders or ck["d_ff"] != d_ff:
            raise ValueError(f"{path}: layer orders {ck.get('layer_orders')} / d_ff {ck['d_ff']} "
                             f"do not match {name} {orders} / {d_ff}")
        m.load_state_dict(ck["state_dict"])
        loaded = path
    return m.to(device), {"orders": list(orders), "d_ff": d_ff, "checkpoint": loaded, "params": bl.n_params(m)}


# ----------------------------------------------------------------------------- workloads

def make_workloads(task, gen_seed=SEED):
    """The same batches for every model. Inference batches come from the test split,
    training batches from the training split (the generator used in training)."""
    infer, singles = [], []
    for gi, b in enumerate(task.bucket_of_group):
        exs = task.eval_sets[b]
        for j in range(INFER_PER_BUCKET):
            chunk = exs[j * BS:(j + 1) * BS]
            if chunk:
                infer.append(task._collate(chunk, lambda _b, gi=gi: gi))
        for ex in exs[:LAT_PER_BUCKET]:
            singles.append(task._collate([ex], lambda _b, gi=gi: gi))
    gen = torch.Generator().manual_seed(gen_seed)
    train = [task.train_batch(BS, gen) for _ in range(TRAIN_BATCHES)]
    return {"infer64": infer, "infer1": singles, "train64": train}


def fingerprint(batches):
    h = hashlib.sha1()
    for b in batches:
        h.update(b.x.numpy().tobytes())
    return h.hexdigest()[:12]


def to_dev(batches, device):
    return [bl.Batch(*[t.to(device) for t in b]) for b in batches]


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - t


def infer_pass(model, batches):
    def f():
        model.eval()
        with torch.no_grad():
            for b in batches:
                model(b.x)
    return f


def train_pass(model, opt, batches):
    def f():
        model.train()
        for b in batches:
            loss = F.cross_entropy(model(b.x)[b.mask], b.y[b.mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return f


# ----------------------------------------------------------------------------- harness and statistics

def run_rounds(passes, rounds=ROUNDS, warmup=WARMUP, seed=SEED, device="cpu"):
    """passes: {model name: zero-argument callable}. Warm-up, then `rounds` rounds with the
    models in a seeded random order each round. -> ({name: [seconds per round]}, [orders])."""
    names = list(passes)
    for n in names:
        for _ in range(warmup):
            passes[n]()
    rng = random.Random(seed)
    times, orders = {n: [] for n in names}, []
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        orders.append(order)
        for n in order:
            times[n].append(timed(passes[n], device))
    return times, orders


def ratios(times, a, b):
    """Per-round ratio time[a] / time[b]."""
    return [x / y for x, y in zip(times[a], times[b])]


def boot_ci(xs, n=BOOT, seed=SEED, q=(0.025, 0.975)):
    """Bootstrap interval of the median of xs (resampling rounds)."""
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(q[0] * (n - 1))], meds[int(q[1] * (n - 1))]


def decide(ci):
    """A saving is claimed only if the whole interval lies below CLAIM_BELOW: a difference
    of a few percent can come from the harness itself (order, caching), not the model."""
    return "MEASURED SAVING" if ci[1] < CLAIM_BELOW else "NO CLAIM"


def accuracy(model, batches, G):
    """Exact match on the inference batches: a sanity check that trained weights were loaded."""
    model.eval()
    right, total = 0.0, 0.0
    with torch.no_grad():
        for b in batches:
            r, t = bl.score(model(b.x), b, "exact", G)
            right += float(r.sum()); total += float(t.sum())
    return right / total if total else float("nan")


def gpu_busy():
    """True if a training script of this project is running (the GPU must be idle)."""
    try:
        out = subprocess.run(["pgrep", "-f", "exp_gate_train.py|exp_boxes_order.py|exp_boxes_gate.py"],
                             capture_output=True, text=True)
        return bool(out.stdout.strip())
    except FileNotFoundError:
        return False


# ----------------------------------------------------------------------------- self-test

def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    print("B1  model sizes")
    a = bl.make_args()
    vocab = 123
    f22, f21, f11 = (match_d_ff(vocab, a, ORDERS[k]) for k in ("fixed2", "layers21", "fixed1"))
    check("fixed 2 -> d_ff 455 and [2, 1] -> d_ff 522, as trained", f22 == 455 and f21 == 522, f"{f22}, {f21}")
    a1 = bl.make_args(n_h=1)
    ref11 = bl.match_d_ff(vocab, a1, "delta", None, eg.TARGET_PARAMS)
    check("fixed 1 width = the order sweep's fixed-1 width", f11 == ref11, f"{f11} vs {ref11}")
    ps = {k: bl.n_params(eg.build_model(vocab, a, "fixed21", f, orders=ORDERS[k])) for k, f in
          (("fixed2", f22), ("layers21", f21), ("fixed1", f11))}
    check("all three within 0.2% of 469,120", all(abs(p - eg.TARGET_PARAMS) / eg.TARGET_PARAMS < 0.002 for p in ps.values()), str(ps))
    m21, _ = build("layers21", vocab, a)
    check("[2, 1] built with orders [2, 1]", eg.layer_orders(m21) == [2, 1])

    print("B2  checkpoints: loaded when present, refused when they do not match")
    import tempfile
    d = tempfile.mkdtemp()
    small = bl.make_args(d=32, heads=2, head_dim=16, layers=2)
    ms = eg.build_model(40, small, "fixed21", match_d_ff(40, small, (2, 1)), orders=(2, 1))
    torch.save({"state_dict": ms.state_dict(), "d_ff": match_d_ff(40, small, (2, 1)), "layer_orders": [2, 1]},
               os.path.join(d, CKPT["layers21"]))
    mb, info = build("layers21", 40, small, d)
    check("matching checkpoint is loaded (same weights)", info["checkpoint"] is not None
          and all(torch.equal(x, y) for x, y in zip(ms.state_dict().values(), mb.state_dict().values())))
    torch.save({"state_dict": ms.state_dict(), "d_ff": match_d_ff(40, small, (2, 1)), "layer_orders": [2, 1]},
               os.path.join(d, CKPT["fixed2"]))
    try:
        build("fixed2", 40, small, d)
        refused = False
    except ValueError:
        refused = True
    check("a [2, 1] checkpoint offered as fixed 2 is refused", refused)
    check("no checkpoint -> untrained model, recorded as such", build("fixed1", 40, small, d)[1]["checkpoint"] is None)
    check("gated control is skipped when its checkpoint is absent", build("gated", 40, small, d)[0] is None)

    print("B3  harness on known answers (sleeps of known length)")
    nap = lambda s: (lambda: time.sleep(s))
    t, orders = run_rounds({"fast": nap(0.004), "slow": nap(0.008), "same": nap(0.008)}, rounds=12, warmup=1)
    r = ratios(t, "fast", "slow")
    med = statistics.median(r)
    ci = boot_ci(r)
    check("4 ms against 8 ms: median ratio ~0.5", 0.4 < med < 0.6, f"{med:.3f}")
    check("4 ms against 8 ms: interval below 1 -> MEASURED SAVING", decide(ci) == "MEASURED SAVING", f"{ci[0]:.3f}-{ci[1]:.3f}")
    r2 = ratios(t, "same", "slow")
    check("8 ms against 8 ms: NO CLAIM", decide(boot_ci(r2)) == "NO CLAIM", f"median {statistics.median(r2):.3f}")
    check("models run in a different order across rounds", len({tuple(o) for o in orders}) > 1)
    t2, orders2 = run_rounds({"fast": nap(0.0), "slow": nap(0.0), "same": nap(0.0)}, rounds=12, warmup=0)
    check("round order is reproducible (seeded)", orders2 == orders)

    print("B4  statistics on known data")
    check("interval of a constant is that constant", boot_ci([0.7] * 30) == (0.7, 0.7))
    check("decision: interval [0.70, 0.94] -> saving; [0.80, 0.97] and [0.80, 1.00] -> no claim",
          decide((0.70, 0.94)) == "MEASURED SAVING" and decide((0.80, 0.97)) == "NO CLAIM" and decide((0.80, 1.00)) == "NO CLAIM")
    xs = [0.5] * 21 + [1.5] * 10
    check("interval is of the MEDIAN, not of single rounds (21 x 0.5, 10 x 1.5 -> upper 0.5)", boot_ci(xs)[1] == 0.5, str(boot_ci(xs)))
    check("ratio direction: time(a) / time(b)", ratios({"a": [1.0], "b": [2.0]}, "a", "b") == [0.5])

    print("B5  workloads: the same batches for every model; training updates; inference does not")
    tk = eg.FakeBoxes()
    gen = torch.Generator().manual_seed(3)
    batches = [tk.train_batch(4, gen) for _ in range(2)]
    check("fingerprint is stable and changes with the data",
          fingerprint(batches) == fingerprint(batches) and fingerprint(batches) != fingerprint(batches[:1]))
    ab = bl.make_args(d=32, heads=2, head_dim=16, layers=2)
    m = eg.build_model(tk.vocab_size, ab, "fixed21", 32, orders=(2, 1))
    before = [p.detach().clone() for p in m.parameters()]
    m.train()
    infer_pass(m, batches)()
    check("inference pass leaves the weights unchanged and the model in eval mode",
          all(torch.equal(x, y) for x, y in zip(before, m.parameters())) and not m.training)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    train_pass(m, opt, batches)()
    check("training pass changes the weights", any(not torch.equal(x, y) for x, y in zip(before, m.parameters())))
    check("training pass puts the model in train mode", m.training)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt_dir", default=os.path.join(REPO, "checkpoints", "boxes_gate"))
    ap.add_argument("--boxes_dir", default=os.path.join(REPO, "boxes_data"))
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--warmup", type=int, default=WARMUP, help="warm-up passes (smoke runs only should change it)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "boxes_timing.json"))
    ap.add_argument("--device", default=bl.DEVICE)
    ap.add_argument("--force", action="store_true", help="run even if a training script is running (timings unreliable)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if gpu_busy() and not args.force:
        raise SystemExit("a training script is running: timings would be unreliable. Wait for it to finish.")
    dev = args.device
    torch.backends.cudnn.benchmark = False
    a = bl.make_args(models="delta", boxes_dir=args.boxes_dir, device=dev)
    task = bl.BoxesTask(a.boxes_dir, None, a.max_train_ops, a.train_n, a.eval_per_bucket)
    G = len(task.group_names)
    wl_cpu = make_workloads(task)
    fps = {k: fingerprint(v) for k, v in wl_cpu.items()}
    wl = {k: to_dev(v, dev) for k, v in wl_cpu.items()}
    print(f"workloads: infer64 {len(wl['infer64'])} batches, infer1 {len(wl['infer1'])} examples, "
          f"train64 {len(wl['train64'])} batches; fingerprints {fps}", flush=True)

    models, info = {}, {}
    for name in ("fixed2", "layers21", "fixed1", "gated"):
        m, inf = build(name, task.vocab_size, a, args.ckpt_dir, dev)
        if m is None:
            print(f"  {name}: checkpoint not found, skipped", flush=True)
            continue
        models[name], info[name] = m, inf
        inf["acc_on_infer64"] = accuracy(m, wl["infer64"], G)
        print(f"  {name}: orders {inf['orders']} d_ff {inf['d_ff']} params {inf['params']:,} "
              f"checkpoint {inf['checkpoint']}  exact match on the timed batches {inf['acc_on_infer64']:.3f}", flush=True)

    res = {"device": torch.cuda.get_device_name(0) if str(dev).startswith("cuda") else "cpu",
           "torch": torch.__version__, "rounds": args.rounds, "warmup": args.warmup, "fingerprints": fps,
           "models": info, "workloads": {}}
    for w in WORKLOADS:
        if w == "train64":   # fresh copies so training does not touch the timed inference weights
            import copy
            passes = {}
            for n, m in models.items():
                mc = copy.deepcopy(m)
                passes[n] = train_pass(mc, torch.optim.AdamW(mc.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd), wl[w])
        else:
            passes = {n: infer_pass(m, wl[w]) for n, m in models.items()}
        times, orders = run_rounds(passes, rounds=args.rounds, warmup=args.warmup, device=dev)
        r = ratios(times, "layers21", "fixed2")
        ci = boot_ci(r)
        entry = {"times": times, "round_orders": orders, "median_s": {n: statistics.median(v) for n, v in times.items()},
                 "ratio_21_over_2": {"median": statistics.median(r), "ci95": ci, "decision": decide(ci)}}
        for other in ("fixed1", "gated"):
            if other in times:
                ro = ratios(times, other, "fixed2")
                entry[f"ratio_{other}_over_2"] = {"median": statistics.median(ro), "ci95": boot_ci(ro)}
        res["workloads"][w] = entry
        per = {"infer64": len(wl["infer64"]), "infer1": len(wl["infer1"]), "train64": len(wl["train64"])}[w]
        print(f"\n{w}: median ms per {'batch' if w != 'infer1' else 'example'}  "
              + "  ".join(f"{n} {1000 * s / per:.2f}" for n, s in entry["median_s"].items()), flush=True)
        print(f"  [2, 1] / fixed 2: median {entry['ratio_21_over_2']['median']:.3f}  95% interval "
              f"{ci[0]:.3f}-{ci[1]:.3f}  -> {entry['ratio_21_over_2']['decision']}", flush=True)
        for other in ("fixed1", "gated"):
            k = f"ratio_{other}_over_2"
            if k in entry:
                print(f"  {other} / fixed 2: median {entry[k]['median']:.3f}  ({entry[k]['ci95'][0]:.3f}-{entry[k]['ci95'][1]:.3f})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
