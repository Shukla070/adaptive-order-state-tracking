"""
Where do the 22 minutes per epoch actually go?

Measured: one S3 epoch took 22m34s. GPU compute for 387 steps at batch 256 is
~21 seconds (work/bench_batch.py). So ~98% of wall clock is NOT the GPU.

Three candidates, three different fixes:
  A. num_workers=0        -> collation blocks the GPU        -> 3 DataLoader args
  B. Arrow cache on /mnt/c -> random reads over 9p is slow   -> move the repo
  C. Triton recompiling    -> minutes of startup per run     -> set TRITON_CACHE_DIR

This measures A and B and reports where the cache lives. Run from the repo root:
    python work/bench_dataloader.py
"""
import os
import time
import torch
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader

CSV = Path("state_tracking/data/S3=128.csv")
BATCH = 256


def where_is_the_cache(ds):
    files = set()
    for f in getattr(ds, "cache_files", []) or []:
        p = f.get("filename")
        if p:
            files.add(p)
    return files


def simple_collate(samples):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }


def main():
    if not CSV.exists():
        raise SystemExit(f"not found: {CSV}  (run from the repo root)")

    print("=" * 68)
    print("1. LOAD")
    t0 = time.perf_counter()
    ds = load_dataset("csv", data_files=str(CSV), split="all")
    print(f"   load_dataset: {time.perf_counter()-t0:.1f}s   rows={ds.num_rows}")

    cache = where_is_the_cache(ds)
    print(f"\n2. WHERE THE ARROW CACHE LIVES")
    for c in cache or ["(none reported)"]:
        slow = "  <<< ON /mnt/c — 9p, slow random reads" if "/mnt/" in str(c) else "  (native fs, fine)"
        print(f"   {c}{slow}")
    print(f"   HF_HOME={os.environ.get('HF_HOME', '(unset -> ~/.cache/huggingface)')}")
    print(f"   TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR', '(unset -> ~/.triton)')}")

    # tokenize-ish: turn the space-separated columns into fixed-length tensors
    print(f"\n3. MAP (this is what took 25s in the real run)")
    t0 = time.perf_counter()
    def to_ids(ex):
        return {"input_ids": [int(x) for x in ex["input"].split()],
                "labels": [int(x) for x in ex["target"].split()]}
    cols = ds.column_names
    ds = ds.map(to_ids, remove_columns=[c for c in cols if c in ("input", "target")])
    ds.set_format("torch", columns=["input_ids", "labels"])
    print(f"   map+format: {time.perf_counter()-t0:.1f}s")

    print(f"\n4. RANDOM ACCESS (what shuffle=True does, {BATCH} rows)")
    import random
    idx = [random.randrange(ds.num_rows) for _ in range(BATCH)]
    t0 = time.perf_counter()
    _ = [ds[i] for i in idx]
    dt = time.perf_counter() - t0
    print(f"   {BATCH} random single-row reads: {dt*1000:.0f} ms  ({dt/BATCH*1e6:.0f} us/row)")

    print(f"\n5. DATALOADER THROUGHPUT (30 batches each)")
    print(f"   {'workers':>8} {'s/batch':>9} {'batches/s':>10} {'est. epoch':>12}")
    print("   " + "-" * 43)
    n_batches_per_epoch = 99000 // BATCH
    best = None
    for nw in (0, 2, 4, 8):
        try:
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=simple_collate,
                            num_workers=nw, pin_memory=(nw > 0),
                            persistent_workers=(nw > 0))
            it = iter(dl)
            next(it)                      # warm up workers
            t0 = time.perf_counter()
            for _ in range(30):
                next(it)
            dt = (time.perf_counter() - t0) / 30
            epoch = dt * n_batches_per_epoch
            mark = ""
            if best is None or dt < best[1]:
                best = (nw, dt); mark = ""
            print(f"   {nw:>8} {dt:>9.4f} {1/dt:>10.1f} {epoch/60:>10.1f} min{mark}")
            del dl, it
        except Exception as e:
            print(f"   {nw:>8}  ERROR {type(e).__name__}: {e}")

    print("\n" + "=" * 68)
    if best:
        nw, dt = best
        print(f"FASTEST: num_workers={nw} -> {dt*n_batches_per_epoch/60:.1f} min/epoch of data loading")
        print(f"GPU compute for one epoch is ~21 s, so the floor is ~0.4 min/epoch.")
    print("Compare against the measured 22.6 min/epoch to see how much is recoverable.")


if __name__ == "__main__":
    main()
