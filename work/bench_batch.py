"""
Find the batch size where this GPU falls off a cliff.

On Windows/WSL2, exceeding VRAM does NOT raise a CUDA OOM. The NVIDIA driver
silently spills to shared system memory over PCIe, and everything keeps running
-- 10-100x slower. That looks exactly like "training is stuck".

This times one forward+backward at increasing batch sizes and reports peak VRAM.
Look for the point where time-per-step jumps by an order of magnitude while
allocated memory stops rising: that is the spill.

    python work/bench_batch.py
"""
import time
import torch
from fla.models import GatedDeltaProductConfig
from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM

DEV = "cuda"
SEQ = 128          # --k=128
VOCAB = 13         # n_vocab from the S3 run


def make_model(num_householder=2):
    cfg = GatedDeltaProductConfig(
        hidden_size=128,            # d_state
        head_dim=32,
        num_heads=8,                # --n_heads=8
        num_hidden_layers=1,        # --n_layers=1
        vocab_size=VOCAB,
        num_householder=num_householder,
        allow_neg_eigval=True,
        use_forget_gate=True,
        expand_v=2,
        max_position_embeddings=2048,
    )
    return GatedDeltaProductForCausalLM(cfg).to(DEV, torch.bfloat16)


def bench(model, bs, iters=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, VOCAB, (bs, SEQ), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup (triton compiles here)
    out = model(input_ids=x, labels=x)
    out.loss.backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    peak = torch.cuda.max_memory_allocated() / 1e9
    resv = torch.cuda.max_memory_reserved() / 1e9
    return dt, peak, resv


def main():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM {total:.1f} GB")
    print(f"seq_len={SEQ}, n_h=2, 1 layer, bf16\n")
    print(f"{'batch':>6} {'s/step':>9} {'steps/s':>9} {'alloc GB':>9} {'resv GB':>9}  note")
    print("-" * 62)

    model = make_model()
    prev = None
    for bs in (64, 128, 256, 512, 1024, 2048):
        try:
            dt, peak, resv = bench(model, bs)
        except torch.OutOfMemoryError:
            print(f"{bs:>6} {'OOM':>9}")
            break
        except Exception as e:
            print(f"{bs:>6}  ERROR {type(e).__name__}: {e}")
            break

        note = ""
        if prev is not None:
            # time should scale ~linearly with batch. much worse => spilling.
            expected = prev[0] * (bs / prev[1])
            if dt > 3 * expected:
                note = "<<< SPILL to system RAM"
            elif dt > 1.6 * expected:
                note = "<-- degrading"
        print(f"{bs:>6} {dt:>9.3f} {1/dt:>9.2f} {peak:>9.2f} {resv:>9.2f}  {note}")
        prev = (dt, bs)

    print("\nPick the largest batch BEFORE any spill/degrading marker.")
    print("Steps per epoch = 99000 / batch. Epoch time = steps / (steps per second).")


if __name__ == "__main__":
    main()
