import torch
import json
from pathlib import Path

from kernels.triton_flash import triton_attention


# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda"
DTYPE = torch.float16

BATCHES = [1, 2]
HEADS = [4, 8]
SEQ_LENS = [128, 256, 512]
DIM = 64
ITERS = 100


# -----------------------------
# PyTorch Attention
# -----------------------------
def torch_attention(q, k, v):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d**0.5
    mask = torch.triu(torch.ones_like(scores), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


# -----------------------------
# Benchmark helper
# -----------------------------
def benchmark(fn, q, k, v):
    # warmup
    for _ in range(10):
        fn(q, k, v)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        fn(q, k, v)
    end.record()

    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / ITERS
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    return latency_ms, peak_vram_mb


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    results = []

    for B in BATCHES:
        for H in HEADS:
            for M in SEQ_LENS:
                q = torch.randn(B, H, M, DIM, device=DEVICE, dtype=DTYPE)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                torch_lat, torch_mem = benchmark(torch_attention, q, k, v)
                triton_lat, triton_mem = benchmark(triton_attention, q, k, v)

                entry = {
                    "batch": B,
                    "heads": H,
                    "seq_len": M,
                    "torch_latency_ms": torch_lat,
                    "torch_vram_mb": torch_mem,
                    "triton_latency_ms": triton_lat,
                    "triton_vram_mb": triton_mem,
                    "speedup": torch_lat / triton_lat,
                }

                print(entry)
                results.append(entry)

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "attention.json", "w") as f:
        json.dump(results, f, indent=2)
