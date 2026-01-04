import torch
from models.mini_block import TorchMiniBlock, OptimizedMiniBlock


def benchmark(model, x, iters=100):
    # warmup
    for _ in range(10):
        model(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        model(x)
    end.record()

    torch.cuda.synchronize()

    latency = start.elapsed_time(end) / iters
    vram = torch.cuda.max_memory_allocated() / 1e6
    return latency, vram


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, M, D = 2, 8, 256, 64
    x = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

    torch_block = TorchMiniBlock(D).cuda().half()
    opt_block = OptimizedMiniBlock(D)

    torch_lat, torch_mem = benchmark(torch_block, x)
    opt_lat, opt_mem = benchmark(opt_block, x)

    print("\n=== MINI BLOCK COMPARISON ===")
    print(f"PyTorch   : {torch_lat:.4f} ms | {torch_mem:.2f} MB")
    print(f"Optimized : {opt_lat:.4f} ms | {opt_mem:.2f} MB")
    print(f"Speedup   : {torch_lat / opt_lat:.2f}x")
