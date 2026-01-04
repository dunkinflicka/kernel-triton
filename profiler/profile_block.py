import torch
import nvtx
from torch.profiler import profile, record_function, ProfilerActivity

from models.mini_block import TorchMiniBlock, OptimizedMiniBlock


def run(model, x, tag):
    with nvtx.annotate(tag):
        model(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, M, D = 2, 8, 256, 64
    x = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

    torch_block = TorchMiniBlock(D).cuda().half()
    opt_block = OptimizedMiniBlock(D)

    # Warmup
    for _ in range(10):
        run(torch_block, x, "torch_warmup")
        run(opt_block, x, "opt_warmup")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        torch.cuda.synchronize()
        run(torch_block, x, "torch_block")
        torch.cuda.synchronize()
        run(opt_block, x, "optimized_block")
        torch.cuda.synchronize()


    print(
        prof.key_averages()
        .table(sort_by="cuda_time_total", row_limit=15)
    )

prof.export_chrome_trace("profiler/mini_block_trace.json")
