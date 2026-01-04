import torch
import time
import json
from pathlib import Path


BATCH=1
HEADS=4
SEQ_LEN=256
DIM=64
DEVICE="cuda"



def torch_attention(q,k,v):
    d=q.size(-1)
 
    # Q@K^t

    scores=torch.matmul(q, k.transpose(-2,-1))/d**0.5
   
    # casual mask no future

    mask=torch.triu(torch.ones_like(scores),diagonal=1)
    scores=scores.masked_fill(mask.bool(), float("-inf"))
    
    #softmax

    attn=torch.softmax(scores,dim=-1)


    # attention(weights) @ V

    return torch.matmul(attn,v)



def benchmark():
    torch.manual_seed(0)

    q=torch.randn(BATCH,SEQ_LEN,DIM,device=DEVICE, dtype=torch.float16)
    k=torch.randn_like(q)
    v=torch.randn_like(q)




    for _ in range(10):
        torch_attention(q,k,v)
    

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()



    start= torch.cuda.Event(enable_timing=True)
    end= torch.cuda.Event(enable_timing=True)

    iters = 100
    start.record()
    for _ in range(iters):
        out = torch_attention(q, k, v)
    end.record()

    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "batch": BATCH,
        "heads": HEADS,
        "seq_len": SEQ_LEN,
        "dim": DIM,
        "latency_ms": latency_ms,
        "peak_vram_mb": peak_vram_mb,
    }

# -----------------------------
# Save results
# -----------------------------
if __name__ == "__main__":
    results = benchmark()

    print("\nBASELINE RESULTS")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "baseline.json", "w") as f:
        json.dump(results, f, indent=2)
