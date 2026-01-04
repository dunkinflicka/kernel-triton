import torch
from kernels.triton_flash import triton_attention

def torch_attention(q, k, v):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d**0.5
    mask = torch.triu(torch.ones_like(scores), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)

torch.manual_seed(0)

q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out_torch = torch_attention(q, k, v)
out_triton = triton_attention(q, k, v)

torch.testing.assert_close(
    out_triton, out_torch,
    rtol=1e-2, atol=1e-2
)

print("✅ Triton attention matches PyTorch")
