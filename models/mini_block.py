import torch
import torch.nn as nn

from kernels.triton_flash import triton_attention
from kernels.quant_linear import QuantLinear


class TorchMiniBlock(nn.Module):
    """
    Baseline PyTorch inference block
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, H, M, D]
        attn = torch_attention(x, x, x)
        return self.linear(attn)


class OptimizedMiniBlock(nn.Module):
    """
    Optimized inference-only block
    """

    def __init__(self, dim):
        super().__init__()

        weight = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
        self.linear = QuantLinear(weight)

    def forward(self, x):
        attn = triton_attention(x, x, x)
        return self.linear(attn)


def torch_attention(q, k, v):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d**0.5
    mask = torch.triu(torch.ones_like(scores), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)
