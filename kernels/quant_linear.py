import torch
import torch.nn as nn


class QuantLinear(nn.Module):
    """
    Weight-only INT8 quantized Linear layer.
    Inference-only. No backward.
    """

    def __init__(self, weight_fp16: torch.Tensor):
        super().__init__()

        # Expect [out_features, in_features]
        assert weight_fp16.ndim == 2
        assert weight_fp16.dtype == torch.float16

        # ----------------------------
        # Per-output-channel scaling
        # ----------------------------
        # For each output row, find max absolute value
        max_val = weight_fp16.abs().max(dim=1, keepdim=True)[0]

        # Prevent division by zero
        scale = (max_val / 127.0).clamp(min=1e-6)

        # Quantize
        qweight = torch.round(weight_fp16 / scale).to(torch.int8)

        # Store as buffers (not parameters)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]
        returns: [..., out_features]
        """
        # Dequantize on the fly
        w = self.qweight.float() * self.scale
        out =x.float() @ w.T
        return out.to(x.dtype)
