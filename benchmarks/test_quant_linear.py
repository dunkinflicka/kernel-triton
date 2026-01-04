import torch
from kernels.quant_linear import QuantLinear


def test_quant_linear():
    torch.manual_seed(0)

    # Shapes
    B = 2
    IN = 128
    OUT = 64

    x = torch.randn(B, IN, device="cuda", dtype=torch.float16)
    w = torch.randn(OUT, IN, device="cuda", dtype=torch.float16)

    # Baseline fp16 Linear
    linear = torch.nn.Linear(IN, OUT, bias=False).cuda().half()
    linear.weight.data.copy_(w)

    # Quantized Linear
    qlinear = QuantLinear(w)

    y_fp16 = linear(x)
    y_int8 = qlinear(x)

    torch.testing.assert_close(
        y_fp16,
        y_int8,
        rtol=2e-1,
        atol=2e-1,
    )

    print("✅ QuantLinear output matches fp16 Linear (approx)")


if __name__ == "__main__":
    test_quant_linear()
