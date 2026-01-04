import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q, K, V, O,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vk, stride_vn,
    stride_oh, stride_om, stride_ok,
    SEQ_LEN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,  # head dim (compile-time)
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # --------------------
    # Load Q
    # --------------------
    q_ptrs = Q + (
        pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # --------------------
    # Loop over K/V blocks
    # --------------------
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        offs_n_curr = start_n + offs_n

        k_ptrs = K + (
            pid_h * stride_kh
            + offs_n_curr[None, :] * stride_kn
            + offs_d[:, None] * stride_kk
        )
        k = tl.load(k_ptrs, mask=offs_n_curr[None, :] < SEQ_LEN, other=0.0)

        scores = tl.dot(q, k)
     #  scores *= 1.0 / tl.sqrt(tl.float32(BLOCK_D))
        scores *= tl.math.rsqrt(tl.full([], BLOCK_D, tl.float32))
        # causal mask
        causal = offs_n_curr[None, :] <= offs_m[:, None]
        scores = tl.where(causal, scores, -1e9)

        p = tl.softmax(scores)

        v_ptrs = V + (
            pid_h * stride_vh
            + offs_n_curr[:, None] * stride_vk
            + offs_d[None, :] * stride_vn
        )
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < SEQ_LEN, other=0.0).to(tl.float32)

        acc += tl.dot(p, v)

    # --------------------
    # Store output
    # --------------------
    o_ptrs = O + (
        pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok
    )
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < SEQ_LEN)


def triton_attention(q, k, v):
    B, H, M, D = q.shape
    assert D == 64, "This kernel assumes head_dim = 64"

    o = torch.empty_like(q)

    grid = (triton.cdiv(M, 16), H)

    attention_kernel[grid](
        q, k, v, o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        M,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_D=64,
    )

    return o

