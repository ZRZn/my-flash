# -*- coding: utf-8 -*-
# 简化版的Flash Attention实现

import math
import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils.op import exp, log, safe_exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV'],
)
@triton.jit
def simple_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_n = i_b
    bos, eos = i_n * T, i_n * T + T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # 将Q块保留在共享内存中
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # 处理当前块之前的所有块
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, 0), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k)

        # [BT, BS]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [BT, BS]
        b_p = safe_exp(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m

    # 处理当前块及其后的块
    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, 0), (BS, BV), (1, 0))

        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, float('-inf'))

        # [BT]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [BT, BS]
        b_p = safe_exp(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m

    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def simple_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def simple_attn_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_n = i_b
    bos, eos = i_n * T, i_n * T + T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT]
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (0, i_s), (BV, BS), (0, 1))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k)

        b_p = safe_exp(b_s - b_lse[:, None])
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (0, i_s), (BV, BS), (0, 1))
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, -float('inf'))

        b_p = safe_exp(b_s - b_lse[:, None])  # 重要：使用safe_exp避免NaN
        b_p = tl.where(o_q[:, None] >= o_k[None, :], b_p, 0)

        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def simple_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_n = i_b
    bos, eos = i_n * T, i_n * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))

    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, 0), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_s = tl.where(o_k[:, None] <= o_q[None, :], b_s, -float('inf'))
        b_p = safe_exp(b_s - b_lse[None, :])
        b_p = tl.where(o_k[:, None] <= o_q[None, :], b_p, 0)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    for i_s in range((i_t + 1) * BT, tl.cdiv(T, BS) * BS, BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, 0), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = safe_exp(b_s - b_lse[None, :])
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def simple_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    chunk_size: int = 128,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    
    # 根据GPU类型选择合适的块大小
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)
    assert NK == 1, "The key dimension can not be larger than 256"

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    
    # 使用更简单的grid，移除了v上的并行维度
    grid = (NT, B * HQ)
    simple_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def simple_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    simple_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def simple_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    scale: float = None,
    chunk_size: int = 128,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    BS = max(16, triton.next_power_of_2(T))
    BS = min(32, BS) if check_shared_mem('ampere') else min(16, BS)
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    NT = triton.cdiv(T, BT)

    delta = simple_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    
    # 移除了v上的并行维度
    grid = (NT, B * HQ)

    simple_attn_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    
    simple_attn_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    
    # 如果使用GQA，需要合并多个查询头到一个键值头
    if G > 1:
        dk = rearrange(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        dv = rearrange(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
        
    return dq, dk, dv


class SimpleAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale):
        ctx.dtype = q.dtype

        chunk_size = min(128, max(16, triton.next_power_of_2(q.shape[1])))

        o, lse = simple_attn_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = simple_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )

        return dq.to(q), dk.to(k), dv.to(v), None


def simple_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    head_first: bool = False
) -> torch.Tensor:
    """
    简化版的Flash Attention实现。

    Args:
        q (torch.Tensor):
            查询张量，形状为 `[B, T, HQ, K]` 如果 `head_first=False` 或 `[B, HQ, T, K]` 如果 `head_first=True`。
        k (torch.Tensor):
            键张量，形状为 `[B, T, H, K]` 如果 `head_first=False` 或 `[B, H, T, K]` 如果 `head_first=True`。
            如果 HQ 能被 H 整除，将应用 GQA。
        v (torch.Tensor):
            值张量，形状为 `[B, T, H, V]` 如果 `head_first=False` 或 `[B, H, T, V]` 如果 `head_first=True`。
        scale (float, optional):
            注意力分数的缩放因子。
            如果未提供，默认为 `1 / sqrt(K)`。默认: `None`。
        head_first (bool, optional):
            输入是否采用head-first格式。默认: `False`。

    Returns:
        o (torch.Tensor):
            输出张量，形状为 `[B, T, HQ, V]` 如果 `head_first=False` 或 `[B, HQ, T, V]` 如果 `head_first=True`。
    """
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v))
    
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = SimpleAttentionFunction.apply(q, k, v, scale)
    
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    
    return o


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from torch.profiler import profile, record_function, ProfilerActivity
    import triton.testing

    def reference_attention(q, k, v, scale=None):
        """PyTorch实现的标准注意力机制，用于结果验证"""
        if scale is None:
            scale = q.shape[-1] ** -0.5
        
        # [B, T, H, K] @ [B, T, H, K].transpose(-2, -1) -> [B, T, H, T]
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        
        # 生成一个上三角矩阵的掩码，以模拟causal attention
        causal_mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], 
                                            device=scores.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # [B, T, H, T] @ [B, T, H, V] -> [B, T, H, V]
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        return output

    def check_correctness(B=4, T=1024, H=4, K=64, V=64, dropout_p=0.0, dtype=torch.float16):
        """验证简化版flash attention的正确性"""
        print(f"测试配置: B={B}, T={T}, H={H}, K={K}, V={V}, dtype={dtype}")
        
        torch.manual_seed(42)
        q = torch.randn((B, T, H, K), dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn((B, T, H, K), dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn((B, T, H, V), dtype=dtype, device='cuda', requires_grad=True)
        
        scale = 1 / math.sqrt(K)
        
        # 运行我们的flash attention实现
        with torch.cuda.amp.autocast(dtype=dtype):
            out_fa = simple_attn(q, k, v, scale=scale)
            
        # 运行参考实现
        with torch.cuda.amp.autocast(dtype=dtype):
            out_ref = reference_attention(q, k, v, scale=scale)
        
        # 计算相对误差
        rel_error = ((out_fa - out_ref).abs() / (out_ref.abs() + 1e-5).clamp(min=1e-5)).mean()
        print(f"相对误差: {rel_error.item():.6f}")
        
        # 计算反向传播的正确性
        grad_out = torch.randn_like(out_fa)
        
        # 计算flash attention的梯度
        out_fa.backward(grad_out)
        
        q_grad_fa = q.grad.clone()
        k_grad_fa = k.grad.clone()
        v_grad_fa = v.grad.clone()
        
        # 重置梯度
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
        
        # 计算参考实现的梯度
        out_ref.backward(grad_out)
        
        # 计算梯度的相对误差
        q_rel_error = ((q_grad_fa - q.grad).abs() / (q.grad.abs() + 1e-5).clamp(min=1e-5)).mean()
        k_rel_error = ((k_grad_fa - k.grad).abs() / (k.grad.abs() + 1e-5).clamp(min=1e-5)).mean()
        v_rel_error = ((v_grad_fa - v.grad).abs() / (v.grad.abs() + 1e-5).clamp(min=1e-5)).mean()
        
        print(f"Q梯度相对误差: {q_rel_error.item():.6f}")
        print(f"K梯度相对误差: {k_rel_error.item():.6f}")
        print(f"V梯度相对误差: {v_rel_error.item():.6f}")
        
        return rel_error.item() < 1e-2 and q_rel_error.item() < 1e-2 and k_rel_error.item() < 1e-2 and v_rel_error.item() < 1e-2

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_len'],  # 参数名称
            x_vals=[128, 256, 512, 1024, 2048, 4096],  # 不同的序列长度
            line_arg='provider',  # 区分不同实现
            line_vals=['simple_flash', 'pytorch'],  # 我们测试的实现
            line_names=['简化Flash Attention', 'PyTorch'],  # 图例名称
            styles=[('blue', '-'), ('red', '--')],  # 线条样式
            ylabel='TFLOPS',  # y轴标签
            plot_name='attention-performance',  # 输出的图表名称
            args={'batch_size': 8, 'num_heads': 16, 'head_dim': 64},  # 默认参数
        )
    )
    def benchmark(seq_len, batch_size=8, num_heads=16, head_dim=64, provider='simple_flash', dtype=torch.float16):
        """对比不同实现的性能"""
        q = torch.randn((batch_size, seq_len, num_heads, head_dim), 
                         dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn((batch_size, seq_len, num_heads, head_dim), 
                         dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn((batch_size, seq_len, num_heads, head_dim), 
                         dtype=dtype, device='cuda', requires_grad=True)
        
        scale = 1 / math.sqrt(head_dim)
        
        if provider == 'simple_flash':
            # 预热GPU
            for _ in range(10):
                out = simple_attn(q, k, v, scale=scale)
                out.sum().backward()
            
            # 计算运行时间
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(100):
                out = simple_attn(q, k, v, scale=scale)
                out.sum().backward()
                
            end_event.record()
            torch.cuda.synchronize()
            
            runtime = start_event.elapsed_time(end_event) / 100
            
        elif provider == 'pytorch':
            # 预热GPU
            for _ in range(10):
                out = reference_attention(q, k, v, scale=scale)
                out.sum().backward()
            
            # 计算运行时间
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(100):
                out = reference_attention(q, k, v, scale=scale)
                out.sum().backward()
                
            end_event.record()
            torch.cuda.synchronize()
            
            runtime = start_event.elapsed_time(end_event) / 100
            
        # 计算FLOPS
        flops_per_matmul_1 = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        flops_per_matmul_2 = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        flops_per_iteration = flops_per_matmul_1 + flops_per_matmul_2
        tflops = flops_per_iteration / (runtime / 1000) / 1e12
        
        return tflops

    def benchmark_with_profiler(B=4, T=1024, H=16, K=64, V=64, dtype=torch.float16):
        """使用PyTorch profiler进行更详细的性能分析"""
        print(f"性能分析配置: B={B}, T={T}, H={H}, K={K}, V={V}, dtype={dtype}")
        
        q = torch.randn((B, T, H, K), dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn((B, T, H, K), dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn((B, T, H, V), dtype=dtype, device='cuda', requires_grad=True)
        
        scale = 1 / math.sqrt(K)
        
        # 预热
        for _ in range(10):
            out = simple_attn(q, k, v, scale=scale)
            out.mean().backward()
        
        # 使用profiler分析
        print("简化Flash Attention性能分析:")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True, with_stack=True) as prof:
            with record_function("simple_flash_attention"):
                out = simple_attn(q, k, v, scale=scale)
                out.mean().backward()
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # 重置梯度
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
        
        # 使用profiler分析PyTorch实现
        print("\nPyTorch原生Attention性能分析:")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True, with_stack=True) as prof:
            with record_function("pytorch_attention"):
                out = reference_attention(q, k, v, scale=scale)
                out.mean().backward()
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    # 运行测试和基准测试
    print("开始测试简化版Flash Attention...")
    
    # 测试正确性
    print("\n======== 正确性测试 ========")
    is_correct = check_correctness(B=2, T=512, H=8, K=64, V=64, dtype=torch.float16)
    if is_correct:
        print("✅ 所有测试通过！结果符合参考实现。")
    else:
        print("❌ 测试失败！结果与参考实现不符。")
    
    # 运行性能基准测试
    print("\n======== 性能基准测试 ========")
    benchmark.run(show_plots=True, print_data=True)
    
    # 详细性能分析
    print("\n======== 详细性能分析 ========")
    benchmark_with_profiler(B=2, T=1024, H=8, K=64, V=64, dtype=torch.float16)
    
    print("\n测试完成！")
