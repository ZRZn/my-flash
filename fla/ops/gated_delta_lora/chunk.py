# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.utils.pooling import mean_pooling
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    triton = None
    tl = None

BLOCK_K = 64
BLOCK_V = 64
NUM_WARPS_FWD = 4
NUM_WARPS_DQ = 4
NUM_WARPS_DH = 4


@triton.jit
def _fwd_kernel(
    Q, H, TOPI, OUT,
    # sizes
    BH: tl.constexpr, T: tl.constexpr, O: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    # strides
    stride_q_bh, stride_q_t, stride_q_k,
    stride_h_bh, stride_h_n, stride_h_k, stride_h_v,
    stride_top_bh, stride_top_t, stride_top_o,
    stride_out_bh, stride_out_t, stride_out_o, stride_out_v,
    # meta
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid0 = tl.program_id(0)  # over BH*T*O
    pid1 = tl.program_id(1)  # over V tiles

    # Map linear id -> (bh, t, o)
    TO = T * O
    bh = pid0 // TO
    rem = pid0 % TO
    t = rem // O
    o = rem % O

    v_offs = pid1 * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    # Load n = top_indices[bh, t, o]
    top_ptr = TOPI + bh * stride_top_bh + t * stride_top_t + o * stride_top_o
    n = tl.load(top_ptr, mask=True, other=0).to(tl.int32)

    # Base pointers
    q_ptr = Q + bh * stride_q_bh + t * stride_q_t
    out_ptr = OUT + bh * stride_out_bh + t * stride_out_t + o * stride_out_o + v_offs * stride_out_v
    h_ptr_base = H + bh * stride_h_bh + n * stride_h_n + v_offs * stride_h_v

    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + k_offs
        k_mask = kk < K
        q = tl.load(q_ptr + kk * stride_q_k, mask=k_mask, other=0.0).to(tl.float32)
        # h tile: [BLOCK_K, BLOCK_V] at (k,v)
        h_tile = tl.load(
            h_ptr_base + kk[:, None] * stride_h_k,
            mask=(k_mask[:, None] & v_mask[None, :]),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(h_tile * q[:, None], axis=0)
    # store
    acc = acc.to(tl.float32)  # keep in fp32 then cast as needed
    # cast to OUT dtype
    acc_cast = acc.to(Q.dtype.element_ty)
    tl.store(out_ptr, acc_cast, mask=v_mask)


@triton.jit
def _dq_kernel(
    DOUT, H, TOPI, DQ,
    # sizes
    BH: tl.constexpr, T: tl.constexpr, O: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    # strides
    stride_do_bh, stride_do_t, stride_do_o, stride_do_v,
    stride_h_bh, stride_h_n, stride_h_k, stride_h_v,
    stride_top_bh, stride_top_t, stride_top_o,
    stride_dq_bh, stride_dq_t, stride_dq_k,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    # Grid: (BH*T, ceil_div(K, BLOCK_K))
    pid0 = tl.program_id(0)  # over BH*T
    pid1 = tl.program_id(1)  # over K tiles

    bh = pid0 // T
    t = pid0 % T

    k_offs = pid1 * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    dq_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Loop over o and V in tiles
    for o in range(0, O):
        # n index for this (bh,t,o)
        top_ptr = TOPI + bh * stride_top_bh + t * stride_top_t + o * stride_top_o
        n = tl.load(top_ptr, mask=True, other=0).to(tl.int32)
        # Pointers
        dout_too_ptr = DOUT + bh * stride_do_bh + t * stride_do_t + o * stride_do_o
        h_base_ptr = H + bh * stride_h_bh + n * stride_h_n

        for v0 in range(0, V, BLOCK_V):
            v_offs = v0 + tl.arange(0, BLOCK_V)
            v_mask = v_offs < V
            dout_tile = tl.load(dout_too_ptr + v_offs * stride_do_v, mask=v_mask, other=0.0).to(tl.float32)
            h_tile = tl.load(
                h_base_ptr + k_offs[:, None] * stride_h_k + v_offs[None, :] * stride_h_v,
                mask=(k_mask[:, None] & v_mask[None, :]),
                other=0.0,
            ).to(tl.float32)
            dq_acc += tl.sum(h_tile * dout_tile[None, :], axis=1)

    dq_ptr = DQ + bh * stride_dq_bh + t * stride_dq_t + k_offs * stride_dq_k
    dq_cast = dq_acc.to(DQ.dtype.element_ty)
    tl.store(dq_ptr, dq_cast, mask=k_mask)


@triton.jit
def _dh_kernel(
    DOUT, Q, TOPI, DH,
    # sizes
    BH: tl.constexpr, T: tl.constexpr, O: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    # strides
    stride_do_bh, stride_do_t, stride_do_o, stride_do_v,
    stride_q_bh, stride_q_t, stride_q_k,
    stride_top_bh, stride_top_t, stride_top_o,
    stride_dh_bh, stride_dh_n, stride_dh_k, stride_dh_v,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    # Grid: (BH*T*O, ceil_div(V, BLOCK_V))
    pid0 = tl.program_id(0)  # over BH*T*O
    pid1 = tl.program_id(1)  # over V tiles

    TO = T * O
    bh = pid0 // TO
    rem = pid0 % TO
    t = rem // O
    o = rem % O

    v_offs = pid1 * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    # load n index
    top_ptr = TOPI + bh * stride_top_bh + t * stride_top_t + o * stride_top_o
    n = tl.load(top_ptr, mask=True, other=0).to(tl.int32)

    dout_ptr = DOUT + bh * stride_do_bh + t * stride_do_t + o * stride_do_o + v_offs * stride_do_v
    q_ptr = Q + bh * stride_q_bh + t * stride_q_t
    dh_base_ptr = DH + bh * stride_dh_bh + n * stride_dh_n + v_offs * stride_dh_v

    # Outer-product accumulation with atomics: for each K tile, atomic_add into DH
    k_offs = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + k_offs
        k_mask = kk < K
        q = tl.load(q_ptr + kk * stride_q_k, mask=k_mask, other=0.0)
        dout = tl.load(dout_ptr, mask=v_mask, other=0.0)
        outer = (q[:, None].to(tl.float32)) * (dout[None, :].to(tl.float32))
        # cast to DH dtype before atomics
        # outer_cast = outer.to(DH.dtype.element_ty)
        outer_cast = outer.to(tl.float32)
        tl.atomic_add(
            dh_base_ptr + kk[:, None] * stride_dh_k,
            outer_cast,
            mask=(k_mask[:, None] & v_mask[None, :]),
        )


# =============================
# PyTorch Autograd Wrapper
# =============================

class _TopkLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, h: torch.Tensor, top_indices: torch.Tensor):
        if triton is None:
            raise RuntimeError("Triton is not available in this environment.")

        B, H, T, K = q.shape
        _, _, N, _, V = h.shape
        O = top_indices.shape[-1]
        BH = B * H

        # Flatten BH leading dims for simpler stride math
        q_bh = q.reshape(BH, T, K)
        h_bh = h.reshape(BH, N, K, V)
        top_bh = top_indices.reshape(BH, T, O)
        out = torch.empty((BH, T, O, V), device=q.device, dtype=q.dtype)

        grid = (BH * T * O, triton.cdiv(V, BLOCK_V))
        _fwd_kernel[grid](
            q_bh, h_bh, top_bh, out,
            # sizes
            BH, T, O, K, V,
            # strides (note: PyTorch strides are in elements)
            q_bh.stride(0), q_bh.stride(1), q_bh.stride(2),
            h_bh.stride(0), h_bh.stride(1), h_bh.stride(2), h_bh.stride(3),
            top_bh.stride(0), top_bh.stride(1), top_bh.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=NUM_WARPS_FWD,
        )

        ctx.save_for_backward(q_bh, h_bh, top_bh)
        ctx.meta = (BH, T, O, K, V)
        return out.reshape(B, H, T, O, V)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_bh, h_bh, top_bh = ctx.saved_tensors
        BH, T, O, K, V = ctx.meta

        grad_out_bh = grad_out.reshape(BH, T, O, V).contiguous()

        # dq
        dq = torch.empty((BH, T, K), device=grad_out.device, dtype=grad_out.dtype)
        grid_dq = (BH * T, triton.cdiv(K, BLOCK_K))
        _dq_kernel[grid_dq](
            grad_out_bh, h_bh, top_bh, dq,
            BH, T, O, K, V,
            grad_out_bh.stride(0), grad_out_bh.stride(1), grad_out_bh.stride(2), grad_out_bh.stride(3),
            h_bh.stride(0), h_bh.stride(1), h_bh.stride(2), h_bh.stride(3),
            top_bh.stride(0), top_bh.stride(1), top_bh.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=NUM_WARPS_DQ,
        )

        # dh (atomic adds)
        dh = torch.zeros_like(h_bh, dtype=torch.float32)
        grid_dh = (BH * T * O, triton.cdiv(V, BLOCK_V))
        _dh_kernel[grid_dh](
            grad_out_bh, q_bh, top_bh, dh,
            BH, T, O, K, V,
            grad_out_bh.stride(0), grad_out_bh.stride(1), grad_out_bh.stride(2), grad_out_bh.stride(3),
            q_bh.stride(0), q_bh.stride(1), q_bh.stride(2),
            top_bh.stride(0), top_bh.stride(1), top_bh.stride(2),
            dh.stride(0), dh.stride(1), dh.stride(2), dh.stride(3),
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=NUM_WARPS_DH,
        )

        # Reshape grads back to original shapes
        B = grad_out.shape[0]
        H = grad_out.shape[1]
        dq = dq.reshape(B, H, T, K)
        # dh = dh.reshape(B, H, -1, K, V)
        dh = dh.to(h_bh.dtype).reshape(B, H, -1, K, V)

        # No grad for top_indices
        return dq, dh, None


def topk_lora_triton(q: torch.Tensor, h: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
    """User-facing functional API."""
    return _TopkLoRAFunction.apply(q, h, top_indices)



def chunk_gated_delta_lora_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u   is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    chunk_size = 64
    B, T, H, K = k.shape
    V = u.shape[-1]
    N = T // 64
    h_independent = torch.einsum("bnchk, bnchv -> bnhkv", k[:, :, :, int(K * 0.875):].view(B, N, chunk_size, H, int(K * 0.125)), u[:, :, :, int(V * 0.875):].view(B, N, chunk_size, H, int(V * 0.125)))
    # print("h_independent = ", h_independent.shape)

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return g, o, A, h_independent, final_state


def chunk_gated_delta_lora_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dh_lora: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    B, T, H, K = k.shape
    V = u.shape[-1]
    N = T // 64
    chunk_size = 64
    if dh_lora is not None:
        # du_from_h_prime = k @ dh_independent.transpose(-2, -1)
        # print("dh_lora = ", dh_lora.shape)
        k_lora = k[:, :, :, int(K * 0.875):].view(B, N, chunk_size, H, int(K * 0.125))
        u_lora = u[:, :, :, int(V * 0.875):].view(B, N, chunk_size, H, int(V * 0.125))
        dk_lora = torch.einsum("bnhkv, bnchv -> bnchk", dh_lora, u_lora).reshape(B, T, H, int(K * 0.125))
        du_lora = torch.einsum("bnhkv, bnchk -> bnchv", dh_lora, k_lora).reshape(B, T, H, int(V * 0.125))
    
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # print("do = ", torch.mean(do[..., int(V * 0.875):]))
    # print("dh1 = ", torch.mean(dh[..., int(K * 0.875):, :]))
    # print("dh2 = ", torch.mean(dh[..., int(V * 0.875):]))
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if dh_lora is not None:
        # Accumulate gradient for u (represented by dv).
        # This MUST be done before calling prepare_wy_repr_bwd, which consumes dv (as du).
        # dv.add_(du_from_h_prime)
        # print("dv = ", torch.mean(dv[..., int(V * 0.875):]))
        dv[..., int(V * 0.875):] += du_lora
        # Accumulate gradient for k.
        # This can be done here or later, but doing it now keeps things tidy.
        # dk.add_(dk_from_h_prime)
        # print("dk = ", torch.mean(dk[..., int(K * 0.875):]))
        # print("dq = ", torch.mean(dq[..., int(K * 0.875):]))
        dk[..., int(K * 0.875):] += dk_lora

    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaLoraFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False
    ):

        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, h, final_state = chunk_gated_delta_lora_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        # V = v.shape[-1]
        # head_v_ori = V - V//8
        # print("o1 = ", torch.mean(o[:, :, :, :head_v_ori]))
        # print("o2 = ", torch.mean(o[:, :, :, head_v_ori:]))
        return o.to(q.dtype), final_state, h

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
        dh_lora: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        # V = v.shape[-1]
        # head_v_ori = V - V//8
        # head_k_ori = head_v_ori//2
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_lora_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dh_lora=dh_lora,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None


@torch.compile
def old_lora_topk(
    q,                 # [B, T, H, K_total]
    k_cmp,             # [B, H, N, K_lora]  —— 已做mean_pooling后的K
    h,                 # [B, H, N, K_lora, V]
    *,
    top_k: int,
    chunk_size: int,
    head_k_ori: int = 0,
    block_t: int = 0, 
    use_triton: bool = False,    
):
    B, T, H, _ = q.shape
    _, _, N, K = k_cmp.shape
    V = h.shape[-1]
    q_lora = q[:, :, :, head_k_ori:].transpose(1, 2)   #q_lora [B, H, T, k_lora]
    scores = torch.matmul(q_lora, k_cmp.transpose(2, 3))    # scores [B, H, T, N]
    t_indices = torch.arange(T, device=q.device).view(T, 1)
    n_indices = torch.arange(N, device=q.device).view(1, N)
    chunk_indices_for_t = t_indices // chunk_size
    # causal_mask 的形状是 [T, N], 当 n < (t // C) 时，值为 True，表示允许访问
    causal_mask = n_indices < chunk_indices_for_t
    masked_scores = scores.masked_fill_(~causal_mask, -torch.inf)
    top_scores, top_indices = torch.topk(masked_scores, k=top_k, dim=-1)

    # indices_for_gather = top_indices.view(B, H, T, top_k, 1, 1).expand(-1, -1, -1, -1, K, V)
    # h_expanded = h.unsqueeze(2).expand(-1, -1, T, -1, -1, -1)
    # S = torch.gather(h_expanded, dim=3, index=indices_for_gather)  # S [B, H, T, topk, K, V]

    
    if use_triton:
        o_lora_all = topk_lora_triton(
            q_lora,    # [B, H, T, K]
            h,         # [B, H, N, K, V]
            top_indices  # [B, H, T, top_k]
        )   #[B, H, T, topk, V]

    else:
        b_idx = torch.arange(B, device=h.device)[:, None, None, None]
        h_idx = torch.arange(H, device=h.device)[None, :, None, None]
        # 分块计算以节省显存
        if block_t <= 0:
            S = h[b_idx, h_idx, top_indices]   #[B, H, T, top_k, K, V]
            o_lora_all = torch.einsum('bhtk, bhtokv -> bhtov', q_lora, S)  #[B, H, T, topk, V]
        else:
            o_lora_all = torch.empty(B, H, T, top_k, V, device=h.device, dtype=q.dtype)
            for t_start in range(0, T, block_t):
                t_end = min(t_start + block_t, T)
                q_chunk = q_lora[:, :, t_start:t_end]
                top_indices_chunk = top_indices[:, :, t_start:t_end]
                S_chunk = h[b_idx, h_idx, top_indices_chunk]
                o_lora_chunk = torch.einsum('bhtk, bhtokv -> bhtov', q_chunk, S_chunk)
                o_lora_all[:, :, t_start:t_end] = o_lora_chunk

    # q_lora_expand = q_lora.unsqueeze(3).unsqueeze(-2).expand(-1, -1, -1, top_k, -1, -1)
    # o_lora_all = torch.matmul(q_lora_expand, S).squeeze(-2)
    weights = F.softmax(top_scores, dim=-1).unsqueeze(-1)  # weights [B, H, T, topk, 1]
    weights = torch.nan_to_num(weights, nan=0.0)
    o_lora = torch.sum(o_lora_all * weights, dim=-2)  # [B, T, H, v_lora]
    return o_lora.transpose(1, 2)


@torch.compiler.disable(recursive=False)
def chunk_gated_delta_lora(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    head_k_ori: int,
    head_v_ori: int,
    top_k: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state, h = ChunkGatedDeltaLoraFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel
    )

    # 计算lora逻辑
    h = h.transpose(1, 2)
    # h = h[:, :, :, head_k_ori:, head_v_ori:]
    chunk_size = 64
    B, T, H, _ = q.shape
    N = T//chunk_size
    _, _, _, K, V = h.shape
    k_sim = k[:, :, :, head_k_ori:].transpose(1, 2)
    k_cmp = mean_pooling(k_sim, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=True)
    # 旧逻辑
    o_lora = old_lora_topk(q, k_cmp, h, top_k=top_k, chunk_size=chunk_size, head_k_ori=head_k_ori, block_t=0, use_triton=True)
    # 新逻辑
    # o_lora = chunk_topk_lora_pytorch(q, k_cmp, h, top_k=top_k, chunk_size=chunk_size, head_k_ori=head_k_ori, block_t=256)
    o_final = torch.concat([o[:, :, :, :head_v_ori], o_lora], dim=-1)
    return o_final, final_state
