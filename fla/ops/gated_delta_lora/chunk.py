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
def chunk_topk_lora_pytorch(
    q,                 # [B, T, H, K_total]
    k_cmp,             # [B, H, N, K_lora]  —— 已做mean_pooling后的K
    h,                 # [B, H, N, K_lora, V]
    *,
    top_k: int,
    chunk_size: int,
    head_k_ori: int = 0,
    block_t: int = 256,     # T方向tile
):
    """
    返回：o_lora [B, T, H, V]
    与你原逻辑等价：q_lora对k_cmp做top-k，取对应chunk的h，并计算q@h后按softmax(topk scores)加权求和。
    """
    device = q.device
    dtype_q = q.dtype

    # 预处理到 [BH, ...] 的扁平视图，减少花式广播和维度转置成本
    B, T, H, K_total = q.shape
    BH = B * H
    # 只取 LoRA 部分
    q_lora = q[:, :, :, head_k_ori:].transpose(1, 2).contiguous()        # [B, H, T, K]
    k_cmp  = k_cmp.contiguous()                                           # [B, H, N, K]
    h      = h.contiguous()                                               # [B, H, N, K, V]

    _, _, N, K = k_cmp.shape
    V = h.shape[-1]

    q_bh = q_lora.reshape(BH, T, K)               # [BH, T, K]
    k_bh = k_cmp.reshape(BH, N, K)                # [BH, N, K]
    h_bh = h.reshape(BH, N, K, V)                 # [BH, N, K, V]

    o_bh = torch.zeros((BH, T, V), device=device, dtype=dtype_q)

    for t0 in range(0, T, block_t):
        t1 = min(T, t0 + block_t)
        tb = t1 - t0
        q_blk = q_bh[:, t0:t1, :]                                  # [BH, tb, K]

        # 计算 Top-K（两种方式：一次性 or N方向流式）
        # 一次性算 [BH, tb, N] 分数（适合N中等）
        # scores = q_blk @ k_bh^T
        scores = torch.bmm(q_blk, k_bh.transpose(1, 2))        # [BH, tb, N]

        # 微型因果mask：[tb, N]
        t_idx = torch.arange(t0, t1, device=device).view(1, tb, 1)   # 真实时间步
        n_idx = torch.arange(N, device=device).view(1, 1, N)
        causal_mask = (n_idx < (t_idx // chunk_size))                # True=允许
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        top_scores, top_idx = torch.topk(scores, k=top_k, dim=-1)    # [BH, tb, topk]
        

        # softmax over top-k（对 -inf 行会得到 NaN，随后设为 0）
        w = F.softmax(top_scores.to(torch.float32), dim=-1).to(q_blk.dtype)     # [BH, tb, topk]
        w = torch.nan_to_num(w, nan=0.0)

        # 累加输出：按 j 循环避免物化 [BH,tb,topk,K,V] 或 [BH,tb,topk,V]
        out_blk = torch.zeros((BH, tb, V), device=device, dtype=q_blk.dtype)

        # 预生成 [BH, tb] 的批次索引，做高级索引时少分配
        bh_arange = torch.arange(BH, device=device).view(BH, 1).expand(BH, tb)

        for j in range(top_k):
            idx_j = top_idx[:, :, j]                                       # [BH, tb] in [0..N)
            # h_sel: [BH, tb, K, V]
            h_sel = h_bh[bh_arange, idx_j]

            # 计算 (q_blk @ h_sel) -> [BH, tb, V]
            # 先把维度摊平成 batched bmm
            q_lin = q_blk.reshape(BH * tb, K).unsqueeze(1)                  # [BH*tb, 1, K]
            h_lin = h_sel.reshape(BH * tb, K, V)                            # [BH*tb, K, V]
            prod = torch.bmm(q_lin, h_lin).squeeze(1).reshape(BH, tb, V)    # [BH, tb, V]

            out_blk = out_blk + prod * w[:, :, j].unsqueeze(-1)             # 加权累加

        o_bh[:, t0:t1, :] = out_blk

    # 还原回 [B, T, H, V]
    o = o_bh.view(B, H, T, V).transpose(1, 2).contiguous()
    return o


@torch.compile
def old_lora_topk(
    q,                 # [B, T, H, K_total]
    k_cmp,             # [B, H, N, K_lora]  —— 已做mean_pooling后的K
    h,                 # [B, H, N, K_lora, V]
    *,
    top_k: int,
    chunk_size: int,
    head_k_ori: int = 0):
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

    b_idx = torch.arange(B, device=h.device)[:, None, None, None]
    h_idx = torch.arange(H, device=h.device)[None, :, None, None]
    S = h[b_idx, h_idx, top_indices]   #[B, H, T, top_k, K, V]

    o_lora_all = torch.einsum('bhtk, bhtokv -> bhtov', q_lora, S)  #[B, H, T, topk, V]
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
    o_lora = old_lora_topk(q, k_cmp, h, top_k=top_k, chunk_size=chunk_size, head_k_ori=head_k_ori)
    # 新逻辑
    # o_lora = chunk_topk_lora_pytorch(q, k_cmp, h, top_k=top_k, chunk_size=chunk_size, head_k_ori=head_k_ori, block_t=256)
    o_final = torch.concat([o[:, :, :, :head_v_ori], o_lora], dim=-1)
    return o_final, final_state
