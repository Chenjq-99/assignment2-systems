import torch
import triton
import triton.language as tl

def _flash_backward_impl(Q, K, V, O, dO, L, is_causal):
    *batch_dims, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    value_dim = V.shape[-1]
    scale = d ** -0.5

    acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype

    q = Q.reshape(-1, n_queries, d).to(acc_dtype)
    k = K.reshape(-1, n_keys, d).to(acc_dtype)
    v = V.reshape(-1, n_keys, value_dim).to(acc_dtype)
    o = O.reshape(-1, n_queries, value_dim).to(acc_dtype)
    do = dO.reshape(-1, n_queries, value_dim).to(acc_dtype)
    l = L.reshape(-1, n_queries).to(acc_dtype)

    s = torch.matmul(q, k.transpose(-1, -2)) * scale

    if is_causal:
        q_indices = torch.arange(n_queries, device=Q.device)
        k_indices = torch.arange(n_keys, device=Q.device)
        causal_mask = q_indices[:, None] >= k_indices[None, :]
        s = torch.where(causal_mask, s, -1e6)

    p = torch.exp(s - l.unsqueeze(-1))

    d_vec = torch.sum(do * o, dim=-1)  # (batch, n_queries)
    d_v_grad = torch.matmul(p.transpose(-1, -2), do)  # (batch, n_keys, d_v)
    d_p = torch.matmul(do, v.transpose(-1, -2))  # (batch, n_queries, n_keys)
    d_s = p * (d_p - d_vec.unsqueeze(-1))  # (batch, n_queries, n_keys)
    d_q = torch.matmul(d_s, k) * scale  # (batch, n_queries, d)
    d_k = torch.matmul(d_s.transpose(-1, -2), q) * scale  # (batch, n_keys, d)

    d_q = d_q.reshape(*batch_dims, n_queries, d).to(Q.dtype)
    d_k = d_k.reshape(*batch_dims, n_keys, d).to(K.dtype)
    d_v_grad = d_v_grad.reshape(*batch_dims, n_keys, value_dim).to(V.dtype)

    return d_q, d_k, d_v_grad

flash_backward_compiled = torch.compile(_flash_backward_impl)

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        *batch_dims, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        d_v = V.shape[-1]
        
        q = Q.reshape(-1, n_queries, d)
        k = K.reshape(-1, n_keys, d)
        v = V.reshape(-1, n_keys, d_v)

        batch = q.shape[0]
        scale = d ** -0.5
        acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype

        block_q = 16
        block_k = 16

        O = torch.empty((batch, n_queries, d_v), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch, n_queries), device=Q.device, dtype=acc_dtype)

        for q_start in range(0, n_queries, block_q):
            q_end = min(q_start + block_q, n_queries)
            q_i = q[..., q_start:q_end, :].to(acc_dtype)

            m_i = torch.full((batch, q_end - q_start), -float('inf'), device=Q.device, dtype=acc_dtype)
            l_i = torch.zeros((batch, q_end - q_start), device=Q.device, dtype=acc_dtype)
            o_i = torch.zeros((batch, q_end - q_start, d_v), device=Q.device, dtype=acc_dtype)

            for k_start in range(0, n_keys, block_k):
                k_end = min(k_start + block_k, n_keys)
                k_j = k[..., k_start:k_end, :].to(acc_dtype)
                v_j = v[..., k_start:k_end, :].to(acc_dtype)

                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2)) * scale

                if is_causal:
                    q_indices = torch.arange(q_start, q_end, device=Q.device)
                    k_indices = torch.arange(k_start, k_end, device=Q.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    s_ij = torch.where(causal_mask, s_ij, -1e6)
                
                m_ij = s_ij.amax(dim=-1)
                p_ij = torch.exp(s_ij - m_ij.unsqueeze(-1))
                l_ij = p_ij.sum(dim=-1)

                m_new = torch.maximum(m_i, m_ij)
        
                alpha = torch.exp(m_i - m_new)
                beta = torch.exp(m_ij - m_new)

                o_i = o_i * alpha.unsqueeze(-1) + torch.matmul(p_ij, v_j) * beta.unsqueeze(-1)
                l_i = l_i * alpha + l_ij * beta
                m_i = m_new

            O[:, q_start:q_end, :] = (o_i / l_i.unsqueeze(-1)).to(Q.dtype)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O.reshape(*batch_dims, n_queries, d_v)

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
    
        
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # Flash attention forward pass
    q = tl.load(Q_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_start = query_tile_index * Q_TILE_SIZE
    q_offsets = q_start + tl.arange(0, Q_TILE_SIZE)
    k_offsets = tl.arange(0, K_TILE_SIZE)

    curr_k_ptr = K_block_ptr
    curr_v_ptr = V_block_ptr

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        k_j = tl.load(curr_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        v_j = tl.load(curr_v_ptr, boundary_check=(0, 1))

        s_ij = tl.dot(q, tl.trans(k_j)) * scale

        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] >= (k_start + k_offsets[None, :])
            s_ij = tl.where(causal_mask, s_ij, -1e6)

        m_ij = tl.max(s_ij, axis=1)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_ij = tl.sum(p_ij, axis=1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        p_ij = p_ij.to(v_j.dtype)
        pv_ij = tl.dot(p_ij, v_j, acc=o_i)

        o_i = o_i * alpha[:, None] + (pv_ij - o_i) * beta[:, None]
        l_i = l_i * alpha + l_ij * beta
        m_i  = m_new

        curr_k_ptr = tl.advance(curr_k_ptr, (K_TILE_SIZE, 0))
        curr_v_ptr = tl.advance(curr_v_ptr, (K_TILE_SIZE, 0))

    o_i = o_i / l_i[:, None]
    lse_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, lse_i[:, None], boundary_check=(0, 1))

@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_base_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_base_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_base_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_base_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    k_j = tl.load(K_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    v_j = tl.load(V_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    
    dk_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dv_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    curr_q_ptr = Q_base_ptr
    curr_o_ptr = O_base_ptr
    curr_do_ptr = dO_base_ptr
    curr_l_ptr = L_base_ptr

    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        q_i = tl.load(curr_q_ptr, boundary_check=(0, 1)).to(tl.float32)
        o_i = tl.load(curr_o_ptr, boundary_check=(0, 1)).to(tl.float32)
        do_i = tl.load(curr_do_ptr, boundary_check=(0, 1)).to(tl.float32)
        l_i = tl.load(curr_l_ptr, boundary_check=(0, 1)).to(tl.float32)

        s_ij = tl.dot(q_i.to(k_j.dtype), tl.trans(k_j)) * scale
        if IS_CAUSAL:
            q_offsets = q_start + tl.arange(0, Q_TILE_SIZE)
            k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s_ij = tl.where(q_offsets[:, None] >= k_offsets[None, :], s_ij, -1e10)
            
        p_ij = tl.exp(s_ij - l_i)

        dv_j += tl.dot(tl.trans(p_ij.to(do_i.dtype)), do_i)
        
        dp_ij = tl.dot(do_i.to(v_j.dtype), tl.trans(v_j))
        d_i = tl.sum(o_i * do_i, axis=1)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * scale

        dk_j += tl.dot(tl.trans(ds_ij.to(q_i.dtype)), q_i)

        dq_step = tl.dot(ds_ij.to(k_j.dtype), k_j)
        
        off_q = q_start + tl.arange(0, Q_TILE_SIZE)
        off_d = tl.arange(0, D)
        dq_ptr_tile = dQ_ptr + batch_index * stride_qb + off_q[:, None] * stride_qq + off_d[None, :] * stride_qd
        tl.atomic_add(dq_ptr_tile, dq_step, mask=(off_q[:, None] < N_QUERIES) & (off_d[None, :] < D))

        curr_q_ptr = tl.advance(curr_q_ptr, (Q_TILE_SIZE, 0))
        curr_o_ptr = tl.advance(curr_o_ptr, (Q_TILE_SIZE, 0))
        curr_do_ptr = tl.advance(curr_do_ptr, (Q_TILE_SIZE, 0))
        curr_l_ptr = tl.advance(curr_l_ptr, (Q_TILE_SIZE, 0))

    tl.store(dK_block_ptr, dk_j.to(dK_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_j.to(dV_ptr.dtype.element_ty), boundary_check=(0, 1))

class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        *batch_dims, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        d_v = V.shape[-1]

        q = Q.reshape(-1, n_queries, d)
        k = K.reshape(-1, n_keys, d)
        v = V.reshape(-1, n_keys, d_v)

        batch = q.shape[0]
        scale = d ** -0.5

        O = torch.empty((batch, n_queries, d_v), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch, n_queries), device=Q.device, dtype=torch.float32)

        block_q = 16
        block_k = 16

        grid = (triton.cdiv(n_queries, block_q), batch)

        flash_fwd_kernel[grid](
            q, k, v,
            O, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=block_q,
            K_TILE_SIZE=block_k,
            IS_CAUSAL=is_causal
        )

        O_out = O.reshape(*batch_dims, n_queries, d_v)
        L_out = L.reshape(*batch_dims, n_queries)

        ctx.save_for_backward(L_out, Q, K, V, O_out)
        ctx.is_causal = is_causal
        return O_out

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        *batch_dims, n_keys, d = K.shape
        n_queries = Q.shape[-2]
        d_v = V.shape[-1]

        q = Q.reshape(-1, n_queries, d)
        k = K.reshape(-1, n_keys, d)
        v = V.reshape(-1, n_keys, d_v)

        batch = q.shape[0]
        scale = d ** -0.5

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        block_q = 16
        block_k = 16

        grid = (triton.cdiv(n_keys, block_k), batch)
        flash_bwd_kernel[grid](
            q, k, v, O, dO, L,
            dQ, dK, dV,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=block_q,
            K_TILE_SIZE=block_k,
            IS_CAUSAL=is_causal
        )

        dQ = dQ.reshape(*batch_dims, n_queries, d)
        dK = dK.reshape(*batch_dims, n_keys, d)
        dV = dV.reshape(*batch_dims, n_keys, d_v)

        return dQ, dK, dV, None
        