import torch


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # The assignment says you can ignore causality for this milestone.
        del is_causal

        *batch_dims, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        d_v = V.shape[-1]

        if K.shape[:-2] != tuple(batch_dims) or V.shape[:-2] != tuple(batch_dims):
            raise ValueError("Q, K, and V must have matching batch dimensions.")
        if K.shape[-1] != d:
            raise ValueError("Q and K must have the same head dimension.")
        if V.shape[-2] != n_keys:
            raise ValueError("K and V must have the same sequence length.")

        # Flatten arbitrary batch dims so the implementation works for
        # both [B, N, D] and [..., N, D].
        q = Q.reshape(-1, n_queries, d)
        k = K.reshape(-1, n_keys, d)
        v = V.reshape(-1, n_keys, d_v)

        batch = q.shape[0]
        scale = d ** -0.5

        # Use fp32 accumulation for stability when inputs are low precision.
        acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype

        # Tile sizes only need to be at least 16 x 16.
        B_q = 64
        B_k = 64

        O = torch.empty((batch, n_queries, d_v), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch, n_queries), device=Q.device, dtype=acc_dtype)

        q_acc = q.to(acc_dtype)
        k_acc = k.to(acc_dtype)
        v_acc = v.to(acc_dtype)

        for q_start in range(0, n_queries, B_q):
            q_end = min(q_start + B_q, n_queries)
            q_i = q_acc[:, q_start:q_end, :]

            m_i = torch.full(
                (batch, q_end - q_start),
                -torch.inf,
                device=Q.device,
                dtype=acc_dtype,
            )
            l_i = torch.zeros(
                (batch, q_end - q_start),
                device=Q.device,
                dtype=acc_dtype,
            )
            o_i = torch.zeros(
                (batch, q_end - q_start, d_v),
                device=Q.device,
                dtype=acc_dtype,
            )

            for k_start in range(0, n_keys, B_k):
                k_end = min(k_start + B_k, n_keys)
                k_j = k_acc[:, k_start:k_end, :]
                v_j = v_acc[:, k_start:k_end, :]

                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2)) * scale
                m_ij = s_ij.amax(dim=-1)
                p_ij = torch.exp(s_ij - m_ij.unsqueeze(-1))
                l_ij = p_ij.sum(dim=-1)
                pv_ij = torch.matmul(p_ij, v_j)

                m_new = torch.maximum(m_i, m_ij)
                alpha = torch.exp(m_i - m_new)
                beta = torch.exp(m_ij - m_new)

                l_new = alpha * l_i + beta * l_ij
                o_i = (
                    o_i * (alpha * l_i).unsqueeze(-1)
                    + pv_ij * beta.unsqueeze(-1)
                ) / l_new.unsqueeze(-1)

                m_i = m_new
                l_i = l_new

            O[:, q_start:q_end, :] = o_i.to(Q.dtype)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        O = O.reshape(*batch_dims, n_queries, d_v)
        L = L.reshape(*batch_dims, n_queries)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("FlashAttention2.backward is not implemented yet.")