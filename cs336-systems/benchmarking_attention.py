import torch
import pandas as pd
import timeit
from cs336_basics.model import scaled_dot_product_attention

device = torch.device("cuda")

compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

def benchmark_attention(Q, K, V, warmup=10, steps=100, torch_compile=False):
    if torch_compile:
        func = compiled_scaled_dot_product_attention
    else:
        func = scaled_dot_product_attention

    T = Q.shape[1]
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    for _ in range(warmup):
        attn_output = func(Q, K, V, causal_mask)
        torch.cuda.synchronize()

        attn_output.backward(torch.ones_like(attn_output))
        torch.cuda.synchronize()

        for param in [Q, K, V]:
            param.grad.zero_()

        torch.cuda.empty_cache()

    forward_time = 0
    backward_time = 0
    pre_back_memory = 0

    for _ in range(steps):
        # measure forward time
        forward_start = timeit.default_timer()
        attn_output = func(Q, K, V, causal_mask)
        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        forward_time += forward_end - forward_start

        # measure memory usage
        memory = torch.cuda.memory_allocated()
        pre_back_memory += memory / (1024 ** 2)

        # measure backward time
        backward_start = timeit.default_timer()
        attn_output.backward(torch.ones_like(attn_output))
        torch.cuda.synchronize()
        backward_end = timeit.default_timer()
        backward_time += backward_end - backward_start

        for param in [Q, K, V]:
            param.grad.zero_()

        torch.cuda.empty_cache()

    forward_time /= steps
    backward_time /= steps
    pre_back_memory /= steps

    return forward_time, backward_time, pre_back_memory


def main():
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    d_model_list = [16, 32, 64, 128]
    batch_size = 8

    rows = []

    for d_model in d_model_list:
        for seq_len in seq_len_list:
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

            try:
                fwd, bwd, mem = benchmark_attention(Q, K, V, torch_compile=True)
                rows.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_time": fwd,
                    "backward_time": bwd,
                    "pre_back_memory": mem
                })
            except RuntimeError as e:
                print(f"OOM at d_model={d_model}, seq_len={seq_len}")
                continue

    results = pd.DataFrame(rows)
    results.to_markdown("benchmark_attn_compile_results.md")


if __name__ == "__main__":
    main()
