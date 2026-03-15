from __future__ import annotations

import argparse
import timeit
from pathlib import Path

import pandas as pd
import torch

from cs336_basics.model import scaled_dot_product_attention


device = torch.device("cuda")
compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)
SCRIPT_DIR = Path(__file__).resolve().parent


def is_oom_error(error: RuntimeError) -> bool:
    error_text = str(error).lower()
    return "out of memory" in error_text or "oom" in error_text


def synchronize() -> None:
    torch.cuda.synchronize(device)


def zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def benchmark_attention(Q, K, V, warmup=10, steps=100, use_compile=False):
    func = compiled_scaled_dot_product_attention if use_compile else scaled_dot_product_attention

    T = Q.shape[1]
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    for _ in range(warmup):
        attn_output = func(Q, K, V, causal_mask)
        synchronize()
        attn_output.backward(torch.ones_like(attn_output))
        synchronize()
        zero_grads(Q, K, V)

    forward_times = []
    backward_times = []
    pre_back_memories = []

    for _ in range(steps):
        forward_start = timeit.default_timer()
        attn_output = func(Q, K, V, causal_mask)
        synchronize()
        forward_times.append(timeit.default_timer() - forward_start)

        pre_back_memories.append(torch.cuda.memory_allocated(device) / (1024 ** 2))

        backward_start = timeit.default_timer()
        attn_output.backward(torch.ones_like(attn_output))
        synchronize()
        backward_times.append(timeit.default_timer() - backward_start)
        zero_grads(Q, K, V)

    return {
        "forward_time": sum(forward_times) / steps,
        "backward_time": sum(backward_times) / steps,
        "pre_back_memory": sum(pre_back_memories) / steps,
    }


def run_grid(use_compile: bool, batch_size: int, warmup: int, steps: int) -> pd.DataFrame:
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    d_model_list = [16, 32, 64, 128]

    rows = []
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            torch.cuda.empty_cache()
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

            try:
                result = benchmark_attention(Q, K, V, warmup=warmup, steps=steps, use_compile=use_compile)
                rows.append(
                    {
                        "implementation": "compiled" if use_compile else "eager",
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "forward_time": result["forward_time"],
                        "backward_time": result["backward_time"],
                        "pre_back_memory": result["pre_back_memory"],
                        "status": "ok",
                    }
                )
            except RuntimeError as error:
                torch.cuda.empty_cache()
                if not is_oom_error(error):
                    raise
                print(f"OOM for implementation={'compiled' if use_compile else 'eager'} d_model={d_model}, seq_len={seq_len}")
                rows.append(
                    {
                        "implementation": "compiled" if use_compile else "eager",
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "forward_time": None,
                        "backward_time": None,
                        "pre_back_memory": None,
                        "status": "oom",
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention and its torch.compile variant.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip_compile", action="store_true")
    parser.add_argument("--output_path", type=Path, default=SCRIPT_DIR / "benchmark_attn_results.md")
    parser.add_argument("--compiled_output_path", type=Path, default=SCRIPT_DIR / "benchmark_attn_compile_results.md")
    parser.add_argument("--combined_output_path", type=Path, default=SCRIPT_DIR / "benchmark_attn_comparison_results.md")
    args = parser.parse_args()

    eager_results = run_grid(use_compile=False, batch_size=args.batch_size, warmup=args.warmup, steps=args.steps)
    args.output_path.write_text(eager_results.to_markdown(index=False) + "\n")

    combined_frames = [eager_results]
    if not args.skip_compile:
        compiled_results = run_grid(use_compile=True, batch_size=args.batch_size, warmup=args.warmup, steps=args.steps)
        args.compiled_output_path.write_text(compiled_results.to_markdown(index=False) + "\n")
        combined_frames.append(compiled_results)

    combined_results = pd.concat(combined_frames, ignore_index=True)
    args.combined_output_path.write_text(combined_results.to_markdown(index=False) + "\n")
    print(combined_results.to_markdown(index=False))


if __name__ == "__main__":
    main()
