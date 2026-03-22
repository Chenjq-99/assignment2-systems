from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import triton.testing

from cs336_systems.flash_attention import TritonFlashAttention2


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
SEQ_LENS = [2**power for power in range(7, 17)]
EMBED_DIMS = [2**power for power in range(4, 8)]
BATCH_SIZE = 1
IS_CAUSAL = True
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "experiment_results" / "benchmark_flash_attention_results.md"
DEFAULT_CSV_PATH = SCRIPT_DIR / "experiment_results" / "benchmark_flash_attention_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton FlashAttention-2 against regular PyTorch attention on a single CUDA device."
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    return parser.parse_args()


def is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "oom" in message


def regular_pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_indices = torch.arange(n_queries, device=q.device)[:, None]
        k_indices = torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(q_indices >= k_indices, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def make_base_inputs(seq_len: int, embed_dim: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    q = torch.randn(BATCH_SIZE, seq_len, embed_dim, device=device, dtype=dtype)
    k = torch.randn(BATCH_SIZE, seq_len, embed_dim, device=device, dtype=dtype)
    v = torch.randn(BATCH_SIZE, seq_len, embed_dim, device=device, dtype=dtype)
    d_out = torch.randn(BATCH_SIZE, seq_len, embed_dim, device=device, dtype=dtype)
    return q, k, v, d_out


def clone_leaf_tensors(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [tensor.clone().requires_grad_(True) for tensor in tensors]


def benchmark_forward(
    implementation,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    rep: int,
) -> float:
    return triton.testing.do_bench(
        lambda: implementation(q, k, v, IS_CAUSAL),
        warmup=warmup,
        rep=rep,
        return_mode="mean",
    )


def benchmark_backward(
    implementation,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_out: torch.Tensor,
    warmup: int,
    rep: int,
) -> float:
    output = implementation(q, k, v, IS_CAUSAL)
    return triton.testing.do_bench(
        lambda: output.backward(d_out, retain_graph=True),
        warmup=warmup,
        rep=rep,
        grad_to_none=[q, k, v],
        return_mode="mean",
    )


def benchmark_forward_backward(
    implementation,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_out: torch.Tensor,
    warmup: int,
    rep: int,
) -> float:
    return triton.testing.do_bench(
        lambda: implementation(q, k, v, IS_CAUSAL).backward(d_out),
        warmup=warmup,
        rep=rep,
        grad_to_none=[q, k, v],
        return_mode="mean",
    )


def benchmark_implementation(
    name: str,
    implementation,
    seq_len: int,
    embed_dim: int,
    dtype_name: str,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    rep: int,
) -> dict[str, object]:
    base_q, base_k, base_v, base_d_out = make_base_inputs(seq_len, embed_dim, dtype, device)
    row: dict[str, object] = {
        "implementation": name,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "dtype": dtype_name,
        "batch_size": BATCH_SIZE,
        "is_causal": IS_CAUSAL,
        "forward_ms": None,
        "backward_ms": None,
        "forward_backward_ms": None,
        "status": "ok",
    }

    try:
        q_fwd, k_fwd, v_fwd = clone_leaf_tensors(base_q, base_k, base_v)
        row["forward_ms"] = benchmark_forward(implementation, q_fwd, k_fwd, v_fwd, warmup, rep)

        q_bwd, k_bwd, v_bwd = clone_leaf_tensors(base_q, base_k, base_v)
        row["backward_ms"] = benchmark_backward(implementation, q_bwd, k_bwd, v_bwd, base_d_out, warmup, rep)

        q_full, k_full, v_full = clone_leaf_tensors(base_q, base_k, base_v)
        row["forward_backward_ms"] = benchmark_forward_backward(
            implementation,
            q_full,
            k_full,
            v_full,
            base_d_out,
            warmup,
            rep,
        )
    except RuntimeError as error:
        torch.cuda.empty_cache()
        if not is_oom_error(error):
            raise
        row["status"] = "oom"

    return row


def run_benchmarks(device: torch.device, warmup: int, rep: int) -> pd.DataFrame:
    implementations = [
        ("pytorch", regular_pytorch_attention),
        ("triton_flash2", TritonFlashAttention2.apply),
    ]
    rows: list[dict[str, object]] = []

    for dtype_name, dtype in DTYPE_MAP.items():
        for embed_dim in EMBED_DIMS:
            for seq_len in SEQ_LENS:
                torch.cuda.empty_cache()
                for implementation_name, implementation in implementations:
                    print(
                        f"Benchmarking impl={implementation_name} dtype={dtype_name} seq_len={seq_len} embed_dim={embed_dim}"
                    )
                    rows.append(
                        benchmark_implementation(
                            name=implementation_name,
                            implementation=implementation,
                            seq_len=seq_len,
                            embed_dim=embed_dim,
                            dtype_name=dtype_name,
                            dtype=dtype,
                            device=device,
                            warmup=warmup,
                            rep=rep,
                        )
                    )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Running benchmarks on {gpu_name}")

    results = run_benchmarks(device=device, warmup=args.warmup, rep=args.rep)
    results.insert(0, "gpu", gpu_name)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(results.to_markdown(index=False) + "\n")
    results.to_csv(args.csv_path, index=False)
    print(results.to_markdown(index=False))


if __name__ == "__main__":
    main()
