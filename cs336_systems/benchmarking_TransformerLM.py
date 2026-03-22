from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import timeit
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_model_config(args: argparse.Namespace) -> dict[str, int | float]:
    if args.model_size != "custom":
        config = dict(MODEL_SPECS[args.model_size])
    else:
        config = {
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.n_layers,
            "num_heads": args.n_heads,
        }
        missing = [name for name, value in config.items() if value is None]
        if missing:
            raise ValueError(f"custom model_size requires arguments: {', '.join(missing)}")

    config["vocab_size"] = args.vocab_size
    config["context_length"] = args.seq_len
    config["rope_theta"] = args.rope_theta
    return config


def build_model(config: dict[str, int | float], device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
    )
    return model.to(device)


def maybe_compile_model(model: nn.Module, enabled: bool, backend: str | None, mode: str | None) -> nn.Module:
    if not enabled:
        return model

    compile_kwargs = {}
    if backend is not None:
        compile_kwargs["backend"] = backend
    if mode is not None:
        compile_kwargs["mode"] = mode
    return torch.compile(model, **compile_kwargs)


def is_oom_error(error: RuntimeError) -> bool:
    error_text = str(error).lower()
    return "out of memory" in error_text or "oom" in error_text


class Benchmark:
    def __init__(self, model: nn.Module, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.model = model.to(device)
        self.dtype = dtype

    def generate_random_batch(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        return x, y

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _autocast_context(self):
        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        if self.device.type != "cuda":
            raise ValueError("Mixed precision benchmarking requires CUDA.")
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def _nvtx_range(self, enabled: bool, name: str):
        if enabled and self.device.type == "cuda":
            return torch.cuda.nvtx.range(name)
        return contextlib.nullcontext()

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    def _summarize(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    def _step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str,
        optimizer: AdamW | None,
        annotate_nvtx: bool,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        self.model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        if mode == "forward":
            self.model.eval()
        else:
            self.model.train()

        with self._nvtx_range(annotate_nvtx, "forward"):
            self._synchronize()
            forward_start = timeit.default_timer()
            if mode == "forward":
                with torch.no_grad():
                    with self._autocast_context():
                        _ = self.model(x)
            else:
                with self._autocast_context():
                    logits = self.model(x)
                    loss = self._compute_loss(logits, y)
            self._synchronize()
            metrics["forward_time"] = timeit.default_timer() - forward_start

        if mode == "forward":
            return metrics

        with self._nvtx_range(annotate_nvtx, "backward"):
            self._synchronize()
            backward_start = timeit.default_timer()
            loss.backward()
            self._synchronize()
            metrics["backward_time"] = timeit.default_timer() - backward_start

        if mode == "train_step":
            if optimizer is None:
                raise ValueError("mode='train_step' requires an optimizer")
            with self._nvtx_range(annotate_nvtx, "optimizer_step"):
                self._synchronize()
                optimizer_start = timeit.default_timer()
                optimizer.step()
                self._synchronize()
                metrics["optimizer_time"] = timeit.default_timer() - optimizer_start

        return metrics

    def run_benchmark(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        num_warmup_steps: int,
        num_steps: int,
        mode: str = "forward_backward",
        profile_memory: bool = False,
        memory_snapshot_path: str | None = None,
        annotate_nvtx: bool = False,
        optimizer: AdamW | None = None,
    ) -> dict[str, object]:
        x, y = self.generate_random_batch(batch_size, seq_len, vocab_size)

        with self._nvtx_range(annotate_nvtx, "warmup"):
            for _ in range(num_warmup_steps):
                self._step(x, y, mode, optimizer, annotate_nvtx)

        self._synchronize()

        if profile_memory:
            if self.device.type != "cuda":
                raise ValueError("Memory profiling requires CUDA.")
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)

        measurements: list[dict[str, float]] = []
        peak_memory_values: list[float] = []

        try:
            with self._nvtx_range(annotate_nvtx, "measurement"):
                for _ in range(num_steps):
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)

                    measurements.append(self._step(x, y, mode, optimizer, annotate_nvtx))

                    if self.device.type == "cuda":
                        peak_memory_values.append(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        finally:
            if profile_memory:
                assert memory_snapshot_path is not None
                torch.cuda.memory._dump_snapshot(memory_snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)

        timing_summary: dict[str, dict[str, float]] = {}
        timing_keys = sorted({key for measurement in measurements for key in measurement})
        for key in timing_keys:
            timing_summary[key] = self._summarize([measurement[key] for measurement in measurements])

        return {
            "timings": timing_summary,
            "peak_memory_mb": self._summarize(peak_memory_values),
        }


def maybe_build_optimizer(mode: str, model: BasicsTransformerLM) -> AdamW | None:
    if mode == "train_step":
        return AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    return None


def run_single_configuration(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    config = resolve_model_config(args)
    raw_model = build_model(config, device)
    parameter_count = raw_model.get_num_params()
    model = maybe_compile_model(raw_model, args.compile_model, args.compile_backend, args.compile_mode)
    benchmark = Benchmark(model, device=device, dtype=DTYPE_MAP[args.dtype])
    optimizer = maybe_build_optimizer(args.mode, raw_model)
    result = benchmark.run_benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        mode=args.mode,
        profile_memory=args.profile_memory,
        memory_snapshot_path=str(args.memory_snapshot_path),
        annotate_nvtx=args.annotate_nvtx,
        optimizer=optimizer,
    )
    return {
        "model_size": args.model_size,
        "device": str(device),
        "dtype": args.dtype,
        "mode": args.mode,
        "compiled": args.compile_model,
        "parameter_count": parameter_count,
        "model_config": config,
        **result,
    }


def run_model_size_sweep(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    results = []
    for model_size in MODEL_SPECS:
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.model_size = model_size
        if args.profile_memory:
            sweep_args.memory_snapshot_path = args.memory_snapshot_path.with_name(
                f"{model_size}_{args.memory_snapshot_path.name}"
            )
        try:
            results.append(run_single_configuration(sweep_args, device))
        except RuntimeError as error:
            if not is_oom_error(error):
                raise
            torch.cuda.empty_cache()
            results.append(
                {
                    "model_size": model_size,
                    "device": str(device),
                    "dtype": args.dtype,
                    "mode": args.mode,
                    "compiled": args.compile_model,
                    "status": "oom",
                    "error": str(error),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Transformer Performance")
    parser.add_argument("--model_size", choices=[*MODEL_SPECS.keys(), "custom"], default="small")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Vocabulary size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--d_model", type=int, help="Model dimension for model_size=custom")
    parser.add_argument("--d_ff", type=int, help="Feed-forward dimension for model_size=custom")
    parser.add_argument("--n_layers", type=int, help="Number of layers for model_size=custom")
    parser.add_argument("--n_heads", type=int, help="Number of heads for model_size=custom")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="forward_backward")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--annotate_nvtx", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--memory_snapshot_path", type=Path, default=Path("memory_snapshot.pickle"))
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_backend", type=str)
    parser.add_argument("--compile_mode", type=str)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--sweep_model_sizes", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if args.sweep_model_sizes:
        summary: dict[str, object] | list[dict[str, object]] = run_model_size_sweep(args, device)
    else:
        summary = run_single_configuration(args, device)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
