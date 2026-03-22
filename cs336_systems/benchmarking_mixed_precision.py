from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect autocast dtypes for the mixed-precision toy model.")
    parser.add_argument("--in_features", type=int, default=1024)
    parser.add_argument("--out_features", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--autocast_dtype", choices=list(DTYPE_MAP.keys()), default="float16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def dtype_name(dtype: torch.dtype | None) -> str | None:
    return None if dtype is None else str(dtype).replace("torch.", "")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script is intended to run on CUDA so autocast behavior matches the assignment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    autocast_dtype = DTYPE_MAP[args.autocast_dtype]
    model = ToyModel(in_features=args.in_features, out_features=args.out_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(args.batch_size, args.in_features, device=device)
    y = torch.randint(0, args.out_features, (args.batch_size,), device=device)

    observed_dtypes: dict[str, str | None] = {
        "parameter_dtype_in_autocast": dtype_name(next(model.parameters()).dtype),
    }

    def record_fc1_dtype(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed_dtypes["fc1_output_dtype"] = dtype_name(output.dtype)

    def record_ln_dtype(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed_dtypes["layer_norm_output_dtype"] = dtype_name(output.dtype)

    fc1_handle = model.fc1.register_forward_hook(record_fc1_dtype)
    ln_handle = model.ln.register_forward_hook(record_ln_dtype)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    if autocast_dtype == torch.float16:
        scaler = GradScaler(device="cuda")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    fc1_handle.remove()
    ln_handle.remove()

    observed_dtypes["logits_dtype"] = dtype_name(logits.dtype)
    observed_dtypes["loss_dtype"] = dtype_name(loss.dtype)
    observed_dtypes["gradient_dtype"] = dtype_name(model.fc1.weight.grad.dtype)

    print(
        json.dumps(
            {
                "device": str(device),
                "autocast_dtype": args.autocast_dtype,
                "observed_dtypes": observed_dtypes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()