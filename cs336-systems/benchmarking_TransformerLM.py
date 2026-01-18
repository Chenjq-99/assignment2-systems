import torch
import torch.nn as nn
from typing import Dict
from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse


class Benchmark:
    def __init__(self, model: nn.Module):
        # Initialize the model with the given hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
    def generate_random_batch(self, batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        return x

    def run_benchmark(
        self, 
        batch_size: int, 
        seq_len: int, 
        vocab_size: int, 
        num_warmup_steps: int, 
        num_steps: int, 
        mode: str = "forward_backward"
    ) -> Dict[str, float]:
        """
        Perform basic end-to-end benchmarking of the forward and backward passes.
        :param mode: "forward" to time only forward pass, "forward_backward" for both.
        """
        x = self.generate_random_batch(batch_size, seq_len, vocab_size)
        
        # 1. Warm-up phase: Run w steps before starting measurement
        for _ in range(num_warmup_steps):
            self._step(x, mode)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 2. Timing phase: Time the execution of n steps
        start_time = timeit.default_timer()
        
        for _ in range(num_steps):
            self._step(x, mode)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
        end_time = timeit.default_timer()
        
        total_time = end_time - start_time
        return {
            "total_time": total_time,
            "avg_time_per_step": total_time / num_steps
        }

    def _step(self, x: torch.Tensor, mode: str):
        if mode == "forward":
            with torch.no_grad():
                _ = self.model(x)
        else:
            self.model.train()
            output = self.model(x)
            loss = output.sum() 
            loss.backward()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer Performance")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Vacabulary size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=4 * 768, help="Feed-forward dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    
    parser.add_argument("--warmup_steps", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=args.d_model,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
    ).to(args.device)

    benchmark = Benchmark(model)
    results = benchmark.run_benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        mode="forward_backward"
    )
    print(f"Model parameters: {model.get_num_params()}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per step: {results['avg_time_per_step']:.2f} seconds")

if __name__ == "__main__":
    main()