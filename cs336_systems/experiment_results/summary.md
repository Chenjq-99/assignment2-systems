# Experiment Summary

## Transformer Forward (seq_len=128, batch=4)

| model | eager forward (s) | compiled forward (s) | compile speedup | status |
|---|---:|---:|---:|---|
| small | 0.028665 | 0.010138 | 2.828x | ok |
| medium | 0.056028 | 0.033892 | 1.653x | ok |
| large | 0.082639 | 0.068078 | 1.214x | ok |
| xl | 0.157246 | 0.139617 | 1.126x | ok |
| 2.7B | n/a | n/a | n/a | oom |

## Transformer Forward+Backward (seq_len=128, batch=4)

| model | fp32 forward (s) | fp32 backward (s) | bf16 forward (s) | bf16 backward (s) | status |
|---|---:|---:|---:|---:|---|
| small | 0.034252 | 0.050901 | 0.038070 | 0.059580 | ok |
| medium | 0.064471 | 0.096022 | 0.070080 | 0.080234 | ok |
| large | 0.091812 | 0.159347 | 0.102134 | 0.119204 | ok |
| xl | n/a | n/a | n/a | n/a | oom |
| 2.7B | n/a | n/a | n/a | n/a | oom |

## Transformer Train Step (seq_len=128, batch=4)

| model | eager total (s) | compiled total (s) | compile speedup | status |
|---|---:|---:|---:|---|
| small | 0.097744 | 0.056123 | 1.742x | ok |
| medium | 0.217433 | 0.169464 | 1.283x | ok |
| large | n/a | n/a | n/a | oom |
| xl | n/a | n/a | n/a | oom |
| 2.7B | n/a | n/a | n/a | oom |

## Warmup Sensitivity (small, seq_len=128, batch=4)

| warmup | forward mean (s) | forward std (s) | backward mean (s) | backward std (s) |
|---:|---:|---:|---:|---:|
| 0 | 0.087531 | 0.163158 | 0.069715 | 0.049707 |
| 1 | 0.037745 | 0.003119 | 0.053767 | 0.002752 |
| 2 | 0.034285 | 0.001213 | 0.039650 | 0.002919 |
| 5 | 0.035439 | 0.003602 | 0.038713 | 0.001802 |

## Notes

- Mixed precision dtype inspection results are stored in mixed_precision_fp16.json and mixed_precision_bf16.json.
- Attention eager/compiled comparison is stored in benchmark_attn_comparison_results.md.
- 2.7B memory profiling failed during model materialization on this 12 GB GPU, so no memory snapshot could be produced.
