# Qwen 2.5 1.5B/3B Profiling Plan

## Goal
Profile Qwen 2.5 models (1.5B and 3B) to find the optimal speed/quality trade-off on CPU-only hardware (8 cores, 32GB RAM). Note: "Qwen 3.5 2B" is likely a typo for Qwen 2.5 1.5B or 3B; I will profile both to be thorough.

## Research Summary
- **Model IDs**: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` and `Qwen/Qwen2.5-3B-Instruct-GGUF`.
- **Chat Template**: `chatml`.
- **Hardware Constraints**: CPU-only, 8 cores, ~29GB available RAM. Max context 4096.
- **Quantization**: Q4_K_M is the standard starting point for CPU inference.

## Approach
Since I am in a CPU-only environment, I will focus on quantization levels (Q4_K_M, Q8_0) and context lengths. I will start with the 1.5B model as a baseline and then move to the 3B model.

## Subtasks
1. **Baseline 1.5B (Q4_K_M)**: Profile Qwen2.5-1.5B-Instruct with Q4_K_M quantization at 2048 context.
2. **Quality Check 1.5B (Q8_0)**: Profile Qwen2.5-1.5B-Instruct with Q8_0 to see quality/TPS trade-off.
3. **Context Stress 1.5B (Q4_K_M)**: Increase context to 4096 to check memory/TPS impact.
4. **Baseline 3B (Q4_K_M)**: Profile Qwen2.5-3B-Instruct with Q4_K_M at 2048 context.
5. **Comparison Report**: Summarize the findings in `experiments/qwen_report.md`.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/root/AutoProfiler/configs/qwen1.5b_q4.json` | Config for 1.5B Q4_K_M |
| `/root/AutoProfiler/configs/qwen1.5b_q8.json` | Config for 1.5B Q8_0 |
| `/root/AutoProfiler/configs/qwen3b_q4.json` | Config for 3B Q4_K_M |
| `/root/AutoProfiler/ledger.tsv` | Updated results ledger |
| `/root/AutoProfiler/experiments/qwen_report.md` | Final profiling report |

## Evaluation Criteria
- **TPS**: Tokens per second (target > 5 TPS for 1.5B on CPU).
- **Quality Score**: > 70% on eval tasks.
- **Perplexity**: Stable across quantization levels.

## Notes
- Using `chatml` template.
- `n_gpu_layers` set to 0 (CPU-only).
