# AutoProfiler: Autonomous Inference Optimization Framework

## Goal
Create a structured experimentation framework (inspired by Karpathy's AutoResearch) that allows an AI agent to autonomously discover the optimal balance between inference speed (TPS) and model quality for local LLMs on specific hardware.

## Research Summary
- **AutoResearch Pattern**: Uses a `program.md` as an operating manual, a `results.tsv` ledger for state, and a fixed execution script (`train.py`). The agent is responsible for proposing experiments and analyzing the ledger.
- **Inference Knobs**: Quantization (GGUF, AWQ, EXL2), Engines (llama.cpp, vLLM), KV Cache (FP16, INT8, 4-bit), and Speculative Decoding.
- **Quality Metrics**: Perplexity (WikiText-2) and Task Accuracy (Mini-MMLU/HumanEval).
- **Hardware Awareness**: Requires a discovery phase to set memory and compute bounds.

## Approach
The framework will consist of:
1. **`program.md`**: The instruction set for the agent. It mandates running hardware discovery first, then iterating on experiments.
2. **`profile_hardware.py`**: A script to detect CPU, RAM, and GPU capabilities.
3. **`run_experiment.py`**: A unified entry point that takes a configuration, runs inference, benchmarks TPS/latency, and evaluates quality.
4. **`ledger.tsv`**: A persistent log of all experiments, metrics, and failures.
5. **`eval_suite/`**: A lightweight set of tasks to verify the model hasn't "collapsed" during optimization.

## Subtasks
1. **Setup Project Structure**: Create directories for experiments, configs, and evals.
2. **Implement Hardware Profiler**: Build `profile_hardware.py` to output system constraints in a format the agent can easily parse.
3. **Build Inference & Eval Engine**: Create `run_experiment.py` using a flexible backend (starting with `llama-cpp-python` for broad hardware compatibility) that measures TPS and Perplexity.
4. **Create Mini-Eval Suite**: Implement a small set of deterministic prompts to check for quality degradation beyond just perplexity.
5. **Write `program.md`**: Draft the comprehensive guide that tells the agent how to use the tools, read the ledger, and propose new experiments.
6. **Initialize Ledger**: Create the `ledger.tsv` with headers.
7. **Verification Run**: Run a manual "Experiment 0" to ensure the pipeline works end-to-end and logs correctly.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/root/autoprofiler_2/program.md` | The agent's operating manual. |
| `/root/autoprofiler_2/profile_hardware.py` | Hardware discovery tool. |
| `/root/autoprofiler_2/run_experiment.py` | The main execution engine. |
| `/root/autoprofiler_2/ledger.tsv` | The experiment tracking ledger. |
| `/root/autoprofiler_2/eval_suite/` | Quality verification tasks. |

## Evaluation Criteria
- The agent can successfully read `program.md` and identify its first task (hardware profiling).
- `run_experiment.py` correctly captures TPS and quality metrics for a given config.
- Failed experiments are logged in the ledger without crashing the system.
- The ledger provides enough data for an agent to "learn" (e.g., seeing that 4-bit is faster than 8-bit).

## Notes
- Initial implementation will focus on `llama-cpp-python` as it supports CPU and GPU (Metal/CUDA/Vulkan) and has built-in quantization support.
- The "Agent" in this context is any LLM (like Neo) interacting with this file structure.
- **Exploration Philosophy**: The `program.md` must explicitly encourage the agent to explore unconventional configurations. Examples include: asymmetric quantization (different bits for embeddings vs FFN layers), extreme context window truncation with KV-cache compression, speculative decoding with non-obvious draft models (e.g., using a different architecture entirely), or CPU/GPU hybrid offloading strategies. The agent should treat "strange" combinations as high-potential opportunities, not just stick to standard grid search.