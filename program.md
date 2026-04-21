# AutoProfiler: Agent Operating Manual for LLM Inference Optimization

## Overview

AutoProfiler is a structured experimentation framework for discovering optimal LLM inference configurations. As an agent operating this framework, your goal is to systematically explore the configuration space to maximize tokens-per-second (TPS) while maintaining model quality above acceptable thresholds.

**Core Philosophy**: This is an exploration task. You are not optimizing a single metric—you are discovering the Pareto frontier of speed vs. quality. Failed experiments are valuable data points.

---

## Phase 0: Hardware Discovery (MANDATORY FIRST STEP)

**Before proposing ANY experiments, you MUST run hardware profiling.**

```bash
python profile_hardware.py
```

This generates `hardware_profile.json` containing:
- CPU cores, architecture
- Total/available RAM
- GPU type and VRAM (if available)
- System constraints (max context length, recommended quantization)

**Read this file and internalize the constraints.** Your experiments must respect:
- `constraints.max_quantization_bits`: Don't exceed this
- `constraints.max_context_length`: Don't exceed this
- `constraints.offload_strategy`: Use "hybrid" or "disk" for low-resource systems

## Phase 0.5: Template Validation (CRITICAL PREREQUISITE)

**Before running any experiment, you MUST audit and validate the chat template.**

A 0% or low quality score is often a sign of a broken template, not a bad model. For models with complex reasoning or specific stop tokens, you must ensure:
1. **Stop Tokens**: Verify that all required stop tokens (e.g., `<|im_end|>`, `<|reasoning|>`, `<|/reasoning|>`) are included in the inference call.
2. **Formatting**: Ensure the `format_prompt_for_chat_template` function correctly wraps the user input.
3. **Reasoning Suppression**: For reasoning-capable models, ensure the template prevents the model from outputting internal reasoning before the final answer unless specifically requested.

**Validation Step**: Run a single-task test and inspect the raw output logs to verify that special tokens are correctly handled and the model isn't "leaking" template artifacts.

---

## Phase 1: Understanding the Ledger

The `ledger.tsv` file is your memory. It tracks every experiment with these columns:

```
timestamp | config_hash | model_id | quant_bits | n_gpu_layers | n_ctx | tps | latency_p95 | memory_gb | perplexity | quality_score | status | notes
```

**How to read the ledger:**
- Each row is one experiment
- `config_hash` uniquely identifies the configuration
- `status` can be: success, failed, timeout
- `tps` = tokens per second (higher is better)
- `perplexity` = quality metric (lower is better, inf = failed)
- `quality_score` = percentage of eval tasks passed (0-100)

**How to use the ledger:**
1. Read all previous experiments
2. Identify patterns: "4-bit quantization gives 2x speed with minimal quality loss"
3. Notice failures: "8-bit on 8GB VRAM causes OOM"
4. Propose experiments that explore gaps in the data

---

## Phase 2: Experiment Configuration (for example if model is phi 3 mini)

Create JSON files in `configs/` directory. Each experiment needs:

```json
{
  "model_id": "phi-3-mini",
  "quant_bits": 4,
  "n_gpu_layers": 0,
  "n_ctx": 4096,
  "chat_template": "phi3",
  "benchmark_runs": 5,
  "max_tokens": 100
}
```

**Configuration Parameters:**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `model_id` | Model identifier | "phi-3-mini", "llama-3-8b", "qwen2-7b", "mistral-7b" |
| `quant_bits` | Quantization level | 4, 8 (higher = better quality, slower) |
| `n_gpu_layers` | Layers on GPU | 0 (CPU), -1 (all GPU), or specific number |
| `n_ctx` | Context window | 2048, 4096, 8192, 16384 |
| `chat_template` | Format for prompts | "phi3", "llama3", "gemma", "chatml" |
| `benchmark_runs` | Number of benchmark iterations | 3-10 |
| `max_tokens` | Max tokens per generation | 50-200 |

---

## CRITICAL: Template Integrity & Engine Portability

### 1. Chat Template Discipline
Optimization benchmarks are invalid if the model is prompted incorrectly.
- **Identify**: For every new model family, you MUST identify the correct chat template (Gemma, Llama-3, ChatML, etc.).
- **Specify**: Explicitly define the `chat_template` in the experiment config.
- **Verify**: If quality scores are unexpectedly low (e.g., <20%), you MUST inspect the raw prompt output in the logs to ensure special tokens are correctly placed. Do not mistake a template error for a quantization failure.

### 2. Engine Portability & Metric Consistency
- **Mutable Infrastructure**: You are encouraged to modify `run_experiment.py` to support new inference engines (vLLM, MLX, etc.) or optimization libraries.
- **Metric Parity**: If you change the engine, you MUST ensure the `perplexity` and `quality_score` remain comparable. If the new engine calculates perplexity differently, you must note this in the `notes` column of the ledger.
- **Re-Baselining**: After any major change to the execution engine, re-run a previously successful experiment to verify that the metrics (TPS, Perplexity) haven't shifted due to code changes rather than model changes.

---

## Phase 3: Running Experiments

Execute experiments with:

```bash
python run_experiment.py configs/your_config.json
```

**What happens:**
1. Model is downloaded (if not cached)
2. Inference benchmark runs (measures TPS, latency)
3. Perplexity is computed on WikiText sample
4. Quality tasks are evaluated
5. Results are logged to `ledger.tsv`

**Expected runtime:** 2-10 minutes per experiment depending on model size and hardware.

---

## Phase 4: The Continuous Exploration Loop (MANDATORY WORKFLOW)

Do NOT plan a sequence of experiments. Instead, follow this **Loop-of-One** protocol:

1. **Analyze**: Read the `ledger.tsv`. What is the current Pareto frontier? Where are the gaps?
2. **Hypothesize**: Formulate a specific question. 
   - *Standard*: "Can I hit 20 TPS with Q4 quantization?"
   - *Strange*: "Will a 2-bit draft model with a 512 context window enable 'instant' 50+ TPS for short logic tasks?"
3. **Propose**: Create ONE `configs/next_experiment.json`.
4. **Execute**: Run `python run_experiment.py`.
5. **Evaluate**: Immediately review the TPS **and** Quality Score.
6. **Learn**: Update your mental model. Was the hypothesis correct?
7. **Repeat**: Go back to Step 1.

### Integrated Exploration Strategy

You should mix "Standard" and "Strange" angles in your search. Do not wait for a "Phase 2" to be creative.

| Strategy Type | Approach | When to use |
|---------------|----------|-------------|
| **Standard** | Quantization sweeps (Q8 -> Q4), GPU offloading | To establish a baseline and find safe speedups. |
| **Strange** | Asymmetric quants, 4-bit KV cache, extreme context truncation, non-obvious draft models | When standard gains plateau or when aiming for "instant" (50+ TPS) performance. |
| **Aggressive** | Combining multiple strange angles (e.g., Q2 draft + 4-bit KV + 256 context) | To find the absolute physical limit of the hardware. |

---

## Quality Guardrails (Per-Experiment)

Every experiment is a speed-quality trade-off. Evaluate the result **immediately**:

- **If Quality > 80%**: You have "headroom." Try a more aggressive/strange optimization next.
- **If Quality 50-80%**: You are on the "Edge." This is the most valuable data. Try to stabilize quality while maintaining speed.
- **If Quality < 50%**: The model has collapsed. Your next experiment must be more conservative (higher bits, larger context, or better template).

---

## Failure Handling

**Experiments WILL fail.** This is expected and valuable.

Common failures:
- **OOM (Out of Memory)**: Reduce n_ctx, increase quantization, or reduce n_gpu_layers
- **Timeout**: Model too large for hardware, try smaller model
- **Perplexity = inf**: Model collapsed, reduce quantization or try different model

**When a failure occurs:**
1. Note the error in the ledger (automatic)
2. Analyze why it failed
3. Propose a modified configuration that might work
4. Document your reasoning in the notes field

---

## Stopping Criteria

Continue experimenting until ONE of these is true:

1. **Convergence**: No improvement in TPS for 5 consecutive experiments
2. **Quality Floor**: Quality score drops below 50% and cannot be recovered
3. **Coverage**: You've tested all reasonable combinations for your hardware
4. **User Interrupt**: The user tells you to stop

---

## Best Practices

1. **Start Conservative**: Begin with 4-bit quantization, moderate context (4096)
2. **Change One Variable**: Don't change quant_bits AND n_ctx in the same experiment
3. **Document Reasoning**: Use the notes field to explain WHY you tried a configuration
4. **Learn from Failures**: A failed experiment teaches you the boundary
5. **Compare Pareto-Optimal**: Keep track of configs that dominate others
6. **Never Substitute the User's Model**: If the model name specified by the user cannot be verified (e.g., no matching HuggingFace repo found), **stop and ask the user to confirm** the correct model identifier. Do not assume it is a typo and silently replace it with a different model.

---

## Example Agent Session

```
# Step 1: Hardware discovery
> python profile_hardware.py
  → 8 CPU cores, 32GB RAM, no GPU
  → constraints: max_quantization_bits=4, max_context_length=4096

# Step 2: Read ledger (empty, first run)
> cat ledger.tsv

# Step 3: Create first experiment
> cat configs/exp_001.json
{
  "model_id": "phi-3-mini",
  "quant_bits": 4,
  "n_gpu_layers": 0,
  "n_ctx": 4096
}

# Step 4: Run experiment
> python run_experiment.py configs/exp_001.json
  → TPS: 12.5, Perplexity: 8.2, Quality: 85%

# Step 5: Analyze and iterate
  → Good baseline! Let's try 8-bit for quality comparison.

> cat configs/exp_002.json
{
  "model_id": "phi-3-mini",
  "quant_bits": 8,
  "n_gpu_layers": 0,
  "n_ctx": 4096
}

# Step 6: Run and compare
> python run_experiment.py configs/exp_002.json
  → TPS: 6.2, Perplexity: 7.1, Quality: 92%

# Step 7: Document finding
  → 4-bit gives 2x speed with 15% perplexity increase. Acceptable trade-off.
```

---

## Quick Reference

**Hardware Profile:**
```bash
python profile_hardware.py
```

**Create Experiment:**
```bash
cat > configs/my_exp.json << 'EOF'
{
  "model_id": "phi-3-mini",
  "quant_bits": 4,
  "n_gpu_layers": 0,
  "n_ctx": 4096,
  "benchmark_runs": 5,
  "max_tokens": 100
}
EOF
```

**Run Experiment:**
```bash
python run_experiment.py configs/my_exp.json
```

**View Ledger:**
```bash
column -t -s $'\t' ledger.tsv
```

---

## Remember

- **You are an explorer, not just an optimizer**
- **Failed experiments are data**
- **Quality matters as much as speed**
- **Document your reasoning**
- **Be creative with unconventional strategies**

Now begin your hardware discovery and start experimenting!
