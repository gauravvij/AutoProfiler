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
| `benchmark_runs` | Number of benchmark iterations | 3-10 |
| `max_tokens` | Max tokens per generation | 50-200 |

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

## Phase 4: Iterative Experimentation Strategy

### The Exploration Loop

1. **Analyze** the ledger for patterns
2. **Hypothesize** what configuration might improve TPS/quality trade-off
3. **Create** a config JSON
4. **Execute** the experiment
5. **Log** results (automatic)
6. **Repeat**

### Exploration Strategies (USE THESE!)

#### Strategy A: Quantization Sweep
Start with the same model, vary quantization:
- 8-bit baseline → 4-bit → 6-bit (if available)
- Compare TPS gain vs. perplexity increase

#### Strategy B: GPU Layer Offloading
For GPU systems, find the sweet spot:
- Start with n_gpu_layers=0 (CPU only)
- Increase until all layers on GPU or OOM
- Note: Partial offloading can be optimal!

#### Strategy C: Context Length Scaling
Test how context affects performance:
- 2048 → 4096 → 8192
- Watch for memory cliff

#### Strategy D: Model Comparison
Same configuration, different models:
- Compare phi-3-mini vs. qwen2-7b at 4-bit
- Which gives better TPS/quality for your hardware?

### UNCONVENTIONAL STRATEGIES (EXPLICITLY ENCOURAGED!)

The framework supports exploration beyond standard approaches. Try these:

#### Asymmetric Quantization
- Different quantization for different layer types
- Embeddings at 8-bit, attention at 4-bit, FFN at 6-bit
- Document which layers are most sensitive

#### Hybrid Offloading
- Put embeddings + first N layers on GPU, rest on CPU
- Test if "front-heavy" offloading works better than uniform
- Try "sandwich" patterns: GPU-CPU-GPU

#### Strange Draft Models (Speculative Decoding)
- Use a completely different architecture as draft (e.g., GPT-2 for Llama)
- Test tiny models (125M) as drafters for larger models
- Experiment with acceptance thresholds

#### KV Cache Compression
- Test INT8 vs FP16 KV cache
- Try 4-bit KV cache if supported
- Measure impact on long-context performance

#### Batch Size Experiments
- Even with single-user inference, batching can help
- Test batch sizes 1, 2, 4, 8
- Note: Only effective for GPU inference

---

## Quality Thresholds

**DO NOT sacrifice quality for speed blindly.** Use these guidelines:

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| Perplexity | < 2x baseline | < 1.5x baseline | < 1.2x baseline |
| Quality Score | > 60% | > 75% | > 90% |

**If quality drops below acceptable:**
- Increase quantization bits
- Reduce context length
- Try a different model
- Document the failure in notes

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
