# AutoProfiler: Autonomous Inference Optimization

This project is a structured experimentation framework for optimizing LLM inference on local hardware. It follows the "AutoResearch" pattern: the infrastructure provides the tools and the ledger, while an AI Agent (like you) provides the intelligence to explore, hypothesize, and iterate.

## 🚀 How to Start as an Agent

If a user gives you an instruction like: 
> "I want you to read README.md and follow the instructions in it to profile this model: Model name"

Follow these steps:

1. **Read the Manual**: Your entire operating logic is defined in `program.md`. Read it first to understand the rules, the exploration philosophy, and the "Template Integrity" protocol.
2. **Profile Hardware**: Run `python profile_hardware.py`. You cannot propose meaningful experiments until you know your VRAM, RAM, and CPU constraints.
3. **Check the Ledger**: Inspect `ledger.tsv`. See what has already been tried so you don't waste resources.
4. **Propose & Execute**: 
    - Create a JSON config in `configs/`.
    - Run `python run_experiment.py --config configs/your_config.json`.
5. **Analyze & Iterate**: Read the results in the ledger. Form a hypothesis (e.g., "Lowering KV cache precision might save enough VRAM to increase batch size") and try again.

## 🛠 Example Experiment Configurations

### Example 1: Standard Quantization Check
```json
{
  "experiment_name": "gemma2_q4_baseline",
  "model_id": "google/gemma-2-2b-it-GGUF",
  "filename": "gemma-2-2b-it-Q4_K_M.gguf",
  "chat_template": "gemma",
  "n_ctx": 4096,
  "n_gpu_layers": -1
}
```

### Example 2: Unconventional "Strange" Angle (Asymmetric/Aggressive)
*Hypothesis: Extreme context truncation + 4-bit KV cache on a high-bit model might yield "Instant" response for short tasks.*
```json
{
  "experiment_name": "gemma2_q8_tiny_ctx",
  "model_id": "google/gemma-2-2b-it-GGUF",
  "filename": "gemma-2-2b-it-Q8_0.gguf",
  "chat_template": "gemma",
  "n_ctx": 512,
  "cache_type_k": "q4_0",
  "cache_type_v": "q4_0",
  "n_gpu_layers": -1
}
```

### Example 3: Speculative Decoding (Llama-3)
```json
{
  "experiment_name": "llama3_speculative",
  "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
  "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
  "draft_model_id": "unsloth/Llama-3-8B-Instruct-GGUF", 
  "draft_filename": "Meta-Llama-3-8B-Instruct-Q2_K.gguf",
  "chat_template": "llama3",
  "n_ctx": 2048
}
```

Experiments are not suppoed to stop until user pauses or cancels the runs.

## ⚠️ Critical Reminders
- **Template Integrity**: Always verify the `chat_template`. A 0% quality score usually means a template error, not a bad model.
- **Hardware Bounds**: Do not exceed the VRAM reported in `hardware_profile.json`.
- **Exploration**: Don't just do a grid search. Try "strange" combinations.
