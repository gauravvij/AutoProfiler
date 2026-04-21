#!/usr/bin/env python3
"""
AutoProfiler Experiment Runner
Runs inference benchmarks with quality evaluation using llama-cpp-python.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

# llama-cpp-python imports
from llama_cpp import Llama


def compute_config_hash(config: Dict) -> str:
    """Compute a unique hash for the experiment configuration."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def download_model(model_id: str, quant_bits: int) -> str:
    """
    Download or locate a GGUF model.
    Returns the path to the model file.
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Map common models to their GGUF URLs
    model_urls = {
        "phi-3-mini": {
            4: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            8: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q8.gguf",
        },
        "llama-3-8b": {
            4: "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q4_K_M.gguf",
            8: "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q8_0.gguf",
        },
        "qwen2-7b": {
            4: "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_0.gguf",
            8: "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q8_0.gguf",
        },
        "mistral-7b": {
            4: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            8: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf",
        },
        "unsloth/Qwen3.5-2B-GGUF": {
            4: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
            8: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf",
        },
    }
    
    # Default to phi-3-mini if model not in mapping
    if model_id not in model_urls:
        print(f"⚠ Model '{model_id}' not in URL mapping, defaulting to phi-3-mini")
        model_id = "phi-3-mini"
    
    # Default to 4-bit if quant not available
    if quant_bits not in model_urls[model_id]:
        quant_bits = 4
    
    url = model_urls[model_id][quant_bits]
    filename = url.split("/")[-1]
    model_path = models_dir / filename
    
    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return str(model_path)
    
    print(f"⬇ Downloading model from {url}...")
    print(f"   This may take several minutes...")
    
    # Download using wget or curl
    import urllib.request
    import ssl
    
    # Create SSL context that doesn't verify certificates (for HuggingFace)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Downloaded to: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"✗ Download failed: {e}")
        # Try alternative: use curl
        import subprocess
        result = subprocess.run(
            ["curl", "-L", "-o", str(model_path), url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and model_path.exists():
            print(f"✓ Downloaded via curl to: {model_path}")
            return str(model_path)
        raise RuntimeError(f"Failed to download model: {e}")


def format_prompt_for_chat_template(prompt: str, chat_template: str) -> str:
    """
    Format a prompt according to the specified chat template.
    """
    if chat_template == "chatml" or chat_template == "qwen":
        # ChatML format used by Qwen
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif chat_template == "llama3":
        # Llama-3 format
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif chat_template == "gemma":
        # Gemma format
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif chat_template == "phi3":
        # Phi-3 format
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        # Default: no formatting
        return prompt


def load_eval_suite() -> Tuple[List[str], List[Dict]]:
    """
    Load evaluation prompts and expected outputs.
    Returns (perplexity_texts, task_prompts).
    """
    eval_dir = Path("eval_suite")
    
    # Load perplexity texts (WikiText-2 sample)
    perplexity_texts = []
    wiki_file = eval_dir / "wikitext_sample.txt"
    if wiki_file.exists():
        with open(wiki_file, "r") as f:
            content = f.read()
            # Split into chunks of ~100 tokens each
            words = content.split()
            chunk_size = 80  # Approximate tokens
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk) > 50:
                    perplexity_texts.append(chunk)
    else:
        # Fallback sample
        perplexity_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require significant computational resources.",
            "Natural language processing has evolved rapidly in recent years.",
        ]
    
    # Load task prompts
    task_prompts = []
    tasks_file = eval_dir / "tasks.json"
    if tasks_file.exists():
        with open(tasks_file, "r") as f:
            task_prompts = json.load(f)
    else:
        # Fallback tasks
        task_prompts = [
            {
                "name": "simple_math",
                "prompt": "What is 15 + 27? Answer with just the number.",
                "expected_keywords": ["42"],
                "category": "math"
            },
            {
                "name": "logic",
                "prompt": "If all cats are mammals and some mammals are pets, are all cats pets? Answer yes or no.",
                "expected_keywords": ["no", "not"],
                "category": "logic"
            },
            {
                "name": "coding",
                "prompt": "Write a Python function to calculate factorial. Just the code.",
                "expected_keywords": ["def", "factorial", "return"],
                "category": "code"
            },
        ]
    
    return perplexity_texts, task_prompts


def compute_perplexity(llm: Llama, texts: List[str]) -> float:
    """
    Compute perplexity on sample texts.
    Lower is better.
    """
    import numpy as np
    total_nll = 0.0
    total_tokens = 0
    
    for text in texts[:3]:  # Use first 3 samples for speed
        try:
            # Tokenize
            tokens = llm.tokenize(text.encode())
            if len(tokens) < 5:
                continue
            
            # We use a simpler approach: evaluate the sequence and get logprobs
            # llama-cpp-python's Llama class has a 'eval' method
            # But for simplicity and compatibility, we'll use the high-level API
            # with logprobs enabled.
            
            # We take the text and ask for 1 token with logprobs to get the 
            # probability of the sequence. Actually, a better way is:
            
            for i in range(1, min(len(tokens), 50)): # Limit to 50 tokens for speed
                prefix_tokens = tokens[:i]
                target_token = tokens[i]
                
                # Evaluate prefix
                llm.reset()
                llm.eval(prefix_tokens)
                logits = np.array(llm.scores[llm.n_tokens - 1])
                
                # Softmax
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                
                token_prob = probs[target_token]
                if token_prob > 0:
                    total_nll += -np.log(token_prob)
                    total_tokens += 1
                else:
                    total_nll += 20.0 # Heavy penalty for 0 prob
                    total_tokens += 1
                    
        except Exception as e:
            print(f"  Warning: perplexity computation failed for sample: {e}")
            continue
    
    if total_tokens == 0:
        return 0.0 # Return 0 instead of inf to avoid breaking ledger
    
    avg_nll = total_nll / total_tokens
    perplexity = np.exp(min(avg_nll, 20.0)) # Cap to avoid overflow
    return float(perplexity)


def evaluate_tasks(llm: Llama, tasks: List[Dict], chat_template: str = "") -> Dict:
    """
    Evaluate model on deterministic tasks.
    Returns quality metrics.
    """
    results = {
        "total_tasks": len(tasks),
        "passed_tasks": 0,
        "task_scores": []
    }
    
    for task in tasks:
        try:
            # Format prompt with chat template
            formatted_prompt = format_prompt_for_chat_template(task["prompt"], chat_template)
            
            # Generate response
            output = llm(
                formatted_prompt,
                max_tokens=100,
                temperature=0.1,  # Low temp for deterministic evaluation
                stop=["\n\n", "User:", "Human:", "<|im_start|>", "<|im_end|>", "<|reasoning|>", "<|/reasoning|>"],
            )
            
            response = output["choices"][0]["text"].strip()
            # Remove reasoning content if it leaked through
            if "<|reasoning|>" in response:
                response = response.split("<|/reasoning|>")[-1].strip()
            
            response = response.lower()
            
            # Check if expected keywords are present
            passed = False
            for keyword in task.get("expected_keywords", []):
                if keyword.lower() in response:
                    passed = True
                    break
            
            if passed:
                results["passed_tasks"] += 1
            
            results["task_scores"].append({
                "name": task["name"],
                "passed": passed,
                "response": response[:100]  # Truncate for logging
            })
        except Exception as e:
            print(f"  Warning: task evaluation failed: {e}")
            results["task_scores"].append({
                "name": task["name"],
                "passed": False,
                "error": str(e)
            })
    
    # Compute overall quality score (0-100)
    if results["total_tasks"] > 0:
        results["quality_score"] = (results["passed_tasks"] / results["total_tasks"]) * 100
    else:
        results["quality_score"] = 0.0
    
    return results


def benchmark_inference(llm: Llama, config: Dict) -> Dict:
    """
    Run inference benchmark and collect metrics.
    """
    print("\n--- Running Inference Benchmark ---")
    
    chat_template = config.get("chat_template", "")
    
    # Warmup
    print("  Warming up...")
    warmup_prompt = format_prompt_for_chat_template("Hello, how are you?", chat_template)
    for _ in range(2):
        _ = llm(warmup_prompt, max_tokens=20)
    
    # Benchmark prompt
    raw_benchmark_prompt = """Explain the concept of machine learning in simple terms.
    Machine learning is a subset of artificial intelligence that enables computers to learn
    and improve from experience without being explicitly programmed."""
    benchmark_prompt = format_prompt_for_chat_template(raw_benchmark_prompt, chat_template)
    
    # Collect latency measurements
    latencies = []
    tokens_generated = []
    
    num_runs = config.get("benchmark_runs", 5)
    max_tokens = config.get("max_tokens", 100)
    
    print(f"  Running {num_runs} benchmark iterations...")
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        output = llm(
            benchmark_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        num_tokens = output["usage"]["completion_tokens"]
        
        latencies.append(latency)
        tokens_generated.append(num_tokens)
        
        print(f"    Run {i+1}: {num_tokens} tokens in {latency:.3f}s ({num_tokens/latency:.2f} TPS)")
    
    # Compute statistics
    tps_values = [tokens / lat for tokens, lat in zip(tokens_generated, latencies)]
    
    results = {
        "tps": statistics.mean(tps_values),
        "tps_std": statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
        "latency_p50": statistics.median(latencies),
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
        "latency_mean": statistics.mean(latencies),
        "tokens_per_run": statistics.mean(tokens_generated),
    }
    
    print(f"\n  Results:")
    print(f"    TPS (mean): {results['tps']:.2f}")
    print(f"    Latency P50: {results['latency_p50']:.3f}s")
    print(f"    Latency P95: {results['latency_p95']:.3f}s")
    
    return results


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    except:
        return 0.0


def log_experiment(config: Dict, results: Dict, status: str, notes: str = ""):
    """Log experiment results to ledger.tsv."""
    ledger_path = Path("ledger.tsv")
    
    # Create headers if file doesn't exist
    if not ledger_path.exists():
        headers = [
            "timestamp", "config_hash", "model_id", "quant_bits", "n_gpu_layers",
            "n_ctx", "tps", "latency_p95", "memory_gb", "perplexity",
            "quality_score", "status", "notes"
        ]
        with open(ledger_path, "w") as f:
            f.write("\t".join(headers) + "\n")
    
    # Prepare row
    row = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": results.get("config_hash", ""),
        "model_id": config.get("model_id", "unknown"),
        "quant_bits": config.get("quant_bits", 4),
        "n_gpu_layers": config.get("n_gpu_layers", 0),
        "n_ctx": config.get("n_ctx", 4096),
        "tps": f"{results.get('tps', 0):.2f}",
        "latency_p95": f"{results.get('latency_p95', 0):.3f}",
        "memory_gb": f"{results.get('memory_gb', 0):.2f}",
        "perplexity": f"{results.get('perplexity', 0):.2f}" if results.get('perplexity') != float('inf') else "inf",
        "quality_score": f"{results.get('quality_score', 0):.1f}",
        "status": status,
        "notes": notes.replace("\t", " ").replace("\n", " ")
    }
    
    # Append to ledger
    with open(ledger_path, "a") as f:
        f.write("\t".join(str(row.get(h, "")) for h in [
            "timestamp", "config_hash", "model_id", "quant_bits", "n_gpu_layers",
            "n_ctx", "tps", "latency_p95", "memory_gb", "perplexity",
            "quality_score", "status", "notes"
        ]) + "\n")
    
    print(f"\n✓ Results logged to {ledger_path}")


def run_experiment(config_path: str):
    """
    Main entry point for running an experiment.
    """
    print("=" * 70)
    print("AutoProfiler Experiment Runner")
    print("=" * 70)
    
    # Load configuration
    print(f"\n📋 Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"  Model: {config.get('model_id', 'unknown')}")
    print(f"  Quantization: {config.get('quant_bits', 4)}-bit")
    print(f"  GPU Layers: {config.get('n_gpu_layers', 0)}")
    print(f"  Context: {config.get('n_ctx', 4096)}")
    
    # Compute config hash
    config_hash = compute_config_hash(config)
    print(f"  Config Hash: {config_hash}")
    
    results = {
        "config_hash": config_hash,
        "status": "pending"
    }
    
    try:
        # Download model
        print("\n⬇ Model Setup")
        model_path = download_model(config["model_id"], config.get("quant_bits", 4))
        
        # Initialize llama-cpp model
        print("\n🚀 Initializing Model")
        print(f"  Loading from: {model_path}")
        print(f"  n_gpu_layers: {config.get('n_gpu_layers', 0)}")
        print(f"  n_ctx: {config.get('n_ctx', 4096)}")
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=config.get("n_gpu_layers", 0),
            n_ctx=config.get("n_ctx", 4096),
            verbose=False,
        )
        print("  ✓ Model loaded successfully")
        
        # Get memory usage
        memory_gb = get_memory_usage()
        print(f"  Memory usage: {memory_gb:.2f} GB")
        results["memory_gb"] = memory_gb
        
        # Run inference benchmark
        benchmark_results = benchmark_inference(llm, config)
        results.update(benchmark_results)
        
        # Load evaluation suite
        print("\n📊 Quality Evaluation")
        perplexity_texts, task_prompts = load_eval_suite()
        
        # Compute perplexity
        print("  Computing perplexity...")
        perplexity = compute_perplexity(llm, perplexity_texts)
        results["perplexity"] = perplexity
        print(f"  Perplexity: {perplexity:.2f}")
        
        # Evaluate tasks
        print("  Evaluating task performance...")
        chat_template = config.get("chat_template", "")
        task_results = evaluate_tasks(llm, task_prompts, chat_template)
        results["quality_score"] = task_results["quality_score"]
        results["task_details"] = task_results
        print(f"  Quality Score: {results['quality_score']:.1f}%")
        
        # Log success
        results["status"] = "success"
        log_experiment(config, results, "success", "Experiment completed successfully")
        
        print("\n" + "=" * 70)
        print("✓ Experiment completed successfully!")
        print("=" * 70)
        
        # Print summary
        print(f"\n📈 Summary:")
        print(f"  TPS: {results['tps']:.2f}")
        print(f"  Latency P95: {results['latency_p95']:.3f}s")
        print(f"  Memory: {results['memory_gb']:.2f} GB")
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print(f"  Quality Score: {results['quality_score']:.1f}%")
        
        return 0
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n✗ Experiment failed: {error_msg}")
        traceback.print_exc()
        
        # Log failure
        results["status"] = "failed"
        log_experiment(config, results, "failed", error_msg)
        
        return 1


def main():
    parser = argparse.ArgumentParser(description="AutoProfiler Experiment Runner")
    parser.add_argument("config", help="Path to experiment configuration JSON file")
    args = parser.parse_args()
    
    return run_experiment(args.config)


if __name__ == "__main__":
    sys.exit(main())
