#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Qwen3-4B Model Comparison
Using validation dataset for proper evaluation.
MacBook Air M1 8GB
"""

import subprocess
import time
import json
import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import sys

@dataclass
class BenchmarkResult:
    prompt: str
    prompt_short: str
    response: str
    total_time: float
    tokens_generated: int
    prompt_tokens: int
    generation_tps: float
    prompt_tps: float
    peak_memory_gb: float
    has_code: bool
    complexity: str

def extract_prompts_from_validation(file_path: str, num_samples: int = 10) -> List[Dict]:
    """Extract diverse prompts from validation dataset."""
    prompts = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Sample diverse prompts
    random.seed(42)  # For reproducibility
    sampled_lines = random.sample(lines, min(num_samples * 3, len(lines)))

    for line in sampled_lines:
        try:
            data = json.loads(line)
            text = data.get('text', '')

            # Extract the user prompt (description)
            if '<|im_start|>user' in text and '<|im_end|>' in text:
                user_start = text.find('<|im_start|>user') + len('<|im_start|>user')
                user_end = text.find('<|im_end|>', user_start)
                user_content = text[user_start:user_end].strip()

                # Try to parse as JSON to get description
                try:
                    user_json = json.loads(user_content)
                    description = user_json.get('description', '')
                    complexity = user_json.get('complexity', 'Unknown')
                    if description and len(description) > 50:
                        prompts.append({
                            'prompt': description,
                            'complexity': complexity
                        })
                except:
                    # Not JSON, use as is
                    if len(user_content) > 20:
                        prompts.append({
                            'prompt': user_content[:500],
                            'complexity': 'Unknown'
                        })

        except Exception as e:
            continue

        if len(prompts) >= num_samples:
            break

    return prompts[:num_samples]

def run_mlx_generate(model: str, prompt: str, max_tokens: int, adapter_path: Optional[str] = None) -> BenchmarkResult:
    """Run mlx_lm.generate and capture timing metrics."""

    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--verbose", "true"
    ]

    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    prompt_short = prompt[:80] + "..." if len(prompt) > 80 else prompt
    print(f"  Running: {prompt_short}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        total_time = time.time() - start_time
        output = result.stdout + result.stderr

        # Parse metrics from output
        generation_tps = 0.0
        prompt_tps = 0.0
        prompt_tokens = 0
        generation_tokens = 0
        peak_memory = 0.0

        # Pattern: "Prompt: X tokens, Y tokens-per-sec"
        prompt_match = re.search(r'Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec', output)
        if prompt_match:
            prompt_tokens = int(prompt_match.group(1))
            prompt_tps = float(prompt_match.group(2))

        # Pattern: "Generation: X tokens, Y tokens-per-sec"
        gen_match = re.search(r'Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec', output)
        if gen_match:
            generation_tokens = int(gen_match.group(1))
            generation_tps = float(gen_match.group(2))

        # Pattern: "Peak memory: X GB"
        mem_match = re.search(r'Peak memory:\s*([\d.]+)\s*GB', output)
        if mem_match:
            peak_memory = float(mem_match.group(1))

        # Extract response (between ========== markers)
        response = ""
        if '==========' in output:
            parts = output.split('==========')
            if len(parts) >= 2:
                response = parts[1].strip()

        # Check if response contains Verilog code
        has_code = 'module' in response.lower() or 'endmodule' in response.lower()

        return BenchmarkResult(
            prompt=prompt,
            prompt_short=prompt_short,
            response=response[:2000],  # Limit response size
            total_time=total_time,
            tokens_generated=generation_tokens,
            prompt_tokens=prompt_tokens,
            generation_tps=generation_tps,
            prompt_tps=prompt_tps,
            peak_memory_gb=peak_memory,
            has_code=has_code,
            complexity="Unknown"
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            prompt=prompt,
            prompt_short=prompt_short,
            response="TIMEOUT",
            total_time=600,
            tokens_generated=0,
            prompt_tokens=0,
            generation_tps=0,
            prompt_tps=0,
            peak_memory_gb=0,
            has_code=False,
            complexity="Unknown"
        )
    except Exception as e:
        return BenchmarkResult(
            prompt=prompt,
            prompt_short=prompt_short,
            response=f"ERROR: {e}",
            total_time=0,
            tokens_generated=0,
            prompt_tokens=0,
            generation_tps=0,
            prompt_tps=0,
            peak_memory_gb=0,
            has_code=False,
            complexity="Unknown"
        )

def run_benchmarks(model: str, prompts: List[Dict], max_tokens: int, adapter_path: Optional[str] = None, label: str = "Model") -> List[BenchmarkResult]:
    """Run benchmarks on all prompts."""
    results = []

    print(f"\n{'='*60}")
    print(f"Running benchmarks for: {label}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data['prompt']
        complexity = prompt_data.get('complexity', 'Unknown')

        print(f"\n[{i+1}/{len(prompts)}] Complexity: {complexity}")

        result = run_mlx_generate(model, prompt, max_tokens, adapter_path)
        result.complexity = complexity
        results.append(result)

        print(f"  Tokens: {result.tokens_generated}, Time: {result.total_time:.2f}s, Speed: {result.generation_tps:.2f} tok/s, Code: {result.has_code}")

    return results

def save_results(baseline_results: List[BenchmarkResult], finetuned_results: List[BenchmarkResult], output_file: str):
    """Save results to JSON."""
    data = {
        "baseline": [
            {
                "prompt_short": r.prompt_short,
                "complexity": r.complexity,
                "tokens_generated": r.tokens_generated,
                "total_time": r.total_time,
                "generation_tps": r.generation_tps,
                "prompt_tps": r.prompt_tps,
                "peak_memory_gb": r.peak_memory_gb,
                "has_code": r.has_code,
                "response_preview": r.response[:500]
            }
            for r in baseline_results
        ],
        "finetuned": [
            {
                "prompt_short": r.prompt_short,
                "complexity": r.complexity,
                "tokens_generated": r.tokens_generated,
                "total_time": r.total_time,
                "generation_tps": r.generation_tps,
                "prompt_tps": r.prompt_tps,
                "peak_memory_gb": r.peak_memory_gb,
                "has_code": r.has_code,
                "response_preview": r.response[:500]
            }
            for r in finetuned_results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")

def print_summary(baseline_results: List[BenchmarkResult], finetuned_results: List[BenchmarkResult]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    # Filter out errors/timeouts
    baseline_valid = [r for r in baseline_results if r.generation_tps > 0]
    finetuned_valid = [r for r in finetuned_results if r.generation_tps > 0]

    if baseline_valid:
        baseline_avg_tps = sum(r.generation_tps for r in baseline_valid) / len(baseline_valid)
        baseline_avg_tokens = sum(r.tokens_generated for r in baseline_valid) / len(baseline_valid)
        baseline_avg_time = sum(r.total_time for r in baseline_valid) / len(baseline_valid)
        baseline_code_rate = sum(1 for r in baseline_valid if r.has_code) / len(baseline_valid) * 100
        baseline_avg_mem = sum(r.peak_memory_gb for r in baseline_valid) / len(baseline_valid)

        print(f"\nBASELINE (No Adapter):")
        print(f"  Valid samples: {len(baseline_valid)}/{len(baseline_results)}")
        print(f"  Avg generation speed: {baseline_avg_tps:.2f} tokens/sec")
        print(f"  Avg tokens generated: {baseline_avg_tokens:.0f}")
        print(f"  Avg response time: {baseline_avg_time:.2f}s")
        print(f"  Code generation rate: {baseline_code_rate:.1f}%")
        print(f"  Avg peak memory: {baseline_avg_mem:.3f} GB")

    if finetuned_valid:
        finetuned_avg_tps = sum(r.generation_tps for r in finetuned_valid) / len(finetuned_valid)
        finetuned_avg_tokens = sum(r.tokens_generated for r in finetuned_valid) / len(finetuned_valid)
        finetuned_avg_time = sum(r.total_time for r in finetuned_valid) / len(finetuned_valid)
        finetuned_code_rate = sum(1 for r in finetuned_valid if r.has_code) / len(finetuned_valid) * 100
        finetuned_avg_mem = sum(r.peak_memory_gb for r in finetuned_valid) / len(finetuned_valid)

        print(f"\nFINE-TUNED (With LoRA Adapter):")
        print(f"  Valid samples: {len(finetuned_valid)}/{len(finetuned_results)}")
        print(f"  Avg generation speed: {finetuned_avg_tps:.2f} tokens/sec")
        print(f"  Avg tokens generated: {finetuned_avg_tokens:.0f}")
        print(f"  Avg response time: {finetuned_avg_time:.2f}s")
        print(f"  Code generation rate: {finetuned_code_rate:.1f}%")
        print(f"  Avg peak memory: {finetuned_avg_mem:.3f} GB")

    if baseline_valid and finetuned_valid:
        tps_diff = ((finetuned_avg_tps - baseline_avg_tps) / baseline_avg_tps) * 100
        tokens_diff = ((finetuned_avg_tokens - baseline_avg_tokens) / baseline_avg_tokens) * 100

        print(f"\nCOMPARISON:")
        print(f"  Speed difference: {tps_diff:+.1f}%")
        print(f"  Tokens difference: {tokens_diff:+.1f}%")
        print(f"  Code rate difference: {finetuned_code_rate - baseline_code_rate:+.1f}%")

if __name__ == "__main__":
    MODEL = "Qwen3-4B-Thinking-2507-MLX-4bit"
    ADAPTER_PATH = "adapters"
    VALIDATION_FILE = "sources/processed_dataset/valid.jsonl"
    MAX_TOKENS = 2000000  # High limit for reasoning model
    NUM_SAMPLES = 10  # Number of samples to test

    print("="*60)
    print("Comprehensive MLX Model Benchmark")
    print("MacBook Air M1 8GB")
    print("="*60)
    print(f"\nModel: {MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Samples: {NUM_SAMPLES}")

    # Extract prompts from validation dataset
    print(f"\nLoading prompts from {VALIDATION_FILE}...")
    prompts = extract_prompts_from_validation(VALIDATION_FILE, NUM_SAMPLES)
    print(f"Loaded {len(prompts)} diverse prompts")

    for i, p in enumerate(prompts):
        print(f"  {i+1}. [{p['complexity']}] {p['prompt'][:60]}...")

    # Run baseline benchmarks
    print("\n[1/2] Running BASELINE benchmarks...")
    baseline_results = run_benchmarks(MODEL, prompts, MAX_TOKENS, adapter_path=None, label="Baseline (No Adapter)")

    # Run fine-tuned benchmarks
    print("\n[2/2] Running FINE-TUNED benchmarks...")
    finetuned_results = run_benchmarks(MODEL, prompts, MAX_TOKENS, adapter_path=ADAPTER_PATH, label="Fine-tuned (With Adapter)")

    # Save and summarize results
    save_results(baseline_results, finetuned_results, "comprehensive_benchmark_results.json")
    print_summary(baseline_results, finetuned_results)

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)
