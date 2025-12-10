#!/usr/bin/env python3
"""
Benchmark script for comparing MLX model performance with and without adapter.
Designed for MacBook Air M1.
"""

import subprocess
import time
import json
import re
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchmarkResult:
    prompt: str
    response: str
    total_time: float
    tokens_generated: int
    prompt_tokens: int
    tokens_per_second: float
    prompt_eval_time: Optional[float] = None
    generation_time: Optional[float] = None

def count_tokens_estimate(text: str) -> int:
    """Rough token count estimation (words * 1.3)"""
    return int(len(text.split()) * 1.3)

def run_mlx_generate(model: str, prompt: str, max_tokens: int, adapter_path: Optional[str] = None) -> BenchmarkResult:
    """Run mlx_lm.generate and capture timing metrics."""

    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--verbose"
    ]

    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    print(f"Running: {' '.join(cmd[:6])}...")

    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    total_time = time.time() - start_time

    output = result.stdout + result.stderr

    # Parse the output for metrics
    tokens_per_second = 0.0
    prompt_tokens = 0
    generation_tokens = 0
    prompt_eval_time = None
    generation_time = None

    # Look for MLX timing info in output
    # Pattern: "Prompt: X tokens, Y tokens-per-sec"
    prompt_match = re.search(r'Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec', output)
    if prompt_match:
        prompt_tokens = int(prompt_match.group(1))

    # Pattern: "Generation: X tokens, Y tokens-per-sec"
    gen_match = re.search(r'Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec', output)
    if gen_match:
        generation_tokens = int(gen_match.group(1))
        tokens_per_second = float(gen_match.group(2))

    # Extract the actual response (everything after the prompt echo)
    response_lines = []
    capture = False
    for line in output.split('\n'):
        if '==========' in line or 'Prompt:' in line or 'Generation:' in line:
            capture = False
            continue
        if capture or (line.strip() and not line.startswith('[')):
            response_lines.append(line)
            capture = True

    response = '\n'.join(response_lines).strip()

    # If we couldn't parse tokens, estimate
    if generation_tokens == 0:
        generation_tokens = count_tokens_estimate(response)
    if tokens_per_second == 0 and total_time > 0:
        tokens_per_second = generation_tokens / total_time

    return BenchmarkResult(
        prompt=prompt,
        response=response,
        total_time=total_time,
        tokens_generated=generation_tokens,
        prompt_tokens=prompt_tokens,
        tokens_per_second=tokens_per_second,
        prompt_eval_time=prompt_eval_time,
        generation_time=generation_time
    )

# Test prompts - EDA/Verilog focused since that's the fine-tuning domain
TEST_PROMPTS = [
    ("Write a Verilog 4-bit counter", 500),
    ("Design a Verilog D flip-flop with async reset", 400),
    ("Write a Verilog 2-to-1 multiplexer", 300),
    ("Create a Verilog FSM for traffic light controller", 600),
    ("Write a Verilog testbench for a 4-bit adder", 500),
]

def run_benchmarks(model: str, adapter_path: Optional[str] = None, label: str = "Model") -> list:
    """Run all benchmark prompts and return results."""
    results = []

    print(f"\n{'='*60}")
    print(f"Running benchmarks for: {label}")
    print(f"{'='*60}\n")

    for prompt, max_tokens in TEST_PROMPTS:
        print(f"\nPrompt: {prompt[:50]}...")
        try:
            result = run_mlx_generate(model, prompt, max_tokens, adapter_path)
            results.append(result)
            print(f"  Tokens: {result.tokens_generated}, Time: {result.total_time:.2f}s, Speed: {result.tokens_per_second:.2f} tok/s")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 300s")
            results.append(BenchmarkResult(
                prompt=prompt,
                response="TIMEOUT",
                total_time=300,
                tokens_generated=0,
                prompt_tokens=0,
                tokens_per_second=0
            ))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(BenchmarkResult(
                prompt=prompt,
                response=f"ERROR: {e}",
                total_time=0,
                tokens_generated=0,
                prompt_tokens=0,
                tokens_per_second=0
            ))

    return results

def save_results(baseline_results: list, finetuned_results: list, output_file: str):
    """Save results to JSON for later analysis."""
    data = {
        "baseline": [
            {
                "prompt": r.prompt,
                "response": r.response,
                "total_time": r.total_time,
                "tokens_generated": r.tokens_generated,
                "tokens_per_second": r.tokens_per_second
            }
            for r in baseline_results
        ],
        "finetuned": [
            {
                "prompt": r.prompt,
                "response": r.response,
                "total_time": r.total_time,
                "tokens_generated": r.tokens_generated,
                "tokens_per_second": r.tokens_per_second
            }
            for r in finetuned_results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    MODEL = "Qwen3-4B-Thinking-2507-MLX-4bit"
    ADAPTER_PATH = "adapters"

    print("MLX Model Benchmark - MacBook Air M1")
    print("="*60)

    # Run baseline (no adapter)
    print("\n[1/2] Running BASELINE benchmarks (no adapter)...")
    baseline_results = run_benchmarks(MODEL, adapter_path=None, label="Baseline (No Adapter)")

    # Run with adapter
    print("\n[2/2] Running FINE-TUNED benchmarks (with adapter)...")
    finetuned_results = run_benchmarks(MODEL, adapter_path=ADAPTER_PATH, label="Fine-tuned (With Adapter)")

    # Save results
    save_results(baseline_results, finetuned_results, "benchmark_results.json")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_avg_speed = sum(r.tokens_per_second for r in baseline_results) / len(baseline_results)
    finetuned_avg_speed = sum(r.tokens_per_second for r in finetuned_results) / len(finetuned_results)

    print(f"\nBaseline average speed: {baseline_avg_speed:.2f} tokens/sec")
    print(f"Fine-tuned average speed: {finetuned_avg_speed:.2f} tokens/sec")
    print(f"Speed difference: {((finetuned_avg_speed - baseline_avg_speed) / baseline_avg_speed * 100):.1f}%")
