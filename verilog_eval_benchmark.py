#!/usr/bin/env python3
"""
VerilogEval Benchmark Script for Qwen3-4B Model Comparison
Evaluates baseline vs fine-tuned model on NVIDIA's VerilogEval benchmark.
Pass@1 metric with iverilog simulation verification.
"""

import subprocess
import os
import re
import json
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import time

@dataclass
class EvalResult:
    problem_id: str
    problem_name: str
    prompt: str
    generated_code: str
    compiles: bool
    passes_tests: bool
    compile_error: str
    simulation_output: str
    generation_time: float
    tokens_generated: int

@dataclass
class BenchmarkSummary:
    model_name: str
    total_problems: int
    compile_success: int
    test_pass: int
    pass_at_1: float
    compile_rate: float
    avg_generation_time: float
    results: List[EvalResult] = field(default_factory=list)

def get_problem_list(dataset_dir: str) -> List[Dict]:
    """Get list of all problems from the spec-to-rtl dataset."""
    problems = []
    dataset_path = Path(dataset_dir)

    # Find all prompt files
    prompt_files = sorted(dataset_path.glob("*_prompt.txt"))

    for prompt_file in prompt_files:
        # Extract problem ID and name
        filename = prompt_file.stem  # e.g., "Prob001_zero_prompt"
        parts = filename.replace("_prompt", "").split("_", 1)
        prob_id = parts[0]  # e.g., "Prob001"
        prob_name = parts[1] if len(parts) > 1 else ""  # e.g., "zero"

        # Find corresponding test and ref files
        test_file = dataset_path / f"{prob_id}_{prob_name}_test.sv"
        ref_file = dataset_path / f"{prob_id}_{prob_name}_ref.sv"

        if test_file.exists() and ref_file.exists():
            problems.append({
                "id": prob_id,
                "name": prob_name,
                "prompt_file": str(prompt_file),
                "test_file": str(test_file),
                "ref_file": str(ref_file)
            })

    return problems

def run_mlx_generate(model: str, prompt: str, max_tokens: int = 2048,
                     adapter_path: Optional[str] = None) -> Tuple[str, float, int]:
    """Run mlx_lm.generate and return the generated code."""

    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--verbose", "true"
    ]

    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per problem
        )

        generation_time = time.time() - start_time
        output = result.stdout + result.stderr

        # Extract tokens generated
        tokens = 0
        gen_match = re.search(r'Generation:\s*(\d+)\s*tokens', output)
        if gen_match:
            tokens = int(gen_match.group(1))

        # Extract response between ========== markers
        response = ""
        if '==========' in output:
            parts = output.split('==========')
            if len(parts) >= 2:
                response = parts[1].strip()
                # Clean up - remove the metrics part
                if '==========' in response:
                    response = response.split('==========')[0].strip()

        return response, generation_time, tokens

    except subprocess.TimeoutExpired:
        return "TIMEOUT", 300.0, 0
    except Exception as e:
        return f"ERROR: {e}", 0.0, 0

def extract_verilog_module(response: str, module_name: str = "TopModule") -> str:
    """Extract Verilog module from LLM response."""

    # Remove thinking tags if present
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Try to find module...endmodule block
    module_pattern = rf'(module\s+{module_name}\s*[\s\S]*?endmodule)'
    match = re.search(module_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try generic module pattern
    module_pattern = r'(module\s+\w+\s*[\s\S]*?endmodule)'
    match = re.search(module_pattern, response, re.IGNORECASE)
    if match:
        code = match.group(1)
        # Rename module to TopModule if needed
        code = re.sub(r'module\s+\w+', f'module {module_name}', code, count=1)
        return code

    # If code is in markdown blocks
    code_blocks = re.findall(r'```(?:verilog|sv|systemverilog)?\s*([\s\S]*?)```', response)
    for block in code_blocks:
        if 'module' in block.lower() and 'endmodule' in block.lower():
            return block.strip()

    # Last resort - return cleaned response
    return response.strip()

def verify_verilog(generated_code: str, test_file: str, ref_file: str,
                   work_dir: str) -> Tuple[bool, bool, str, str]:
    """Verify generated Verilog using iverilog simulation."""

    # Create work directory
    os.makedirs(work_dir, exist_ok=True)

    # Write generated code to file
    gen_file = os.path.join(work_dir, "TopModule.sv")
    with open(gen_file, 'w') as f:
        f.write(generated_code)

    # Copy test and ref files
    shutil.copy(test_file, work_dir)
    shutil.copy(ref_file, work_dir)

    test_basename = os.path.basename(test_file)
    ref_basename = os.path.basename(ref_file)

    # Try to compile
    output_file = os.path.join(work_dir, "sim.vvp")
    compile_cmd = [
        "iverilog", "-g2012", "-Wall",
        "-o", output_file,
        os.path.join(work_dir, test_basename),
        os.path.join(work_dir, ref_basename),
        gen_file
    ]

    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=work_dir
        )

        compile_error = compile_result.stderr
        compiles = compile_result.returncode == 0

        if not compiles:
            return False, False, compile_error, ""

        # Run simulation
        sim_cmd = ["vvp", output_file]
        sim_result = subprocess.run(
            sim_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=work_dir
        )

        sim_output = sim_result.stdout + sim_result.stderr

        # Check for pass/fail
        # VerilogEval uses "Mismatches: 0" to indicate success
        passes = "Mismatches: 0 in" in sim_output or "mismatches: 0" in sim_output.lower()

        # Also check for TIMEOUT which means failure
        if "TIMEOUT" in sim_output:
            passes = False

        return True, passes, compile_error, sim_output

    except subprocess.TimeoutExpired:
        return False, False, "Compilation/simulation timeout", ""
    except Exception as e:
        return False, False, str(e), ""

def run_evaluation(model: str, problems: List[Dict], adapter_path: Optional[str] = None,
                   label: str = "Model", max_problems: int = 156) -> BenchmarkSummary:
    """Run evaluation on all problems."""

    results = []
    compile_success = 0
    test_pass = 0
    total_time = 0.0

    problems_to_run = problems[:max_problems]

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Problems: {len(problems_to_run)}")
    print(f"{'='*60}\n")

    for i, problem in enumerate(problems_to_run):
        prob_id = problem["id"]
        prob_name = problem["name"]

        print(f"[{i+1}/{len(problems_to_run)}] {prob_id}_{prob_name}...", end=" ", flush=True)

        # Read prompt
        with open(problem["prompt_file"], 'r') as f:
            prompt = f.read().strip()

        # Generate code
        response, gen_time, tokens = run_mlx_generate(
            model, prompt, max_tokens=2048, adapter_path=adapter_path
        )
        total_time += gen_time

        # Extract module
        generated_code = extract_verilog_module(response)

        # Verify
        work_dir = tempfile.mkdtemp(prefix=f"verilog_eval_{prob_id}_")
        compiles, passes, compile_error, sim_output = verify_verilog(
            generated_code,
            problem["test_file"],
            problem["ref_file"],
            work_dir
        )

        # Cleanup
        try:
            shutil.rmtree(work_dir)
        except:
            pass

        if compiles:
            compile_success += 1
        if passes:
            test_pass += 1

        status = "PASS" if passes else ("COMPILE_FAIL" if not compiles else "TEST_FAIL")
        print(f"{status} ({gen_time:.1f}s)")

        results.append(EvalResult(
            problem_id=prob_id,
            problem_name=prob_name,
            prompt=prompt[:200],
            generated_code=generated_code[:500],
            compiles=compiles,
            passes_tests=passes,
            compile_error=compile_error[:200] if compile_error else "",
            simulation_output=sim_output[:200] if sim_output else "",
            generation_time=gen_time,
            tokens_generated=tokens
        ))

    # Calculate metrics
    total = len(problems_to_run)
    pass_at_1 = (test_pass / total) * 100 if total > 0 else 0
    compile_rate = (compile_success / total) * 100 if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    return BenchmarkSummary(
        model_name=label,
        total_problems=total,
        compile_success=compile_success,
        test_pass=test_pass,
        pass_at_1=pass_at_1,
        compile_rate=compile_rate,
        avg_generation_time=avg_time,
        results=results
    )

def save_results(baseline: BenchmarkSummary, finetuned: BenchmarkSummary,
                 output_file: str):
    """Save evaluation results to JSON."""

    data = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "VerilogEval v2 (spec-to-rtl)",
        "baseline": {
            "model": baseline.model_name,
            "total_problems": baseline.total_problems,
            "compile_success": baseline.compile_success,
            "test_pass": baseline.test_pass,
            "pass_at_1": baseline.pass_at_1,
            "compile_rate": baseline.compile_rate,
            "avg_generation_time": baseline.avg_generation_time,
            "results": [
                {
                    "id": r.problem_id,
                    "name": r.problem_name,
                    "compiles": r.compiles,
                    "passes": r.passes_tests,
                    "time": r.generation_time
                }
                for r in baseline.results
            ]
        },
        "finetuned": {
            "model": finetuned.model_name,
            "total_problems": finetuned.total_problems,
            "compile_success": finetuned.compile_success,
            "test_pass": finetuned.test_pass,
            "pass_at_1": finetuned.pass_at_1,
            "compile_rate": finetuned.compile_rate,
            "avg_generation_time": finetuned.avg_generation_time,
            "results": [
                {
                    "id": r.problem_id,
                    "name": r.problem_name,
                    "compiles": r.compiles,
                    "passes": r.passes_tests,
                    "time": r.generation_time
                }
                for r in finetuned.results
            ]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")

def print_comparison(baseline: BenchmarkSummary, finetuned: BenchmarkSummary):
    """Print comparison table."""

    print("\n" + "="*70)
    print("VERILOGEVAL BENCHMARK COMPARISON")
    print("="*70)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'Fine-tuned':<20}")
    print("-"*70)
    print(f"{'Total Problems':<30} {baseline.total_problems:<20} {finetuned.total_problems:<20}")
    print(f"{'Compile Success':<30} {baseline.compile_success:<20} {finetuned.compile_success:<20}")
    print(f"{'Test Pass':<30} {baseline.test_pass:<20} {finetuned.test_pass:<20}")
    print(f"{'Pass@1 (%)':<30} {baseline.pass_at_1:<20.2f} {finetuned.pass_at_1:<20.2f}")
    print(f"{'Compile Rate (%)':<30} {baseline.compile_rate:<20.2f} {finetuned.compile_rate:<20.2f}")
    print(f"{'Avg Gen Time (s)':<30} {baseline.avg_generation_time:<20.2f} {finetuned.avg_generation_time:<20.2f}")

    # Calculate improvement
    pass_improvement = finetuned.pass_at_1 - baseline.pass_at_1
    compile_improvement = finetuned.compile_rate - baseline.compile_rate

    print("\n" + "-"*70)
    print("IMPROVEMENT (Fine-tuned vs Baseline):")
    print(f"  Pass@1:       {pass_improvement:+.2f}%")
    print(f"  Compile Rate: {compile_improvement:+.2f}%")
    print("="*70)

def main():
    # Configuration
    MODEL = "Qwen3-4B-Thinking-2507-MLX-4bit"
    ADAPTER_PATH = "adapters"
    DATASET_DIR = "verilog-eval/dataset_spec-to-rtl"
    MAX_PROBLEMS = 156  # Full VerilogEval benchmark suite

    print("="*70)
    print("VerilogEval Benchmark - Model Comparison")
    print("NVIDIA VerilogEval v2 (spec-to-rtl)")
    print("="*70)
    print(f"\nModel: {MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Max Problems: {MAX_PROBLEMS}")

    # Get problems
    print(f"\nLoading problems from {DATASET_DIR}...")
    problems = get_problem_list(DATASET_DIR)
    print(f"Found {len(problems)} problems")

    # Run baseline evaluation
    print("\n[1/2] Running BASELINE evaluation...")
    baseline_results = run_evaluation(
        MODEL, problems, adapter_path=None,
        label="Baseline (No Adapter)",
        max_problems=MAX_PROBLEMS
    )

    # Run fine-tuned evaluation
    print("\n[2/2] Running FINE-TUNED evaluation...")
    finetuned_results = run_evaluation(
        MODEL, problems, adapter_path=ADAPTER_PATH,
        label="Fine-tuned (LoRA Adapter)",
        max_problems=MAX_PROBLEMS
    )

    # Save and compare
    save_results(baseline_results, finetuned_results, "verilog_eval_results.json")
    print_comparison(baseline_results, finetuned_results)

    print("\nBenchmark complete!")
    print("Results saved to: verilog_eval_results.json")

if __name__ == "__main__":
    main()
