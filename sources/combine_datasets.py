#!/usr/bin/env python3
"""
Script to combine all Verilog training datasets into unified train.jsonl and valid.jsonl
Handles: RTLCoder, MG-Verilog, VeriGen (raw), PyraNet-Verilog
"""

import json
import os
import random
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

# Base path
BASE_PATH = Path("/Users/gp/Documents/qwen3_fine_tune/sources")

# Output paths
OUTPUT_DIR = BASE_PATH / "combined_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VALID_PATH = OUTPUT_DIR / "valid.jsonl"

# Statistics
stats = {
    "rtlcoder": 0,
    "mg_verilog": 0,
    "verigen_raw": 0,
    "pyranet": 0,
    "total": 0,
    "train": 0,
    "valid": 0,
    "skipped_invalid": 0
}

combined_data = []

def is_valid_sample(instruction: str, output: str) -> bool:
    """Check if a sample is valid for training"""
    if not instruction or not output:
        return False
    if not isinstance(instruction, str) or not isinstance(output, str):
        return False
    if len(instruction.strip()) < 10 or len(output.strip()) < 10:
        return False
    return True

def clean_text(text: str) -> str:
    """Clean text for JSONL compatibility"""
    if not isinstance(text, str):
        return ""
    text = text.replace('\x00', '')
    return text.strip()

def add_sample(instruction: str, output: str, source: str):
    """Add a sample to the combined data if valid"""
    global stats
    instruction = clean_text(instruction)
    output = clean_text(output)
    
    if is_valid_sample(instruction, output):
        combined_data.append({
            "instruction": instruction,
            "output": output,
            "source": source
        })
        stats[source] = stats.get(source, 0) + 1
    else:
        stats["skipped_invalid"] += 1

print("=" * 60)
print("COMBINING VERILOG TRAINING DATASETS")
print("=" * 60)

# ============================================================
# 1. Load RTLCoder (27K samples)
# ============================================================
print("\n[1/4] Loading RTLCoder dataset...")
rtlcoder_path = BASE_PATH / "RTL-Coder" / "dataset" / "Resyn27k.json"

try:
    with open(rtlcoder_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for line in tqdm(content.strip().split('\n'), desc="RTLCoder"):
        if line.strip():
            try:
                sample = json.loads(line)
                instruction = sample.get("Instruction", "")
                response = sample.get("Response", [])
                
                if isinstance(response, list) and len(response) > 0:
                    output = response[0]
                elif isinstance(response, str):
                    output = response
                else:
                    continue
                
                add_sample(instruction, output, "rtlcoder")
            except json.JSONDecodeError:
                stats["skipped_invalid"] += 1
                continue
    
    print(f"   RTLCoder: {stats['rtlcoder']} samples loaded")
except Exception as e:
    print(f"   Error loading RTLCoder: {e}")

# ============================================================
# 2. Load MG-Verilog merged dataset only (avoid duplicates)
# ============================================================
print("\n[2/4] Loading MG-Verilog merged dataset...")

dataset_path = BASE_PATH / "mg_verilog_datasets" / "packaged_dataset" / "merged_dataset"
try:
    ds = load_from_disk(str(dataset_path))
    for sample in tqdm(ds, desc="MG-Verilog"):
        instruction = sample.get("description", "")
        output = sample.get("code", "")
        add_sample(instruction, output, "mg_verilog")
    print(f"   MG-Verilog: {stats['mg_verilog']} samples loaded")
except Exception as e:
    print(f"   Error loading MG-Verilog: {e}")

# ============================================================
# 3. Load VeriGen (raw code - create generic instruction)
# ============================================================
print("\n[3/4] Loading VeriGen dataset (raw code)...")
verigen_path = BASE_PATH / "verigen_local"

try:
    ds = load_from_disk(str(verigen_path))
    
    generic_instructions = [
        "Complete the following Verilog module:",
        "Implement this Verilog design:",
        "Write the Verilog code for the following module:",
        "Generate the Verilog implementation:",
        "Provide the Verilog code:",
    ]
    
    for i, sample in enumerate(tqdm(ds['train'], desc="VeriGen")):
        code = sample.get("text", "")
        if code and len(code.strip()) > 50:
            lines = code.strip().split('\n')
            module_line = ""
            for line in lines[:10]:
                if "module " in line:
                    module_line = line.strip()
                    break
            
            if module_line:
                instruction = f"Implement the following Verilog module:\n{module_line}"
            else:
                instruction = generic_instructions[i % len(generic_instructions)]
            
            add_sample(instruction, code, "verigen_raw")
    
    print(f"   VeriGen: {stats['verigen_raw']} samples loaded")
except Exception as e:
    print(f"   Error loading VeriGen: {e}")

# ============================================================
# 4. Load PyraNet-Verilog (largest dataset)
# ============================================================
print("\n[4/4] Loading PyraNet-Verilog dataset...")
pyranet_path = BASE_PATH / "pyranet_verilog_local"

try:
    ds = load_from_disk(str(pyranet_path))
    
    for sample in tqdm(ds['train'], desc="PyraNet"):
        code = sample.get("code", "")
        description = sample.get("description", "")
        
        if description and code:
            instruction = f"Implement the following Verilog module:\n{description}"
            add_sample(instruction, code, "pyranet")
    
    print(f"   PyraNet: {stats['pyranet']} samples loaded")
except Exception as e:
    print(f"   Error loading PyraNet: {e}")

# ============================================================
# Combine, shuffle, and split
# ============================================================
print("\n" + "=" * 60)
print("PROCESSING COMBINED DATA")
print("=" * 60)

stats["total"] = len(combined_data)
print(f"\nTotal samples collected: {stats['total']}")
print(f"Skipped invalid samples: {stats['skipped_invalid']}")

print("\nShuffling data...")
random.shuffle(combined_data)

split_idx = int(len(combined_data) * 0.8)
train_data = combined_data[:split_idx]
valid_data = combined_data[split_idx:]

stats["train"] = len(train_data)
stats["valid"] = len(valid_data)

print(f"Train samples: {stats['train']} (80%)")
print(f"Valid samples: {stats['valid']} (20%)")

# ============================================================
# Write JSONL files
# ============================================================
print("\n" + "=" * 60)
print("WRITING JSONL FILES")
print("=" * 60)

def write_jsonl(data: list, filepath: Path, desc: str):
    """Write data to JSONL file with validation"""
    valid_count = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in tqdm(data, desc=desc):
            try:
                json_str = json.dumps(sample, ensure_ascii=False)
                json.loads(json_str)  # Verify
                f.write(json_str + '\n')
                valid_count += 1
            except (json.JSONDecodeError, TypeError) as e:
                continue
    return valid_count

print(f"\nWriting train.jsonl to: {TRAIN_PATH}")
train_written = write_jsonl(train_data, TRAIN_PATH, "Writing train.jsonl")
print(f"   Written: {train_written} samples")

print(f"\nWriting valid.jsonl to: {VALID_PATH}")
valid_written = write_jsonl(valid_data, VALID_PATH, "Writing valid.jsonl")
print(f"   Written: {valid_written} samples")

# ============================================================
# Verify JSONL files
# ============================================================
print("\n" + "=" * 60)
print("VERIFYING JSONL FILES")
print("=" * 60)

def verify_jsonl(filepath: Path) -> tuple:
    """Verify JSONL file is valid"""
    valid_lines = 0
    invalid_lines = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if "instruction" in data and "output" in data:
                    valid_lines += 1
                else:
                    invalid_lines += 1
            except json.JSONDecodeError:
                invalid_lines += 1
    return valid_lines, invalid_lines

train_valid, train_invalid = verify_jsonl(TRAIN_PATH)
valid_valid, valid_invalid = verify_jsonl(VALID_PATH)

print(f"\ntrain.jsonl: {train_valid} valid lines, {train_invalid} invalid lines")
print(f"valid.jsonl: {valid_valid} valid lines, {valid_invalid} invalid lines")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nSamples per source:")
for source in ["rtlcoder", "mg_verilog", "verigen_raw", "pyranet"]:
    print(f"   {source}: {stats.get(source, 0)}")

print(f"\nTotal combined: {stats['total']}")
print(f"Train set: {train_written} samples")
print(f"Valid set: {valid_written} samples")
print(f"\nOutput files:")
print(f"   {TRAIN_PATH}")
print(f"   {VALID_PATH}")

train_size = TRAIN_PATH.stat().st_size / (1024 * 1024)
valid_size = VALID_PATH.stat().st_size / (1024 * 1024)
print(f"\nFile sizes:")
print(f"   train.jsonl: {train_size:.2f} MB")
print(f"   valid.jsonl: {valid_size:.2f} MB")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
