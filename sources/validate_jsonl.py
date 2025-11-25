#!/usr/bin/env python3
"""Validate JSONL files"""
import json
from pathlib import Path

BASE = Path("/Users/gp/Documents/qwen3_fine_tune/sources/combined_dataset")

def validate_jsonl(filepath):
    valid = 0
    invalid = 0
    missing_fields = 0
    sources = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if "instruction" in data and "output" in data:
                    valid += 1
                    src = data.get("source", "unknown")
                    sources[src] = sources.get(src, 0) + 1
                else:
                    missing_fields += 1
            except json.JSONDecodeError as e:
                invalid += 1
                if invalid <= 5:
                    print(f"  Invalid JSON at line {i+1}: {str(e)[:50]}")
    
    return valid, invalid, missing_fields, sources

print("=" * 60)
print("VALIDATING train.jsonl")
print("=" * 60)
train_path = BASE / "train.jsonl"
v, inv, mf, src = validate_jsonl(train_path)
print(f"\nValid lines: {v}")
print(f"Invalid JSON: {inv}")
print(f"Missing fields: {mf}")
print(f"\nSamples by source:")
for s, c in sorted(src.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c:,}")

print("\n" + "=" * 60)
print("VALIDATING valid.jsonl")
print("=" * 60)
valid_path = BASE / "valid.jsonl"
v2, inv2, mf2, src2 = validate_jsonl(valid_path)
print(f"\nValid lines: {v2}")
print(f"Invalid JSON: {inv2}")
print(f"Missing fields: {mf2}")
print(f"\nSamples by source:")
for s, c in sorted(src2.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c:,}")

print("\n" + "=" * 60)
print("SAMPLE ENTRIES")
print("=" * 60)
with open(train_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"\n--- Sample {i+1} (source: {data.get('source', 'unknown')}) ---")
        print(f"Instruction: {data['instruction'][:200]}...")
        print(f"Output: {data['output'][:200]}...")

print("\n" + "=" * 60)
print("FILE SIZES")
print("=" * 60)
train_mb = train_path.stat().st_size / (1024*1024)
valid_mb = valid_path.stat().st_size / (1024*1024)
print(f"train.jsonl: {train_mb:.2f} MB ({v:,} samples)")
print(f"valid.jsonl: {valid_mb:.2f} MB ({v2:,} samples)")
print(f"Total: {train_mb + valid_mb:.2f} MB ({v + v2:,} samples)")
