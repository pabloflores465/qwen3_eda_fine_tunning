#!/usr/bin/env python3
"""
Fix JSONL format issues:
1. Parse nested JSON in instruction field (PyraNet)
2. Fix double-escaped strings in output
3. Remove invalid lines
"""
import json
from pathlib import Path
from tqdm import tqdm

BASE = Path("/Users/gp/Documents/qwen3_fine_tune/sources/combined_dataset")

def fix_sample(sample):
    """Fix format issues in a sample"""
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    source = sample.get("source", "unknown")
    
    # Fix instruction: if it contains JSON, extract the description
    if '{"description":' in instruction or "{'description':" in instruction:
        try:
            # Find where JSON starts
            json_start = instruction.find('{"description":')
            if json_start == -1:
                json_start = instruction.find("{'description':")
            
            if json_start != -1:
                prefix = instruction[:json_start].strip()
                json_str = instruction[json_start:]
                # Parse the JSON
                try:
                    desc_obj = json.loads(json_str)
                    desc = desc_obj.get("description", "")
                    if desc:
                        instruction = f"{prefix}\n{desc}" if prefix else desc
                except json.JSONDecodeError:
                    # Try with single quotes replaced
                    json_str = json_str.replace("'", '"')
                    try:
                        desc_obj = json.loads(json_str)
                        desc = desc_obj.get("description", "")
                        if desc:
                            instruction = f"{prefix}\n{desc}" if prefix else desc
                    except:
                        pass
        except:
            pass
    
    # Fix output: if it's a list represented as string, extract
    if isinstance(output, str):
        # Check if output is a JSON array string
        if output.startswith('["') or output.startswith("['"):
            try:
                parsed = json.loads(output)
                if isinstance(parsed, list) and len(parsed) > 0:
                    output = parsed[0]
            except:
                pass
        
        # Fix double escapes
        if '\\\\n' in output or '\\\\t' in output:
            output = output.replace('\\\\n', '\n').replace('\\\\t', '\t').replace('\\\\r', '\r')
    
    # If output is a list, take first element
    if isinstance(output, list) and len(output) > 0:
        output = output[0]
        if '\\\\n' in output:
            output = output.replace('\\\\n', '\n').replace('\\\\t', '\t').replace('\\\\r', '\r')
    
    return {
        "instruction": instruction.strip(),
        "output": output.strip() if isinstance(output, str) else str(output),
        "source": source
    }

def process_file(input_path, output_path):
    """Process and fix a JSONL file"""
    valid = 0
    fixed = 0
    skipped = 0
    
    # Count lines first
    with open(input_path, 'r') as f:
        total = sum(1 for _ in f)
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin, total=total, desc=f"Fixing {input_path.name}"):
                try:
                    sample = json.loads(line)
                    fixed_sample = fix_sample(sample)
                    
                    # Validate fixed sample
                    if fixed_sample["instruction"] and fixed_sample["output"]:
                        if len(fixed_sample["instruction"]) >= 10 and len(fixed_sample["output"]) >= 10:
                            json_str = json.dumps(fixed_sample, ensure_ascii=False)
                            fout.write(json_str + '\n')
                            valid += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1
                except Exception as e:
                    skipped += 1
    
    return valid, skipped

print("=" * 60)
print("FIXING JSONL FORMAT ISSUES")
print("=" * 60)

# Process train.jsonl
print("\n[1/2] Processing train.jsonl...")
train_in = BASE / "train.jsonl"
train_out = BASE / "train_fixed.jsonl"
train_valid, train_skipped = process_file(train_in, train_out)
print(f"   Valid: {train_valid:,}, Skipped: {train_skipped}")

# Process valid.jsonl
print("\n[2/2] Processing valid.jsonl...")
valid_in = BASE / "valid.jsonl"
valid_out = BASE / "valid_fixed.jsonl"
valid_valid, valid_skipped = process_file(valid_in, valid_out)
print(f"   Valid: {valid_valid:,}, Skipped: {valid_skipped}")

# Replace original files
print("\n" + "=" * 60)
print("REPLACING ORIGINAL FILES")
print("=" * 60)
import shutil
shutil.move(str(train_out), str(train_in))
shutil.move(str(valid_out), str(valid_in))
print("Done!")

# Show sample
print("\n" + "=" * 60)
print("SAMPLE ENTRIES (FIXED)")
print("=" * 60)
with open(train_in, 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"\n--- Sample {i+1} (source: {data.get('source', 'unknown')}) ---")
        inst = data['instruction']
        out = data['output']
        print(f"Instruction ({len(inst)} chars): {inst[:300]}{'...' if len(inst) > 300 else ''}")
        print(f"Output ({len(out)} chars): {out[:300]}{'...' if len(out) > 300 else ''}")

# Final stats
print("\n" + "=" * 60)
print("FINAL STATISTICS")
print("=" * 60)
train_size = train_in.stat().st_size / (1024*1024)
valid_size = valid_in.stat().st_size / (1024*1024)
print(f"train.jsonl: {train_size:.2f} MB ({train_valid:,} samples)")
print(f"valid.jsonl: {valid_size:.2f} MB ({valid_valid:,} samples)")
print(f"Total: {train_size + valid_size:.2f} MB ({train_valid + valid_valid:,} samples)")
