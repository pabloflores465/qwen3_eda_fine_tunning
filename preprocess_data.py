#!/usr/bin/env python3
"""
Pre-process training data: filter by token length to avoid memory issues.
"""

import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mlx_lm import load


def format_verilog_prompt(instruction: str, output: str) -> str:
    """Format instruction-output pair into training prompt"""
    if instruction.startswith('Implement the following Verilog module:'):
        instruction = instruction.replace('Implement the following Verilog module:\n', '')

    if isinstance(output, list):
        output = output[0] if len(output) > 0 else ""

    return f"""<|im_start|>system
You are an expert Verilog hardware designer. Generate clean, efficient, and correct Verilog code.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def process_dataset(input_file: str, output_file: str, tokenizer, max_tokens: int = 512):
    """Process dataset and filter by token length."""

    total = 0
    kept = 0
    skipped_long = 0
    skipped_invalid = 0

    token_lengths = []

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            total += 1

            if total % 10000 == 0:
                print(f"Processed {total} samples, kept {kept}...")

            try:
                item = json.loads(line.strip())
                instruction = item.get('instruction', '')
                output = item.get('output', '')

                # Format the prompt
                formatted = format_verilog_prompt(instruction, output)

                # Tokenize to check length
                tokens = tokenizer.encode(formatted)
                token_len = len(tokens)
                token_lengths.append(token_len)

                # Filter by length
                if token_len <= max_tokens:
                    # Save as text format for TextDataset
                    f_out.write(json.dumps({"text": formatted}) + '\n')
                    kept += 1
                else:
                    skipped_long += 1

            except (json.JSONDecodeError, KeyError) as e:
                skipped_invalid += 1
                continue

    # Statistics
    print(f"\n{'='*60}")
    print(f"Dataset: {input_file}")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Kept (tokens <= {max_tokens}): {kept} ({100*kept/total:.1f}%)")
    print(f"Skipped (too long): {skipped_long} ({100*skipped_long/total:.1f}%)")
    print(f"Skipped (invalid): {skipped_invalid}")

    if token_lengths:
        import statistics
        print(f"\nToken length statistics:")
        print(f"  Min: {min(token_lengths)}")
        print(f"  Max: {max(token_lengths)}")
        print(f"  Mean: {statistics.mean(token_lengths):.1f}")
        print(f"  Median: {statistics.median(token_lengths):.1f}")

    return kept


def main():
    print("="*60)
    print("Pre-processing Verilog Dataset")
    print("="*60)

    # Configuration
    model_path = "Qwen3-4B-Thinking-2507-MLX-4bit"
    max_tokens = 512  # Maximum sequence length

    input_train = "sources/combined_dataset/train.jsonl"
    input_valid = "sources/combined_dataset/valid.jsonl"

    output_dir = Path("sources/processed_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_train = output_dir / "train.jsonl"
    output_valid = output_dir / "valid.jsonl"

    print(f"\nMax tokens: {max_tokens}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer only
    print(f"\nLoading tokenizer from {model_path}...")
    _, tokenizer = load(model_path)
    print("Tokenizer loaded.")

    # Process datasets
    print("\n" + "="*60)
    print("Processing training set...")
    print("="*60)
    train_kept = process_dataset(input_train, str(output_train), tokenizer, max_tokens)

    print("\n" + "="*60)
    print("Processing validation set...")
    print("="*60)
    valid_kept = process_dataset(input_valid, str(output_valid), tokenizer, max_tokens)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training samples: {train_kept}")
    print(f"Validation samples: {valid_kept}")
    print(f"\nProcessed files saved to: {output_dir}")
    print("\nUpdate train_lora.py to use:")
    print(f'  self.train_file = "{output_train}"')
    print(f'  self.valid_file = "{output_valid}"')


if __name__ == "__main__":
    main()
