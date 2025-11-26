#!/bin/bash

# Qwen3 Verilog LoRA Fine-tuning Starter Script
# Optimized for 8GB RAM Apple Silicon

set -e

echo "============================================================"
echo "Qwen3 Verilog LoRA Fine-tuning"
echo "============================================================"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv-mlx/bin/activate

# Check if MLX is available
echo "Checking MLX installation..."
python3 -c "import mlx.core as mx; print(f'MLX Metal available: {mx.metal.is_available()}')" || {
    echo "Error: MLX not properly installed"
    exit 1
}

# Check if datasets exist
if [ ! -f "sources/combined_dataset/train.jsonl" ]; then
    echo "Error: Training dataset not found at sources/combined_dataset/train.jsonl"
    exit 1
fi

if [ ! -f "sources/combined_dataset/valid.jsonl" ]; then
    echo "Error: Validation dataset not found at sources/combined_dataset/valid.jsonl"
    exit 1
fi

# Check if model exists
if [ ! -d "Qwen3-4B-Thinking-2507-MLX-4bit" ]; then
    echo "Error: Model not found at Qwen3-4B-Thinking-2507-MLX-4bit"
    exit 1
fi

echo ""
echo "All checks passed!"
echo ""
echo "Starting training..."
echo "Press Ctrl+C to stop training at any time"
echo ""
echo "============================================================"
echo ""

# Run training
python3 train_lora.py

echo ""
echo "============================================================"
echo "Training session ended"
echo "============================================================"
