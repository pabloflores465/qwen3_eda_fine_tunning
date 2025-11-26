#!/bin/bash
# Memory-efficient training using mlx-lm CLI directly
# This avoids loading all data into Python memory at once

set -e

echo "============================================================"
echo "MLX-LM LoRA Training (Memory Optimized)"
echo "============================================================"

# Activate virtual environment
source .venv-mlx/bin/activate

# Set memory limits
export TOKENIZERS_PARALLELISM=false
export MLX_METAL_MEMORY_LIMIT=5368709120  # 5GB

# Close memory and clear caches
echo "Clearing system caches..."
sync

echo ""
echo "Starting training with ULTRA memory-optimized settings..."
echo "  - batch-size: 1"
echo "  - lora-layers: 4 (minimum)"
echo "  - grad-checkpoint: enabled"
echo "  - max-seq-length: 256 (reduced)"
echo "  - val-batches: 1"
echo ""

# Use the CLI tool directly - it handles data streaming more efficiently
mlx_lm.lora \
    --model Qwen3-4B-Thinking-2507-MLX-4bit \
    --train \
    --data sources/processed_dataset \
    --fine-tune-type lora \
    --batch-size 1 \
    --num-layers 4 \
    --iters 5000 \
    --val-batches 1 \
    --learning-rate 1e-4 \
    --steps-per-report 10 \
    --steps-per-eval 1000 \
    --save-every 500 \
    --adapter-path adapters \
    --max-seq-length 256 \
    --grad-checkpoint \
    --seed 42

echo ""
echo "============================================================"
echo "Training complete!"
echo "Adapters saved to: adapters/"
echo "============================================================"
