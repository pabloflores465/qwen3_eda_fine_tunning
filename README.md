# Qwen3 EDA Fine-Tuning

Fine-tuning Qwen3-4B-Thinking for Electronic Design Automation (EDA) - specifically Verilog and SystemVerilog code generation.

## Project Overview

This project fine-tunes the Qwen3-4B-Thinking model using LoRA (Low-Rank Adaptation) on Apple Silicon Macs with MLX framework. The model is trained on a large dataset of Verilog code examples to improve hardware description language generation capabilities.

## Model Specifications

| Specification | Value |
|---------------|-------|
| Base Model | Qwen3-4B-Thinking-2507-MLX-4bit |
| Architecture | Qwen3ForCausalLM |
| Parameters | 4.02B total |
| Quantization | 4-bit (group_size: 64) |
| Hidden Size | 2560 |
| Layers | 36 |
| Attention Heads | 32 |
| KV Heads | 8 |
| Max Position Embeddings | 262,144 |
| Vocabulary Size | 151,643 |

## LoRA Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 4 |
| LoRA Alpha | 8 |
| LoRA Dropout | 0.05 |
| LoRA Layers | 4 |
| Trainable Parameters | 1.835M (0.046%) |
| Batch Size | 1 |
| Max Sequence Length | 256 |
| Learning Rate | 1e-4 |
| Training Iterations | 5,000 |
| Gradient Checkpointing | Enabled |

## Dataset

### Original Dataset
| Split | Samples |
|-------|---------|
| Train | 697,226 |
| Valid | 174,307 |
| **Total** | **871,533** |

### Processed Dataset (filtered for <= 512 tokens)
| Split | Samples | % Kept |
|-------|---------|--------|
| Train | 276,500 | 39.7% |
| Valid | 69,252 | 39.7% |
| **Total** | **345,752** | - |

### Data Format
```json
{
  "text": "<|im_start|>system\nYou are an expert Verilog hardware designer...<|im_end|>\n<|im_start|>user\n{description}<|im_end|>\n<|im_start|>assistant\n{verilog_code}<|im_end|>"
}
```

### Data Sources
- MG-Verilog datasets
- PyraNet Verilog dataset
- Combined and preprocessed for instruction-following format

## Project Structure

```
qwen3_fine_tune/
├── Qwen3-4B-Thinking-2507-MLX-4bit/  # Base model (2.1GB)
├── adapters/                          # LoRA adapters (77MB)
│   ├── adapters.safetensors          # Final trained weights
│   ├── 0005000_adapters.safetensors  # Checkpoint at 5000 iters
│   ├── adapter_config.json
│   └── training_config.json
├── sources/
│   ├── combined_dataset/             # Original dataset (4.9GB)
│   │   ├── train.jsonl
│   │   └── valid.jsonl
│   ├── processed_dataset/            # Filtered dataset (417MB)
│   │   ├── train.jsonl
│   │   └── valid.jsonl
│   ├── mg_verilog_datasets/          # Source data
│   └── pyranet_verilog_local/        # Source data
├── train_cli.sh                      # Memory-optimized training script
├── train_lora.py                     # Custom training script
├── preprocess_data.py                # Dataset preprocessing
├── start_training.sh                 # Training launcher
├── test_model.py                     # Model testing utilities
└── config.yaml                       # Configuration file
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- 8GB+ RAM (16GB+ recommended)
- Python 3.11+
- MLX framework

### Dependencies
```bash
pip install mlx mlx-lm transformers numpy
```

## Quick Start

```bash
# 1. Activate environment
source .venv-mlx/bin/activate

# 2. Use the fine-tuned model
mlx_lm.generate --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters \
    --prompt "Write a Verilog 4-bit counter" --max-tokens 500
```

## Commands Reference

### Environment Setup

| Command | Description |
|---------|-------------|
| `python3 -m venv .venv-mlx` | Create virtual environment |
| `source .venv-mlx/bin/activate` | Activate virtual environment |
| `pip install mlx mlx-lm transformers numpy` | Install dependencies |
| `deactivate` | Exit virtual environment |

### Data Preprocessing

| Command | Description |
|---------|-------------|
| `python preprocess_data.py` | Filter dataset by token length (<=512 tokens) |
| `python sources/validate_jsonl.py` | Validate JSONL dataset format |
| `python sources/combine_datasets.py` | Combine multiple datasets into one |

### Training Commands

| Command | Description |
|---------|-------------|
| `./train_cli.sh` | Run memory-optimized training (recommended for 8GB Mac) |
| `./start_training.sh` | Run training with custom Python script |
| `python train_lora.py` | Run custom LoRA training script directly |

### MLX-LM Training CLI (Advanced)

```bash
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
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to base model | Required |
| `--train` | Enable training mode | - |
| `--data` | Path to dataset directory (must contain train.jsonl, valid.jsonl) | Required |
| `--fine-tune-type` | Type: `lora`, `dora`, or `full` | `lora` |
| `--batch-size` | Samples per batch (lower = less memory) | 4 |
| `--num-layers` | Number of LoRA layers (lower = less memory) | 16 |
| `--iters` | Total training iterations | 1000 |
| `--val-batches` | Validation batches per eval | 25 |
| `--learning-rate` | Learning rate | 1e-5 |
| `--steps-per-report` | Steps between loss reports | 10 |
| `--steps-per-eval` | Steps between validations | 200 |
| `--save-every` | Steps between checkpoint saves | 100 |
| `--adapter-path` | Directory to save adapters | adapters |
| `--max-seq-length` | Maximum sequence length | 2048 |
| `--grad-checkpoint` | Enable gradient checkpointing (saves memory) | False |
| `--seed` | Random seed for reproducibility | - |

### Model Inference

#### Interactive Chat
```bash
mlx_lm.chat --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters
```

| Chat Command | Description |
|--------------|-------------|
| `q` | Quit chat session |
| `r` | Reset conversation history |
| `h` | Show help commands |
| `/no_think` | Prefix to disable thinking mode |

#### Text Generation
```bash
mlx_lm.generate \
    --model Qwen3-4B-Thinking-2507-MLX-4bit \
    --adapter-path adapters \
    --prompt "Write a Verilog module for a 4-bit counter" \
    --max-tokens 500
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to model | Required |
| `--adapter-path` | Path to LoRA adapters | None |
| `--prompt` | Input prompt text | Required |
| `--max-tokens` | Maximum tokens to generate | 100 |
| `--temp` | Temperature (0.0 = deterministic) | 0.0 |
| `--top-p` | Top-p sampling | 1.0 |

### System Commands (macOS)

| Command | Description |
|---------|-------------|
| `sysctl vm.swapusage` | Check swap memory usage |
| `sudo purge` | Clear system memory cache |
| `ls -lh /private/var/vm/` | Check swap files |

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TOKENIZERS_PARALLELISM` | Disable tokenizer parallelism | `false` |
| `MLX_METAL_MEMORY_LIMIT` | Limit Metal GPU memory (bytes) | `5368709120` (5GB) |

Set before running:
```bash
export TOKENIZERS_PARALLELISM=false
export MLX_METAL_MEMORY_LIMIT=5368709120
```

## Python API

### Load and Generate
```python
from mlx_lm import load, generate

# Load model with fine-tuned adapters
model, tokenizer = load(
    "Qwen3-4B-Thinking-2507-MLX-4bit",
    adapter_path="adapters"
)

# Generate Verilog code
prompt = "Write a Verilog module for a 4-bit synchronous counter with reset"
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
print(response)
```

### Streaming Generation
```python
from mlx_lm import load, stream_generate

model, tokenizer = load(
    "Qwen3-4B-Thinking-2507-MLX-4bit",
    adapter_path="adapters"
)

prompt = "Write a Verilog FSM for a traffic light controller"
for token in stream_generate(model, tokenizer, prompt=prompt, max_tokens=500):
    print(token, end="", flush=True)
```

### Load Without Adapters (Base Model)
```python
from mlx_lm import load, generate

# Load base model only
model, tokenizer = load("Qwen3-4B-Thinking-2507-MLX-4bit")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
```

## Example Prompts for Verilog Generation

```bash
# Simple counter
mlx_lm.generate --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters \
    --prompt "Write a Verilog module for a 4-bit synchronous counter with async reset" \
    --max-tokens 500

# FSM
mlx_lm.generate --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters \
    --prompt "Write a Verilog FSM for a vending machine that accepts 5 and 10 cent coins" \
    --max-tokens 800

# ALU
mlx_lm.generate --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters \
    --prompt "Write a Verilog 8-bit ALU with add, subtract, AND, OR operations" \
    --max-tokens 600

# FIFO
mlx_lm.generate --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters \
    --prompt "Write a Verilog synchronous FIFO with 8 entries of 16-bit width" \
    --max-tokens 800
```

## Training Results

| Metric | Value |
|--------|-------|
| Final Train Loss | 0.676 |
| Final Val Loss | 1.230 |
| Peak Memory | 3.135 GB |
| Training Speed | ~0.35 it/sec |
| Tokens/sec | ~79 |
| Total Tokens Trained | 1,194,342 |

## Memory Optimization (8GB Mac)

For Macs with limited RAM, the following optimizations are applied:

- 4-bit quantized base model
- LoRA with rank 4 (minimal)
- 4 LoRA layers only
- max_seq_length: 256
- batch_size: 1
- Gradient checkpointing enabled
- Metal memory limit: 5GB

## Known Limitations

1. **Thinking Mode**: The base model outputs reasoning before answers. Use direct prompts or add `/no_think` prefix.
2. **Sequence Length**: Training limited to 256 tokens due to memory constraints.
3. **Dataset Coverage**: 60% of original data filtered out due to length.

## License

This project uses:
- Qwen3 model: Subject to Qwen license terms
- MLX: Apache 2.0 License
- Training data: Various open-source Verilog datasets

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM tools for MLX
- [Qwen](https://github.com/QwenLM/Qwen) - Base model by Alibaba
