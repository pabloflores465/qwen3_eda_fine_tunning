import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

# Disable tokenizer parallelism to avoid fork issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Limit Metal memory to leave room for system (5GB for MLX on 8GB Mac)
os.environ["MLX_METAL_MEMORY_LIMIT"] = "5368709120"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from mlx_lm import load, generate
from mlx_lm.tuner import utils as lora_utils
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train
from mlx_lm.tuner.datasets import TextDataset, CacheDataset


class VerilogTrainingConfig:
    """Configuration for Verilog fine-tuning optimized for 8GB RAM"""

    def __init__(self):
        # Model settings
        self.model_path = "Qwen3-4B-Thinking-2507-MLX-4bit"
        self.adapter_path = "adapters"

        # Dataset settings - use pre-processed filtered data
        self.train_file = "sources/processed_dataset/train.jsonl"
        self.valid_file = "sources/processed_dataset/valid.jsonl"

        # LoRA hyperparameters - reduced for 8GB RAM
        self.lora_rank = 4  # Reduced from 8 for memory
        self.lora_alpha = 8  # Scaling factor (typically 2x rank)
        self.lora_dropout = 0.05
        self.lora_layers = 8  # Reduced from 16 for memory

        # Training hyperparameters - aggressive memory optimization
        self.batch_size = 1  # Small batch for 8GB RAM
        self.gradient_accumulation_steps = 4  # Effective batch = 4
        self.learning_rate = 1e-4
        self.num_epochs = 1  # Start with 1 epoch for testing
        self.warmup_steps = 50  # Reduced for smaller dataset
        self.max_seq_length = 512  # Further reduced for 8GB RAM

        # Optimization settings
        self.grad_checkpoint = True  # Enable gradient checkpointing
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # Logging and saving
        self.save_every = 500
        self.eval_every = 1000  # Less frequent eval to save memory
        self.log_every = 10
        self.save_total_limit = 3

        # System settings
        self.seed = 42
        self.num_workers = 2


def format_verilog_prompt(instruction: str, output: str = None) -> str:
    """Format instruction-output pair into training prompt"""
    # Extract description if instruction contains JSON
    if instruction.startswith('Implement the following Verilog module:'):
        instruction = instruction.replace('Implement the following Verilog module:\n', '')

    if output is None:
        # Inference mode
        return f"""<|im_start|>system
You are an expert Verilog hardware designer. Generate clean, efficient, and correct Verilog code.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    else:
        # Training mode
        # Handle output being a list
        if isinstance(output, list):
            output = output[0] if len(output) > 0 else ""

        return f"""<|im_start|>system
You are an expert Verilog hardware designer. Generate clean, efficient, and correct Verilog code.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def load_verilog_dataset(file_path: str, max_samples: Optional[int] = None):
    """Load pre-processed JSONL dataset (already contains 'text' field)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                # Pre-processed data already has 'text' field
                data.append(item)

                if (i + 1) % 50000 == 0:
                    print(f"Loaded {i + 1} samples...")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {i + 1}")
                continue

    print(f"Total samples loaded: {len(data)}")
    return data


class MemoryMonitorCallback(TrainingCallback):
    """Callback to monitor memory usage during training"""

    def __init__(self):
        self.start_time = time.time()

    def on_train_step_end(self, step, loss, learning_rate):
        if step % 10 == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            print(f"Step {step} | Loss: {loss:.4f} | LR: {learning_rate:.2e} | Speed: {steps_per_sec:.2f} steps/s")


def main():
    print("=" * 60)
    print("Qwen3 Verilog LoRA Fine-tuning")
    print("=" * 60)

    config = VerilogTrainingConfig()

    # Set random seed
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    print(f"\nMLX Metal available: {mx.metal.is_available()}")
    print(f"Model: {config.model_path}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max sequence length: {config.max_seq_length}")

    # Load model and tokenizer
    print("\n" + "=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)
    model, tokenizer = load(config.model_path)

    print(f"Model loaded successfully")
    # Get vocabulary size from tokenizer
    try:
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.get_vocab())
        print(f"Vocabulary size: {vocab_size}")
    except:
        print("Tokenizer loaded successfully")

    # Apply LoRA layers to the model
    print("\n" + "=" * 60)
    print("Applying LoRA layers...")
    print("=" * 60)

    lora_config = {
        "rank": config.lora_rank,
        "scale": config.lora_alpha / config.lora_rank,  # scale = alpha / rank
        "dropout": config.lora_dropout,
    }

    lora_utils.linear_to_lora_layers(
        model,
        num_layers=config.lora_layers,
        config=lora_config,
    )

    # Print trainable parameters
    lora_utils.print_trainable_parameters(model)

    # Load datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    print(f"Train file: {config.train_file}")
    print(f"Valid file: {config.valid_file}")

    # Load pre-processed data (already filtered by token length)
    # Set max_samples to limit for testing, or None for full dataset
    max_train_samples = None  # Use full pre-processed dataset
    max_valid_samples = None

    train_data_raw = load_verilog_dataset(config.train_file, max_samples=max_train_samples)
    valid_data_raw = load_verilog_dataset(config.valid_file, max_samples=max_valid_samples)

    print(f"\nTrain samples: {len(train_data_raw)}")
    print(f"Valid samples: {len(valid_data_raw)}")

    # Wrap datasets with TextDataset and CacheDataset for proper tokenization
    train_text_dataset = TextDataset(train_data_raw, tokenizer, text_key="text")
    valid_text_dataset = TextDataset(valid_data_raw, tokenizer, text_key="text")

    # CacheDataset handles lazy tokenization and caching
    train_data = CacheDataset(train_text_dataset)
    valid_data = CacheDataset(valid_text_dataset)

    print("Datasets wrapped with TextDataset and CacheDataset")

    # Calculate training steps
    steps_per_epoch = len(train_data) // (config.batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs

    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {config.warmup_steps}")

    # Create adapter directory
    Path(config.adapter_path).mkdir(parents=True, exist_ok=True)
    adapter_file = f"{config.adapter_path}/adapters.safetensors"

    # Clear memory before training
    gc.collect()
    mx.metal.clear_cache()

    # Create training arguments (only valid parameters)
    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=total_steps,
        val_batches=5,  # Reduced from 25 for memory
        steps_per_report=config.log_every,
        steps_per_eval=config.eval_every,
        steps_per_save=config.save_every,
        adapter_file=adapter_file,
        max_seq_length=config.max_seq_length,
        grad_checkpoint=config.grad_checkpoint,
        grad_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Create learning rate schedule with warmup
    schedule_config = {
        "name": "cosine_decay",
        "arguments": [config.learning_rate, total_steps],
        "warmup": config.warmup_steps,
        "warmup_init": 0.0,
    }
    lr_schedule = lora_utils.build_schedule(schedule_config)

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )

    # Save training configuration
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(f"{config.adapter_path}/training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Save LoRA config for later loading
    lora_config_save = {
        "lora_layers": config.lora_layers,
        "lora_parameters": lora_config,
    }
    with open(f"{config.adapter_path}/adapter_config.json", 'w') as f:
        json.dump(lora_config_save, f, indent=2)

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Adapters will be saved to: {adapter_file}")
    print(f"Training config saved to: {config.adapter_path}/training_config.json")

    # Train using mlx-lm's built-in trainer
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_data,
            val_dataset=valid_data,
            args=training_args,
        )

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"Adapters saved to: {adapter_file}")
        print("\nTo use the fine-tuned model:")
        print(f"  model, tokenizer = load('{config.model_path}', adapter_path='{config.adapter_path}')")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Partial adapters may be available in: {config.adapter_path}")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
