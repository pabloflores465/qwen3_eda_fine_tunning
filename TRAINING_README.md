# Qwen3 Verilog LoRA Fine-tuning

Fine-tuning Qwen3-4B-Thinking model for Verilog code generation using LoRA (Low-Rank Adaptation).

## System Requirements

- Apple Silicon Mac (M1/M2/M3)
- 8GB+ RAM
- macOS with MLX support
- Python 3.9+

## Setup

The virtual environment `.venv-mlx` is already set up with required dependencies:
- mlx
- mlx-lm
- transformers

## Dataset

- **Training samples**: 697,226 (3.97 GB)
- **Validation samples**: 174,307 (1.00 GB)
- **Total**: 871,533 Verilog code samples
- **Sources**: PyraNet, VeriGen, MG-Verilog, RTL-Coder

## Training Configuration

### Memory-Optimized Settings (8GB RAM)

```yaml
LoRA Parameters:
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.05
  - Layers: 16/32

Training:
  - Batch size: 1
  - Gradient accumulation: 4 (effective batch = 4)
  - Learning rate: 1e-4
  - Epochs: 3
  - Max sequence length: 2048
  - Gradient checkpointing: Enabled
```

### Why LoRA?

- **Memory efficient**: Uses only ~0.1-1% of model parameters
- **Faster training**: 2-3x faster than full fine-tuning
- **Preserves base model**: Can switch between different adapters
- **Optimized for MLX**: Excellent Apple Silicon support

## Quick Start

### 1. Start Training

```bash
./start_training.sh
```

Or manually:

```bash
source .venv-mlx/bin/activate
python3 train_lora.py
```

### 2. Monitor Training

The script will show:
- Training loss
- Learning rate
- Training speed (steps/sec)
- Validation metrics

### 3. Test the Model

After training, test the fine-tuned model:

```bash
source .venv-mlx/bin/activate
python3 test_model.py
```

Test base model (without fine-tuning):

```bash
python3 test_model.py --base
```

Test with specific adapter path:

```bash
python3 test_model.py --adapter-path path/to/adapters
```

## Training Output

Adapters will be saved to `adapters/` directory:
- `adapters.safetensors` - LoRA weights
- `adapter_config.json` - LoRA configuration
- `training_config.json` - Training configuration

## Memory Management Tips

If you encounter memory issues:

1. **Reduce batch size** (in `train_lora.py`):
   ```python
   self.batch_size = 1  # Already at minimum
   ```

2. **Reduce sequence length**:
   ```python
   self.max_seq_length = 1024  # Default: 2048
   ```

3. **Reduce LoRA rank**:
   ```python
   self.lora_rank = 4  # Default: 8
   ```

4. **Reduce LoRA layers**:
   ```python
   self.lora_layers = 8  # Default: 16
   ```

## Performance Optimization

For better quality (if you have more RAM):

1. **Increase LoRA rank**:
   ```python
   self.lora_rank = 16  # or 32
   self.lora_alpha = 32  # or 64
   ```

2. **Increase batch size**:
   ```python
   self.batch_size = 2
   self.gradient_accumulation_steps = 8
   ```

3. **More layers**:
   ```python
   self.lora_layers = 32  # All layers
   ```

## Expected Training Time

With current settings:
- Steps per epoch: ~174,300
- Total steps: ~523,000 (3 epochs)
- Estimated time: 24-48 hours (depends on hardware)

For faster testing, edit `train_lora.py` to limit samples:

```python
train_data = load_verilog_dataset(config.train_file, max_samples=10000)
valid_data = load_verilog_dataset(config.valid_file, max_samples=1000)
```

## Using Fine-tuned Model

### In Python

```python
from mlx_lm import load, generate

# Load model with LoRA adapters
model, tokenizer = load(
    "Qwen3-4B-Thinking-2507-MLX-4bit",
    adapter_path="adapters"
)

# Generate Verilog
prompt = "Design a 4-bit counter with synchronous reset"
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

### Merge Adapters (Optional)

To create a standalone model with adapters merged:

```python
from mlx_lm.tuner import utils

# This will merge LoRA weights into the base model
utils.merge_adapters(
    model_path="Qwen3-4B-Thinking-2507-MLX-4bit",
    adapter_path="adapters",
    output_path="Qwen3-4B-Verilog-Merged"
)
```

## Files

- `train_lora.py` - Main training script
- `test_model.py` - Model testing and inference script
- `start_training.sh` - Training starter script
- `config.yaml` - Training configuration (reference)
- `base_model_test.py` - Original base model test

## Troubleshooting

### Out of Memory

- Reduce `max_seq_length` to 1024 or 512
- Reduce `lora_rank` to 4
- Close other applications

### Training Too Slow

- Use fewer samples for testing first
- Ensure no other heavy processes are running
- Check MLX is using Metal GPU

### Model Not Loading

- Verify model path: `Qwen3-4B-Thinking-2507-MLX-4bit`
- Check adapter path exists after training
- Ensure virtual environment is activated

## Notes

- Training can be interrupted (Ctrl+C) and resumed
- Checkpoints are saved every 1000 steps
- Best practices: Start with small sample for testing
- Monitor memory usage with Activity Monitor

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
