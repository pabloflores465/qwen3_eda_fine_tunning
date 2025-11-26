import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate


def format_verilog_prompt(instruction: str) -> str:
    """Format instruction for inference"""
    if instruction.startswith('Implement the following Verilog module:'):
        instruction = instruction.replace('Implement the following Verilog module:\n', '')

    return f"""<|im_start|>system
You are an expert Verilog hardware designer. Generate clean, efficient, and correct Verilog code.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


def test_model(adapter_path: str = None, use_base_model: bool = False):
    """Test the fine-tuned or base model"""

    print("=" * 60)
    if use_base_model:
        print("Testing BASE model (no adapters)")
    else:
        print(f"Testing FINE-TUNED model")
        print(f"Adapter path: {adapter_path}")
    print("=" * 60)

    # Load model
    model_path = "Qwen3-4B-Thinking-2507-MLX-4bit"

    print(f"\nLoading model: {model_path}")
    if use_base_model:
        model, tokenizer = load(model_path)
    else:
        if not adapter_path or not Path(adapter_path).exists():
            print(f"Error: Adapter path '{adapter_path}' not found")
            print("Use --base to test the base model without adapters")
            sys.exit(1)
        model, tokenizer = load(model_path, adapter_path=adapter_path)

    print("Model loaded successfully!\n")

    # Test prompts
    test_prompts = [
        "Design a 4-bit counter with synchronous reset",
        "Create a parameterized FIFO buffer with configurable depth",
        "Implement a simple ALU with add, subtract, AND, and OR operations",
    ]

    print("=" * 60)
    print("Running test prompts...")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"\nGenerating...")

        formatted_prompt = format_verilog_prompt(prompt)

        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=512,
            temp=0.1,  # Low temperature for more deterministic output
            top_p=0.9,
            verbose=False,
        )

        print(f"\nGenerated Verilog:\n")
        print(response)
        print(f"\n{'=' * 60}\n")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive mode - Enter your prompts (or 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            user_prompt = input("\nYour prompt: ").strip()

            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_prompt:
                continue

            formatted_prompt = format_verilog_prompt(user_prompt)

            print("\nGenerating...\n")
            response = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=512,
                temp=0.1,
                top_p=0.9,
                verbose=False,
            )

            print(f"\nGenerated Verilog:\n")
            print(response)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen3 Verilog model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters",
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Test base model without adapters"
    )

    args = parser.parse_args()

    test_model(
        adapter_path=args.adapter_path,
        use_base_model=args.base
    )


if __name__ == "__main__":
    main()
