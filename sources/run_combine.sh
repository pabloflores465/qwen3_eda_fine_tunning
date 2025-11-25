#!/bin/bash
cd /Users/gp/Documents/qwen3_fine_tune/sources
/Users/gp/Documents/qwen3_fine_tune/.venv-mlx/bin/python combine_datasets.py > combine_log.txt 2>&1
echo "Script completed. Check combine_log.txt for results."
