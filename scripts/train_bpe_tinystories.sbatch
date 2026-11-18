#!/bin/bash
#SBATCH --job-name=tiny-train
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

python cs336_basics/tokenization/bpe.py --input_path data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --nproc 8 --output_dir ./results/tinystories_tok --special_tokens "<|endoftext|>"