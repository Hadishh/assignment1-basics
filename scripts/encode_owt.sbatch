#!/bin/bash
#SBATCH --job-name=owt-tokenizer-encode
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

python cs336_basics/tokenization/bpe.py \
    --encode \
    --input_path data/owt_train.txt \
    --vocab_path results/owt_tok/vocab.pkl \
    --merges_path results/owt_tok/merges.pkl \
    --output ./results/owt_tokenized/train.npy \
    --stop_token "<|endoftext|>" \
    --special_tokens "<|endoftext|>"

python cs336_basics/tokenization/bpe.py \
    --encode \
    --input_path data/owt_valid.txt \
    --vocab_path results/owt_tok/vocab.pkl \
    --merges_path results/owt_tok/merges.pkl \
    --output ./results/owt_tokenized/valid.npy \
    --stop_token "<|endoftext|>" \
    --special_tokens "<|endoftext|>"