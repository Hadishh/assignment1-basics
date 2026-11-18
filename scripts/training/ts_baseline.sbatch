#!/bin/bash
#SBATCH --job-name=ts-baseline-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out


export WANDB_DIR="$PWD/logs/wandb"
mkdir -p "$WANDB_DIR"
source .venv/bin/activate
python cs336_basics/train.py --config configs/ts/ts_baseline.yaml