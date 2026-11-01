#!/bin/bash
# Embed articles with Qwen3-Embedding-0.6B, then select, refit and evaluate the click model.
#   sbatch slurm/click_model.sbatch   (from the project root on Great Lakes)
#SBATCH --job-name=news-bandits-click
#SBATCH --account=mihalcea_owned1
#SBATCH --partition=spgpu2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
module load python3.11-anaconda/2024.02
export HF_HOME=/scratch/mihalcea_owned_root/mihalcea_owned1/asamee/hf_cache
export PYTHONUNBUFFERED=1
nvidia-smi -L

if [ ! -f dataset/processed/sim_qwen3_embeddings.npy ]; then
  .venv/bin/python -m scripts.embed_articles --batch-size 128 --device cuda
fi
.venv/bin/python -m scripts.train_click_model --device cuda "$@"
