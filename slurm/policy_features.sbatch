#!/bin/bash
# Extract policy-side article features: Gemma-3-1B and Gemma-3-12B (all-layer dense,
# Gemma Scope 2 SAE codes at the labeled layers) and bge-m3 sentence embeddings.
#   sbatch slurm/policy_features.sbatch   (from the project root on Great Lakes)
#SBATCH --job-name=news-bandits-features
#SBATCH --account=mihalcea_owned1
#SBATCH --partition=spgpu2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
module load python3.11-anaconda/2024.02
export HF_HOME=/scratch/mihalcea_owned_root/mihalcea_owned1/asamee/hf_cache
export PYTHONUNBUFFERED=1
set -a; source .env; set +a
nvidia-smi -L

[ -f dataset/processed/policy_raw/bge-m3/meta.json ] || .venv/bin/python -m scripts.extract_policy_features --model bge-m3
.venv/bin/python -m scripts.extract_policy_features --model gemma-3-1b --max-tokens 16384
.venv/bin/python -m scripts.extract_policy_features --model gemma-3-12b --max-tokens 16384
