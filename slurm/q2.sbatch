#!/bin/bash
# Q2: reduce policy features, then run every contextual-bandit stage and the analysis.
#   sbatch --dependency=afterok:<features job> slurm/q2.sbatch
#SBATCH --job-name=news-bandits-q2
#SBATCH --account=mihalcea_owned1
#SBATCH --partition=spgpu2
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
module load python3.11-anaconda/2024.02
export HF_HOME=/scratch/mihalcea_owned_root/mihalcea_owned1/asamee/hf_cache
export PYTHONUNBUFFERED=1
nvidia-smi -L

[ -f dataset/processed/policy_features/meta.json ] || .venv/bin/python -m scripts.build_policy_features
# One process per stage, one worker per GPU (GPUs are in exclusive-process mode).
for stage in envs tune layers test churn; do
  .venv/bin/python -m scripts.run_q2 --stage $stage --workers 4
done
.venv/bin/python -m scripts.analyze_q2
