#!/bin/bash
# Q4: LLM agents (with and without the statistics tool) via vLLM.
#   sbatch slurm/q4.sbatch [extra args for scripts.run_q4_llm, e.g. --model google/gemma-3-12b-it]
#SBATCH --job-name=news-bandits-q4
#SBATCH --account=mihalcea_owned1
#SBATCH --partition=spgpu2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
module load python3.11-anaconda/2024.02
export HF_HOME=/scratch/mihalcea_owned_root/mihalcea_owned1/asamee/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT needs nvcc, which compute nodes lack
set -a; source .env; set +a
nvidia-smi -L

.venv-vllm/bin/python -m "${Q4_MODULE:-scripts.run_q4_llm}" "$@"
