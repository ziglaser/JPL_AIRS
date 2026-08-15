#!/bin/bash
#SBATCH --job-name=ff-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
# Generic training step: all front_finder.train arguments pass through, e.g.
#   sbatch slurm/train.sbatch --name E1b-thermo --no-winds
# In-repo defaults (2026-08-10) are sized for one A100-80GB: batch 64 @
# lr 1e-4 (the paper's pairing), 10 steps/epoch, levels=3/kernel-3 model,
# dilation-0 labels, flip augmentation on.
# Small GPUs: --batch 4 --steps 160 --lr 2.5e-5 --filter-num 16,32,64.

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src

python -m front_finder.train "$@"
