#!/bin/bash
#SBATCH --job-name=dlfront-perm
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dlfront_perm_%j.out
# Single-pass permutation importance over the model's input channels
# (user decision 2026-08-18: size how much skill lives in SLP/U10M/V10M --
# which carry no AIRS information -- before spending GPU on the stage-A
# channel ablation ladder).  All dl_front.permutation arguments pass
# through, e.g.
#   sbatch slurm/dlfront_permutation.sbatch \
#       --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f0/D6C-f0.h5" \
#       --classes 6 \
#       --source kriged-airs --years 2016-2018 --repeats 3
# (train.py and both chain scripts write checkpoints to
# RESULTS_DIR/models/<name>/; the models_v2 tree that this example used to
# cite is a superseded hand-made copy on one dev box, so the pasted command
# died with FileNotFoundError anywhere else -- path corrected to match
# dl_front.permutation's own docstring, review 2026-08-18.)
# CPU-ok by design, like slurm/dlfront_eval.sbatch: this is inference only,
# no gradients, so there is no --gres line here and the job runs on any
# node.  A permutation run IS much heavier than one eval leg -- it is
# (1 + n_channels * repeats) full inference passes AND the same number of
# neighborhood-CSI dilation passes, i.e. 16 of each at five channels and
# --repeats 3 -- hence --time 12:00:00 (2x the eval script) and 32G (the
# whole scored span sits in memory at once: a permutation shuffles along
# the sample axis, so it cannot be streamed year by year the way
# evaluate_test streams).  To put it on a GPU, add the flag at submit time:
#   sbatch --gres=gpu:1 slurm/dlfront_permutation.sbatch ...
# (the ablation chain script does exactly that when SBATCH_GRES is set).
# Output: $JPL_AIRS_RESULTS/dl_front/permutation/<ckpt-stem>_<source>.csv
#         + <ckpt-stem>_<source>_run.json (provenance, incl. labels_sha1)

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python -m dl_front.permutation "$@"
