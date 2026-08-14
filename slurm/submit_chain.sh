#!/bin/bash
# Submit the full stage A -> B curriculum as a SLURM dependency chain.
# Edit the exports for your cluster's paths, then:  bash slurm/submit_chain.sh
# Stage C (E3, real AIRS) is submitted manually against the AIRS_FCST_1deg
# fullgrid archive ($JPL_AIRS_FCST, 2003-2022; manifest reorg 2026-08-13)
# -- see docs/SLURM_DEPLOYMENT.md section 4.
#
# 2026-08-10: batch 64 @ lr 1e-4 (paper pairing), 10 steps/epoch, levels=3 /
# kernel 3 / filter_num [32,64,128] and dilation-0 labels are the in-repo
# DEFAULTS, sized for one A100-80GB -- no size overrides needed here.  If
# shards were copied from a dilation-1 era, materialize.sbatch's --labels-only
# pass rebuilds only y_*.npy (minutes, not hours).
set -euo pipefail

export JPL_AIRS_REPO=${JPL_AIRS_REPO:-$PWD}
export JPL_AIRS_DATA=${JPL_AIRS_DATA:-$JPL_AIRS_REPO/data}      # scratch FS
export JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-$JPL_AIRS_REPO/results}
mkdir -p logs

MAT=$(sbatch --parsable slurm/materialize.sbatch)
E1B=$(sbatch --parsable --dependency=afterok:"$MAT" slurm/train.sbatch \
      --name E1b-thermo --no-winds)
E2=$(sbatch --parsable --dependency=afterok:"$E1B" slurm/train.sbatch \
     --name E2-degraded --degraded --lr 5e-5 \
     --retrain "$JPL_AIRS_RESULTS/front_finder/models/E1b-thermo/E1b-thermo.h5")
E1A=$(sbatch --parsable --dependency=afterok:"$MAT" slurm/train.sbatch \
      --name E1a-wind --winds)

echo "submitted: materialize=$MAT E1b=$E1B E2=$E2 E1a=$E1A"
