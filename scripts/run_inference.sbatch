#!/bin/bash
#SBATCH --job-name=coastdiff
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=coastdiff_%j.log

# One coastdiff run on one GPU. All arguments are forwarded to `coastdiff`:
#   sbatch scripts/run_inference.sbatch                       # the default Gulf of Mexico example
#   sbatch scripts/run_inference.sbatch --start "2026-10-08 00:00" --end "2026-10-09 23:00" --interval 1h
#
# coastdiff uses one GPU per run. To use more GPUs, submit several jobs (different
# time ranges, or different sub-regions each with its own --output-dir), or use
# run_inference_array.sbatch -- see README "Parallelizing".
#
# Add your cluster's partition/account lines above (#SBATCH --partition=..., --account=...)
# and activate your environment here:
# source ~/miniforge3/etc/profile.d/conda.sh && conda activate coastdiff

coastdiff "$@"
