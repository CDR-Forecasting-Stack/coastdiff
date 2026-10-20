#!/bin/bash
#SBATCH --job-name=coastdiff_array
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=coastdiff_%A_%a.log

# Splits one continuous time range into N_TASKS contiguous slices; each array task
# runs its own slice on its own GPU (independent jobs, no communication). Files from
# different tasks never collide because their timestamps differ.
# Edit START/END/INTERVAL_H, and keep N_TASKS equal to the --array size above.
# Extra arguments are forwarded to `coastdiff` (region, --num-ensembles, ...):
#   sbatch scripts/run_inference_array.sbatch --lat-min 24 --lat-max 31 --lon-min -98 --lon-max -80
#
# Add your cluster's partition/account lines above and activate your environment below:
# source ~/miniforge3/etc/profile.d/conda.sh && conda activate coastdiff

START="2026-10-08 00:00"
END="2026-10-10 23:00"
INTERVAL_H=1
N_TASKS=4

start_s=$(date -u -d "$START" +%s)
end_s=$(date -u -d "$END" +%s)
step=$((INTERVAL_H * 3600))
total=$(( (end_s - start_s) / step + 1 ))            # timestamps in the range
per=$(( (total + N_TASKS - 1) / N_TASKS ))           # timestamps per task
first=$(( SLURM_ARRAY_TASK_ID * per ))
last=$(( first + per - 1 )); [ "$last" -ge "$total" ] && last=$((total - 1))
[ "$first" -ge "$total" ] && { echo "task $SLURM_ARRAY_TASK_ID: nothing to do"; exit 0; }

t0=$(date -u -d "@$((start_s + first * step))" "+%Y-%m-%d %H:%M")
t1=$(date -u -d "@$((start_s + last  * step))" "+%Y-%m-%d %H:%M")
echo "task $SLURM_ARRAY_TASK_ID: $t0 -> $t1"

coastdiff --start "$t0" --end "$t1" --interval "${INTERVAL_H}h" "$@"
