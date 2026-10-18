#!/bin/bash
# Slurm array job: task i runs line i+1 of a runs file written by scripts/sweep.py.
# Each line is "<run_name> <arguments for sapiens.train>"; results go next to the runs file.
#SBATCH --gpus=1
#SBATCH --time=0-12
# A private Slack message from REAL HPC when this job ends:
#SBATCH --mail-type=END
#SBATCH --mail-user=slack

set -euo pipefail
RUNS_FILE=$1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
read -r NAME ARGS <<< "$LINE"
OUT="$(dirname "$RUNS_FILE")/$NAME"

echo "run $NAME on $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi -L
# tasks of one array start together in a fresh worktree: build its environment once, under a lock
flock .uv-sync.lock uv sync --locked --quiet
# shellcheck disable=SC2086
uv run --locked python -m sapiens.train $ARGS --output "$OUT"
