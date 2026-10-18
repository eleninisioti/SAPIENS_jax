#!/bin/bash
# Short GPU check of the vLLM path: 2 iterations with small batches, sharing on (periodic with
# T_LOW=0 is always connected) and standard GRPO (centralized). Not an experiment.
#   cd ~/SAPIENS_jax/llm && sbatch scripts/gpu_test.sh
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --mail-type=NONE
set -euo pipefail
nvidia-smi -L
for topo in periodic centralized; do
  uv run --locked python -m sapiens_llm.train --config configs/math.yaml \
    --output "$HOME/SAPIENS_jax/results/llm_test/$topo" \
    --set TOPOLOGY=$topo ITERATIONS=2 PROMPTS_PER_ITER=4 T_LOW=0 EVAL_EVERY=2 EVAL_PROBLEMS=8 EVAL_SAMPLES=2
done
