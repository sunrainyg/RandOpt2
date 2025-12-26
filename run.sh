#!/bin/bash
#SBATCH --job-name=es_sglang
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --account=a143
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --environment=/capstor/store/cscs/swissai/a143/yl/yulu.toml


srun --cpu-bind=none -ul \
  --container-writable \
  bash -c "
    cd /capstor/store/cscs/swissai/a143/yl/RandOpt2
    python3 -m llm_experiments.ensemble_inference \
      --num_noises 64 \
      --topk 8 \
      --sigma 1e-3 \
      --task gsm8ktrain \
      --temperature 1.0 \
      --ensemble_temperature 1.0