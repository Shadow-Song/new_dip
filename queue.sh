#!/bin/bash

#SBATCH --job-name=dip_notebook
#SBATCH --partition=gpu_short
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --account=EENG013182
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# 使用提交 sbatch 时所在的目录
cd "$SLURM_SUBMIT_DIR"

source /software/local/languages/miniforge3/etc/profile.d/conda.sh
conda activate dip

echo "Host: $(hostname)"
echo "Directory: $(pwd)"
echo "Python: $(which python)"
echo "Started: $(date)"

# jupyter nbconvert \
#     --to notebook \
#     --execute super-resolution.ipynb \
#     --output result/notebook-executed-$(date +%Y-%m-%d_%H-%M-%S).ipynb \
#     --ExecutePreprocessor.timeout=-1 \
#     --ExecutePreprocessor.kernel_name=python3

# python main.py --function inpainting

echo "Finished: $(date)"
