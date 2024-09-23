#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=IPC_mediator
#SBATCH --mem=4G
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=10
#SBATCH --array=1-10

source /scratch/ssd004/scratch/shuhui/miniconda3/etc/profile.d/conda.sh
conda activate contracting

python run.py