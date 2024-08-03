#!/bin/bash
#SBATCH --mem=4GB
#SBATCH --ntasks=4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

python run.py --array_task_id $SLURM_ARRAY_TASK_ID