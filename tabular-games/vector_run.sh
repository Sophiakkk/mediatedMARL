#!/bin/bash
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

python run.py