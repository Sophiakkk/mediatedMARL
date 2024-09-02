#!/bin/bash
#SBATCH --qos=deadline
#SBATCH --mem=4G
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --array=1-10

python run.py