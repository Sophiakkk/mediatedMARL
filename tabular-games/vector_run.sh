#!/bin/bash
#SBATCH --job-name=mediated_IPC
#SBATCH --qos=cpu_qos
#SBATCH --mem=4G
#SBATCH --time=0-24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

python run.py