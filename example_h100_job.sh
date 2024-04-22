#!/bin/bash
#SBATCH -JExJob
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 80GB
#SBATCH -G H100:1
#SBATCH -t 16:00:00
#SBATCH -oReport-%j.out

# load required modules
module load anaconda3
module load gcc/12.3.0
module load python/3.10.10
module load cuda/12.1.1
module load mvapich2/2.3.7-1 

echo "Start script"

# insert path to script
cd <path to script>

# insert your own env if it has a different name
srun ~/.conda/envs/h100_torch/bin/python <script>