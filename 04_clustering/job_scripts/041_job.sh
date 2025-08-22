#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=2:00:00       # Adjust as needed
#$ -l h_vmem=8G          # Adjust memory if needed
#$ -N split_matrices_job
#$ -o logs/split_matrices.out
#$ -e logs/split_matrices.err

# Load Python module (update version if needed)
module load python/3.10.4
source ~/myenv/bin/activate
# Run the script
python split_matrix.py