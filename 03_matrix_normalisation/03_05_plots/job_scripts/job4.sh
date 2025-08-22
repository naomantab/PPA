#!/bin/bash
#$ -cwd                # Run job from current directory
#$ -pe smp 1           # Request 1 core
#$ -l h_rt=20:0:0      # Request 20 hours runtime
#$ -l h_vmem=64G       # Request 64GB of virtual memory
#$ -o logs/quantile_zscore_output.log  # Output log file
#$ -e logs/quantile_zscore_error.log   # Error log file
#$ -M n.tabassam@se23.qmul.ac.uk  # Email address to receive notifications
#$ -m e                 # Send email when the job ends

# Load necessary modules
module load python/3.11.7-gcc-12.2.0

# Activate the virtual environment
source ~/myenv/bin/activate

# Run the Python script
python /data/home/bt23917/PPA/03_matrix_normalisation/quantile_zscore_plot.py  # Update with your script name if different
