#!/bin/bash
#$ -cwd
#$ -pe smp 1
#$ -l h_rt=01:00:00
#$ -l h_vmem=8G
#$ -o /data/home/bt23917/PPA/04_clustering/logs/combination_job/combine_csvs_output.log
#$ -e /data/home/bt23917/PPA/04_clustering/logs/combination_job/combine_csvs_error.log
#$ -M n.tabassam@se23.qmul.ac.uk
#$ -m e

# Load your Python module (adjust if needed)
module load python/3.11.7-gcc-12.2.0

# Activate your environment if using one (optional)
source ~/myenv/bin/activate

# Run the Python script
python /data/home/bt23917/PPA/04_clustering/combination_job/combine_interim_data.py
