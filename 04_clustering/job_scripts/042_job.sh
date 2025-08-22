#!/bin/bash
#$ -cwd
#$ -t 1-140                # array job for 140 files
#$ -pe smp 1               # 1 core per task
#$ -l h_rt=48:0:0
#$ -l h_vmem=6G
#$ -o /data/home/bt23917/PPA/04_clustering/logs/clustering2_array.log
#$ -j y
#$ -N clustering2_array

##$ -M n.tabassam@se23.qmul.ac.uk
##$ -m e

module load python/3.11.7-gcc-12.2.0
source ~/myenv/bin/activate

MATRIX_DIR="/data/home/bt23917/PPA/04_clustering/split_matrices"
SCRIPT="/data/home/bt23917/PPA/04_clustering/clustering2.py"
OUTPUT_DIR="/data/home/bt23917/PPA/04_clustering/interim_data"

# get the file for this job array task
FILE=$(ls $MATRIX_DIR/*.csv | sed -n "${SGE_TASK_ID}p")

echo "Running on file: $FILE"

python $SCRIPT --file "$FILE"
