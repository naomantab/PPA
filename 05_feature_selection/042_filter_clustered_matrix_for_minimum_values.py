#!/bin/python

# Takes the clustered matrix and filters it depending on the number of values for each feature.

import time
start_time = time.time()
import pandas as pd
import numpy as np
import sys
import os
import argparse

grandgrandparent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(grandgrandparent_dir)
from funcs import generalfuncs
print('Dependencies loaded.')


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA/', help='Base directory for the project')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[50, 100, 150, 200], 
                        help='Thresholds to evaluate')
    return parser.parse_args()


# ----------------- #

if __name__ == '__main__':

    args = parse_arguments()

    print('Loading clustered matrix...')
    matrix = pd.read_csv(f'{args.base_dir}/04_clustering/results/clustered_matrix.csv', header=0)
    matrix = generalfuncs.set_dataset_name_as_index(matrix)
    print(f'Clusters in unfiltered matrix: {len(matrix.columns)}')

    print('Generating one matrix per minimum value threshold...')
    for threshold in args.thresholds:
        matrix_filtered = matrix.loc[:, matrix.describe().loc['count'] > threshold]
        matrix_filtered.to_csv(f'{args.base_dir}/04_clustering/results/clustered_matrix_min{threshold}vals.csv', index=True)
        print(f'Clusters in filtered matrix (threshold {threshold}): {len(matrix_filtered.columns)}')

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')