#!/bin/python

# Computes Fisher scores and retains greatest X features.

import time
start_time = time.time()
import sys
import os
import pandas as pd
import numpy as np
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
    # parser.add_argument('--thresholds', nargs='+', type=int, default=[50, 100, 150, 200], 
    #                     help='Thresholds to evaluate')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[50], 
                        help='Thresholds to evaluate')
    parser.add_argument('--number_of_features', type=int, default=500, help='Number of features to retain')
    return parser.parse_args()


# ----------------- #

if __name__ == '__main__':

    args = parse_arguments()

    for threshold in args.thresholds:
        matrix = pd.read_csv(f'{args.base_dir}/04_clustering/results/clustered_matrix_min{threshold}vals.csv', header=0)
        matrix = generalfuncs.set_dataset_name_as_index(matrix)
        print(f'Clustered matrix (threshold {threshold}): {matrix}')

        fisher_scores_dfs = generalfuncs.compute_fisher_scores(matrix, args.number_of_features)
        print(f'Fishers scores calculated for {threshold} threshold.')

        concat_df = pd.concat(fisher_scores_dfs)
        concat_df = concat_df[concat_df['FisherScore'] != np.inf] # remove infinity values
        
        concat_df.to_csv(f'{args.base_dir}/05_feature_selection/interim_data/top_{args.number_of_features}_fisher_scores_min{threshold}vals.csv', index = False)
        print(f'Fisher scores saved for {threshold} threshold', concat_df.head())

        print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')