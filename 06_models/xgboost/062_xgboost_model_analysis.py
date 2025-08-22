#!/bin/python
"""Analysis of XGBoost outputs before SHAP computation."""

import time
start_time = time.time()
import pandas as pd
import os
import sys
import argparse

grandgrandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(grandgrandparent_dir)
from funcs import mlfuncs


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/PPA', help='Base directory for the project')
    parser.add_argument('--threshold', type=int, default=50, help='Threshold to compute')

    return parser.parse_args()


def create_master_results_file(threshold):
    """Concatenates the best models into a single file.
    
    Iterates through files in a directory, selecting the optimal model 
    parameters from each, and saves these to a new file. The output 
    CSV contains one row per feature, with optimal model parameters 
    stored in columns.
    
    Input:
        loss_function <str>: One of 'mse', 'mae', or 'r2'
        min_vals <int>: Threshold value to specify minimum values per feature
        model_type <str>: One of 'cnn' or 'xgboost'
        
    Output:
        CSV file <csv>: Concatenated file of best models and parameters
    """
    df = pd.DataFrame(columns=['TargetFeature', 'colsample_bytree', 'gamma', 'max_depth', 
                               'min_child_weight', 'n_estimators', 'subsample', 'mean_mse', 
                               'mean_r2'])
    
    for filename in os.listdir(f"/data/home/bt23917/PPA/xgboost/nested_cv_results_files"):
        if f"min{threshold}vals" in filename:
            with open(f'/data/home/bt23917/PPA/xgboost/nested_cv_results_files/{filename}', 'r') as FILE:
                current_file = pd.read_csv(FILE)
                current_row = pd.DataFrame(current_file.loc[0]).T
                df = pd.concat([df, current_row], ignore_index=True)

    df.to_csv(f'/data/home/bt23917/PPA/xgboost/results_files/xgboost_nested_cv_master_results_file_min{threshold}vals.csv', index=False)
    return df



# ----------------- #

if __name__ == '__main__':

    args = parse_arguments()
    
    """Concatenate optimal models into single file containing best model for each cluster."""
    print(f'Concatenating optimal XGBoost models...')
    create_master_results_file(args.threshold)

    """Concatenate optimal models into single file containing best model for each cluster."""
    print(f'Concatenating optimal CNN models...')
    mlfuncs.concat_best_models_all_clusters('mse', args.threshold, 'xgboost')
    
    print(f'Identifying XGBoost models for retraining...')
    mlfuncs.identify_models_for_retraining(args.base_dir, args.threshold, 'xgboost')

    print(f'Concatenating and logging SHAP files...')
    mlfuncs.create_master_shaps_file(args.threshold, 'xgboost')

    # print(f'Grabbing the product of SHAP values and mean R2 and adding onto a new column ')
    # mlfuncs.mult_shap_r2(args.threshold, 'xgboost')
    
    print(f'Execution time: {time.time() - start_time} seconds, {(time.time() - start_time)/60} minutes, {(time.time() - start_time)/3600} hours.')


