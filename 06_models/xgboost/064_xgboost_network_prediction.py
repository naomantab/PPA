#!/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import time
start_time = time.time()
import os
import sys
import argparse

grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(grandparent_dir)
from funcs import plots

network_name = "PI3KAKT"  # Name of the network to compute
prots = ("PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
        "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3",
        "PDPK1", "PTEN", "MTOR", "TSC1", "TSC2", "RHEB",
        "FOXO1", "FOXO3","GSK3B", "BAD", "IRS1")

# prots = (
#     "PIK3CA",  # PI3K catalytic subunit (alpha) – initiates the pathway
#     "PIK3R1",  # PI3K regulatory subunit – regulates catalytic activity
#     "AKT1",    # Main effector kinase in many cell types
#     "PDPK1",   # Phosphorylates and activates AKT
#     "PTEN",    # Key negative regulator – dephosphorylates PIP3
#     "MTOR",    # Downstream effector – regulates growth/metabolism
#     "TSC2",    # Regulates RHEB–mTOR activation (optional inclusion)
#     "FOXO3",   # AKT-regulated transcription factor (optional inclusion)
# )

# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    # parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/PPA', help='Base directory for the project') 
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    parser.add_argument('--threshold', type=int, default=150, help='Threshold to compute')

    return parser.parse_args()


# ----------------- #

if __name__ == "__main__":

    args = parse_arguments()

    # print('Loading linear regression predicted interactions...')
    # lr_df = pd.read_csv(f"{args.base_dir}/08_results/linear_regression/coefficients/linear_regression_nested_cv_coefficients_min{args.threshold}vals.csv")

    # # protein level
    # lr_df['TargetFeature'] = lr_df['TargetFeature'].str.split('_').str[0]
    # lr_df['PredictiveFeature'] = lr_df['PredictiveFeature'].str.split('_').str[0]
    # lr_df = lr_df[['PredictiveFeature', 'TargetFeature', 'MedianCoeff']]
    # print('Protein level linear regression predictions:', lr_df)

    # plots.plot_LR_predicted_network(
    #     args.base_dir,
    #     lr_df, 
    #     strong_boundary=0.57, 
    #     medium_boundary=0.55,
    #     weak_boundary=0.53, 
    #     selected_prots=prots, 
    #     save_as_filename=f"linear_regression_nested_cv_predicted_network_min{args.threshold}vals.html")
    
    # ----------------- #
    
    print('Selecting PI3KAKT proteins...')
    subset_filename = f"xgboost_master_shap_file_cluster_level_min{args.threshold}vals_shapxr2.csv"
    # df = pd.read_csv(f'{args.base_dir}/xgboost/nested_cv_master_shaps/{subset_filename}')
    df = pd.read_csv(f'{args.base_dir}/06_models/xgboost/master_shaps_files/{subset_filename}')
    df['TargetFeature'] = df['TargetFeature'].str.split('_').str[0]
    df['PredictiveFeature'] = df['PredictiveFeature'].str.split('_').str[0]
    print('Protein level xgboost predictions:', df)
    subset_df = df[df.apply(lambda row: row['PredictiveFeature'] in prots or row['TargetFeature'] in prots, axis=1)]
    print('Subset xgboost predictions:', subset_df)

    # Before calling the plot function
    print(subset_df.columns)

    print(subset_df[subset_df['SHAP*R2'] > 0])

    plots.plot_predicted_network(
        base_dir=args.base_dir,
        model_type='xgboost',
        df=subset_df, 
        strong_percentile=1, 
        medium_percentile=3,
        weak_percentile=5, 
        selected_prots=prots,
        save_as_filename=f"xgboost_nested_cv_{network_name}_predicted_network_min{args.threshold}vals.html")
    
    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')


