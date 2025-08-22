#!/bin/python


# Input:
# ======
# columns:          [0] Feature [1] Coefficient [2] TargetFeature [3] FisherScore [4] Correlation
# rows:             Individual Feature-TargetFeature pairs and their corresponding coefficients, fisher scores and correlations


# ----------------- #
# Import dependencies
# ----------------- #

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
# prots = ("PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
#         "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3",
#         "PDPK1", "PTEN", "MTOR", "TSC1", "TSC2", "RHEB",
#         "FOXO1", "FOXO3","GSK3B", "BAD", "IRS1")

prots = (
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
    "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3",
    "PDPK1", "PTEN", "MTOR", "TSC1", "TSC2", "RHEB",
    "FOXO1", "FOXO3", "GSK3B", "BAD", "IRS1",
    "PIK3C2A", "PIK3C2B", "PIK3C2G",   # Class II PI3Ks
    "PI3KC2A", "PI3KC2B", "PI3KC2G"    # Class II PI3Ks (alternative names)
)


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    # parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/06_models', help='Base directory for the project')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    parser.add_argument('--threshold', type=int, default=100, help='Threshold to compute')

    return parser.parse_args()


# ----------------- #

if __name__ == "__main__":

    args = parse_arguments()

    print('Loading Biogrid database...')
    biogrid = pd.read_csv(f"{args.base_dir}/01_input_data/BIOGRID-ORGANISM-Mus_musculus-4.4.246.tab3.txt", sep="\t", header=0)
    print(biogrid.head(5))

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
    subset_filename = f"linear_regression_nested_cv_{network_name}_coefficients_protein_level_min{args.threshold}vals.csv"
    subset_df = pd.read_csv(f'{args.base_dir}/08_results/linear_regression/coefficients/{subset_filename}')
    subset_df = subset_df[['Feature', 'TargetFeature', 'Coeff*R2']]

    plots.plot_LR_predicted_network(
        args.base_dir,
        subset_df, 
        strong_percentile=1, 
        medium_percentile=3,
        weak_percentile=5, 
        selected_prots=prots, 
        save_as_filename=f"linear_regression_nested_cv_{network_name}_predicted_network_min{args.threshold}vals.html")
    
    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')

# ----------------- #
# select FEARMEN proteins
# ----------------- #

# dfs = []
# for i in prots:
#     target_feat_in_prots = coeffs[coeffs['TargetFeature'].str.contains(i)]
#     dfs.append(target_feat_in_prots)
#     predictive_feat_in_prots = coeffs[coeffs['Feature'].str.contains(i)]
#     dfs.append(predictive_feat_in_prots)

# dfs = pd.concat(dfs)            
# dfs = dfs[['TargetFeature', 'Feature', 'Coefficient']]
# dfs['Coefficient'] = pd.to_numeric(dfs['Coefficient'], errors='coerce')
# dfs['TargetFeature'] = dfs['TargetFeature'].str.split('_').str[0]
# dfs['Feature'] = dfs['Feature'].str.split('_').str[0]

# dfs.to_csv(f'/data/home/bty449/ExplainableAI/LR_Coefficients_{network_name}_Min{min_vals}Vals.csv', index=False)


# ----------------- #
# Draw predicted network for all our interactions within thresholds
# colouring nodes orange if in `prots`
# ----------------- #

# plots.plot_LR_predicted_network(coeffs, 
#                                 strong_boundary=6, medium_boundary=3,
#                                 weak_boundary=1, selected_prots=prots, 
#                                 save_as_filename=f"LR_{network_name}_PredictedNetwork_Coefficients_Min{min_vals}Vals.html")

# ----------------- #
# Retain our interactions, only if also in BIOGRID (only specific to proteins in `prots`)
# ----------------- #

# # store only interactors from BIOGRID file
# biogrid_cropped = biogrid_file[['Official Symbol Interactor A', 'Official Symbol Interactor B']]

# # rename biogrid file columns
# biogrid_cropped.columns = ['PredictiveFeature', 'TargetFeature']
# print('Interactions from BIOGRID file:', biogrid_cropped)

# # how many of our interactions are in the biogrid file
# merged_df = pd.merge(coeffs, biogrid_cropped, on=['PredictiveFeature', 'TargetFeature'], how='left', indicator='exists')

# # keep only our interactions found in biogrid file
# confirmed_interactions = merged_df[merged_df['exists'] == 'both'].drop('exists', axis=1)
# confirmed_interactions.to_csv(f'/data/home/bty449/ExplainableAI/LinearRegression/LR_CV_BIOGRID_ConfirmedInteractions_Min{min_vals}Vals.csv', index=False)
# print('Dataframe containing all confirmed_interactions and coefficients:', confirmed_interactions)


# # ----------------- #
# # Draw predicted network of BIOGRID-confirmed interactions colouring
# # nodes orange if in `prots`
# # ----------------- #

# plots.plot_LR_predicted_network(confirmed_interactions, 
#                           strong_boundary=11, 
#                           medium_boundary=6,
#                           weak_boundary=3, 
#                           selected_prots=prots, 
#                           save_as_filename=f'LR_BIOGRID_ConfirmedInteractions_PredictedNetwork_Coefficients_Min{min_vals}Vals.html')

# ----------------- #
# END TIMER
# ----------------- #

