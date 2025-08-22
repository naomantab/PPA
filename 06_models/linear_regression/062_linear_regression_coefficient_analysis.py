#!/bin/python

import time
start_time = time.time()
import pandas as pd
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

grandgrandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(grandgrandparent_dir)
from funcs import generalfuncs, mlfuncs


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    # parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/PPA/06_models', help='Base directory for the project')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    parser.add_argument('--threshold', type=int, default=50, help='Threshold to compute')

    return parser.parse_args()

def overlapping_coefficients_histogram(base_dir, thresholds):
    """Plots overlapping histograms of coefficient values for multiple thresholds."""

    plt.figure(figsize=(5, 6))

    for threshold in thresholds:
        # Load the SHAP values for the current threshold
        df = pd.read_csv(f'{base_dir}/08_evaluation/linear_regression/coefficients/linear_regression_cv_coefficients_min{threshold}vals.csv')

        # Extract the LogSHAPValue column
        df = df[['Feature', 'TargetFeature', 'MedianCoeff']] 
        sorted_df = df.sort_values(by=['MedianCoeff'], ascending=False)
        lr_coeff_vals = sorted_df['MedianCoeff']
        print(lr_coeff_vals)

        counts, bin_edges = np.histogram(lr_coeff_vals, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        window_size = 5  # Adjust the window size for more or less smoothing
        smoothed_counts = np.convolve(counts, np.ones(window_size) / window_size, mode='same')
        plt.xlim(-1, 1)
        plt.plot(bin_centers, smoothed_counts, label=f'Threshold: {threshold}', alpha=0.8)

        # Plot the histogram for the current threshold
        # plt.hist(lr_coeff_vals, bins=200, log=True, alpha=0.5, label=f'Threshold: {threshold}', display_style='stairs')

    # Add labels, title, and legend
    plt.xlabel('Median coefficient value', fontsize=20)
    plt.ylabel('Frequency of features', fontsize=20)
    plt.xticks(fontsize=16, ticks=np.arange(-1, 1.1, 0.5))
    plt.yticks(fontsize=16)
    # plt.legend(fontsize=16)
    plt.title('Linear regression', fontsize=20)
    # plt.title(f'Frequency distibution of linear regression\ncoefficient values across all thresholds')
    plt.savefig(f'{base_dir}/08_evaluation/linear_regression/plots/linear_regression_coefficients_histogram_all_thresholds.png', dpi=300, bbox_inches='tight')
    print(f"Overlapping histogram of coefficient values for all thresholds saved successfully!")


# ----------------- #

if __name__ == '__main__':   

    args = parse_arguments()

    overlapping_coefficients_histogram(
        base_dir=args.base_dir,
        thresholds=[50, 100, 150, 200]
    )

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')