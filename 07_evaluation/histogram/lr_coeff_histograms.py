import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Plot LR coefficient histograms across thresholds.')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    parser.add_argument('--thresholds', type=int, nargs='*', default=[50, 100, 150, 200], help='Thresholds to compute (space separated, e.g. --thresholds 50 100 150 200)')
    return parser.parse_args()

# Coeff values 
def plot_LR_coeff_histograms_across_thresholds(base_dir, thresholds):
    """
    Plots histograms of MedianCoeff values for each threshold (one plot per threshold).
    """
    for threshold in thresholds:
        file_path = f"{base_dir}/08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min{threshold}vals.csv"
        df = pd.read_csv(file_path)
        coeffs = df['MedianCoeff'].dropna()
        plt.figure()
        plt.hist(coeffs, bins=100)
        plt.xlabel('MedianCoeff')
        plt.ylabel('Frequency of Features')
        plt.title(f'Linear Regression MedianCoeff Distribution (Threshold: {threshold})')
        plt.grid(axis='y', linestyle='-', alpha=0.3)
        plt.savefig(f"{base_dir}/08_results/linear_regression/plots/LR_MedianCoeff_Histogram_Min{threshold}Vals.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histogram for threshold {threshold} saved.")

# Coeff * R2 values
def plot_LR_CoeffxR2_histograms_across_thresholds(base_dir, thresholds):
    """
    Plots histograms of MedianCoeff*R2 values for each threshold (one plot per threshold).
    """
    for threshold in thresholds:
        file_path = f"{base_dir}/08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min{threshold}vals.csv"
        df = pd.read_csv(file_path)
        coeff_r2_vals = (df['Coeff*R2']).dropna()
        plt.figure()
        plt.hist(coeff_r2_vals, bins=100)
        plt.xlabel('Coeff*R2')
        plt.ylabel('Frequency of Features')
        plt.title(f'Linear Regression Coeff*R2 Distribution (Threshold: {threshold})')
        plt.grid(axis='y', linestyle='-', alpha=0.3)
        plt.savefig(f"{base_dir}/08_results/linear_regression/plots/LR_CoeffxR2_Histogram_Min{threshold}Vals.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Coeff*R2 histogram for threshold {threshold} saved.")

def combine_histogram_pngs_grid(base_dir, thresholds):
    """
    Combines existing LR histogram PNGs into a 2x4 grid and saves as a single PNG.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i, threshold in enumerate(thresholds):
        # Top row: MedianCoeff
        img1_path = f"{base_dir}/08_results/linear_regression/plots/LR_MedianCoeff_Histogram_Min{threshold}Vals.png"
        img1 = mpimg.imread(img1_path)
        axs[0, i].imshow(img1)
        axs[0, i].axis('off')
        # Bottom row: Coeff*R2
        img2_path = f"{base_dir}/08_results/linear_regression/plots/LR_CoeffxR2_Histogram_Min{threshold}Vals.png"
        img2 = mpimg.imread(img2_path)
        axs[1, i].imshow(img2)
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/08_results/linear_regression/plots/LR_Histograms_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined LR histogram grid saved.")

# ----------------- #

if __name__ == '__main__':
    args = parse_arguments()
    plot_LR_coeff_histograms_across_thresholds(args.base_dir, args.thresholds)
    plot_LR_CoeffxR2_histograms_across_thresholds(args.base_dir, args.thresholds)
    combine_histogram_pngs_grid(args.base_dir, args.thresholds)

