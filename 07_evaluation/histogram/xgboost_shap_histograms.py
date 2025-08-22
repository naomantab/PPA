import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
import argparse
import os

# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Plot XGBoost SHAP value histograms across thresholds.')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    parser.add_argument('--thresholds', type=int, nargs='*', default=[50, 100, 150, 200], help='Thresholds to compute (space separated, e.g. --thresholds 50 100 150 200)')
    return parser.parse_args()

# ----------------- #
# SHAP values 

def plot_XGB_SHAP_histograms_across_thresholds(base_dir, thresholds):
    """
    Plots histograms of SHAPValue values for each threshold (one plot per threshold).
    """
    for threshold in thresholds:
        file_path = f"{base_dir}/06_models/xgboost/master_shaps_files/xgboost_master_shap_file_cluster_level_min{threshold}vals_shapxr2.csv"
        df = pd.read_csv(file_path)
        shap_vals = df['SHAP*R2'].dropna()
        plt.figure()
        plt.hist(shap_vals, bins=100)
        plt.xlabel('SHAPValue*R2')
        plt.ylabel('Frequency of Features')
        plt.title(f'XGBoost SHAPValue*R2 Distribution (Threshold: {threshold})')
        plt.grid(axis='y', linestyle='-', alpha=0.3)
        axs = plt.gca()
        axs.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        # Set x-ticks from -0.01 to 0.01 in steps of 0.005
        min_tick = np.floor(shap_vals.min() / 0.01) * 0.01
        max_tick = np.ceil(shap_vals.max() / 0.01) * 0.01
        xticks = np.arange(min_tick, max_tick + 0.01, 0.01)
        axs.set_xticks(xticks)
        plt.xticks(rotation=90, fontsize=8)
        plt.savefig(f"{base_dir}/08_results/xgboost/plots/XGB_SHAPxR2_Histogram_Min{threshold}Vals.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histogram for threshold {threshold} saved.")

# SHAP * R2 values

def plot_XGB_SHAPxR2_histograms_across_thresholds(base_dir, thresholds):
    """
    Plots histograms of SHAP*R2 values for each threshold (one plot per threshold).
    """
    for threshold in thresholds:
        file_path = f"{base_dir}/06_models/xgboost/master_shaps_files/xgboost_master_shap_file_cluster_level_min{threshold}vals_shapxr2.csv"

        if not os.path.exists(file_path):
            print(f"Skipping threshold {threshold}: CSV file not found.")
            continue

        df = pd.read_csv(file_path)
        shap_vals = df['SHAPValue'].dropna()
        plt.figure()
        plt.hist(shap_vals, bins=100)
        plt.xlabel('SHAPValue')
        plt.ylabel('Frequency of Features')
        plt.title(f'XGBoost SHAPValue Distribution (Threshold: {threshold})')
        plt.grid(axis='y', linestyle='-', alpha=0.3)
        axs = plt.gca()
        axs.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        min_tick = np.floor(shap_vals.min() / 0.01) * 0.01
        max_tick = np.ceil(shap_vals.max() / 0.01) * 0.01
        xticks = np.arange(min_tick, max_tick + 0.01, 0.01)
        axs.set_xticks(xticks)
        plt.xticks(rotation=90, fontsize=8)
        plt.xticks(rotation=90, fontsize=8)
        plt.savefig(f"{base_dir}/08_results/xgboost/plots/XGB_SHAPValue_Histogram_Min{threshold}Vals.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histogram for threshold {threshold} saved.")

def combine_xgb_histogram_pngs_grid(base_dir, thresholds):
    """
    Combines existing XGBoost histogram PNGs into a 2x4 grid and saves as a single PNG.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i, threshold in enumerate(thresholds):
        img1_path = f"{base_dir}/08_results/xgboost/plots/XGB_SHAPValue_Histogram_Min{threshold}Vals.png"
        img2_path = f"{base_dir}/08_results/xgboost/plots/XGB_SHAPxR2_Histogram_Min{threshold}Vals.png"

        # Skip this iteration if either file doesn't exist
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Skipping threshold {threshold}: Missing SHAP or SHAP*R2 image.")
            continue

        # SHAPValue
        img1 = mpimg.imread(img1_path)
        axs[0, i].imshow(img1)
        axs[0, i].axis('off')

        # SHAP*R2
        img2 = mpimg.imread(img2_path)
        axs[1, i].imshow(img2)
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{base_dir}/08_results/xgboost/plots/XGB_Histograms_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined XGBoost histogram grid saved.")

# ----------------- #


if __name__ == '__main__':
    args = parse_arguments()
    plot_XGB_SHAP_histograms_across_thresholds(args.base_dir, args.thresholds)
    plot_XGB_SHAPxR2_histograms_across_thresholds(args.base_dir, args.thresholds)
    combine_xgb_histogram_pngs_grid(args.base_dir, args.thresholds)

