import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np

# base_dir = "/data/home/bt23917/PPA/"
base_dir = "C:/Users/tnaom/OneDrive/Desktop/PPA/"

def load_data(base_dir, filepath):
    df = pd.read_csv(f"{base_dir}{filepath}")
    return df


# lr_df50 = load_data(base_dir, "08_results/linear_regression/results/linear_regression_cv_coefficients_min50vals.csv")
# lr_df100 = load_data(base_dir, "08_results/linear_regression/results/linear_regression_cv_coefficients_min100vals.csv")
# lr_df150 = load_data(base_dir, "08_results/linear_regression/results/linear_regression_cv_coefficients_min150vals.csv")
# lr_df200 = load_data(base_dir, "08_results/linear_regression/results/linear_regression_cv_coefficients_min200vals.csv")

lr_df50 = load_data(base_dir, "08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min50vals.csv")
lr_df100 = load_data(base_dir, "08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min100vals.csv")
lr_df150 = load_data(base_dir, "08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min150vals.csv")
lr_df200 = load_data(base_dir, "08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min200vals.csv")

def remove_outliers(df):
    if 'MeanR2' in df.columns:
        df.rename(columns={'MeanR2': 'mean_r2'}, inplace=True)
    filtered = df[(np.abs(zscore(df['mean_r2'])) < 3)]
    filtered = filtered.drop_duplicates(subset=['mean_r2'])
    return filtered

lr_df50 = remove_outliers(lr_df50)
lr_df100 = remove_outliers(lr_df100)
lr_df150 = remove_outliers(lr_df150)
lr_df200 = remove_outliers(lr_df200)

def plot_r2(base_dir, df50, df100, df150, df200, plot_title, save_path):
    plt.figure(figsize=(6, 5))

    for data, label, color in zip(
        [df50['mean_r2'], df100['mean_r2'], df150['mean_r2'], df200['mean_r2']],
        ['50', '100', '150', '200'],
        ['blue', 'orange', 'green', 'red']
    ):
        counts, bins = np.histogram(data, bins=25)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, counts, label=f"{label}", color=color, linewidth=2, alpha=0.5)
    plt.xlim(left=0)
    plt.xlabel('Mean R2', fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.yticks(fontsize=18)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    plt.text(0.52, 10, 'R2 = 0.5', color='red', fontsize=18)
    plt.legend(fontsize=10, loc='upper right')
    plt.title(f'{plot_title}', fontsize=20)
    plt.savefig(f"{base_dir}{save_path}", dpi=300, bbox_inches='tight')
    plt.show()

plot_r2(base_dir, lr_df50, lr_df100, lr_df150, lr_df200,  'Linear regression R² across thresholds', '08_results/linear_regression/plots/linear_regression_r2_across_thresholds.png')