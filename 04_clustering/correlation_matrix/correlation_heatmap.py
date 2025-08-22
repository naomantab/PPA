#!/bin/python

import time
start_time = time.time()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import argparse
import fastcluster

# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    # parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/PPA', help='Base directory for the project')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    

    return parser.parse_args()

def create_clustered_matrix(base_dir):
    """Correlate all proteins in normalised matrix against each other."""

    norm_matrix = pd.read_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', header=0)
    norm_matrix.set_index('DatasetName', inplace=True)

    # Group columns by their first element after splitting by '_'
    grouped_columns = norm_matrix.columns.str.split('_').str[0]
    norm_matrix_grouped = norm_matrix.groupby(grouped_columns, axis=1).mean()

    # Compute the correlation matrix
    corr_matrix = norm_matrix_grouped.corr(method='spearman')
    corr_matrix = corr_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
    corr_matrix = corr_matrix.fillna(0)

    corr_matrix.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/correlation_matrix/correlation_matrix.csv', index=True)
    print('Correlation matrix:', corr_matrix)
    return corr_matrix

def plot_clustered_heatmap(base_dir, df, output_file, axis_labels=False):
    """Plot a clustered heatmap of the correlation matrix."""

    g = sns.clustermap(
    df,
    cmap='bwr',
    figsize=(10, 10),
    cbar_pos=(0.19, 0.8, 0.02, 0.15),
    cbar_kws={"shrink": .5},
    row_cluster=True,
    col_cluster=True,
    method='complete',
    xticklabels=False,
    yticklabels=False
    )

    # if axis_labels:
    #     g.ax_heatmap.tick_params(axis='x', labelrotation=90, labelsize=8)
    #     g.ax_heatmap.tick_params(axis='y', labelrotation=0, labelsize=8)


    plt.tight_layout()

    g.savefig(f'{base_dir}/04_clustering/correlation_matrix/{output_file}.png', dpi=300)

# def subset_corr_matrix_for_kinases_phosphatases(base_dir, df):
#     """Subset the correlation matrix for kinases and phosphatases."""

#     df = pd.read_csv(f'{base_dir}/04_clustering/hierarchical_clustering/kinases_phosphatases.csv', header=0)
#     kinases = df['kinases'].dropna().tolist()
#     phosphatases = df['phosphatases'].dropna().tolist()
#     print(kinases)
#     print(phosphatases)

#     prots_to_keep = set(kinases + phosphatases)
#     print(prots_to_keep)

#     filtered_df = df.loc[df.index.intersection(prots_to_keep), df.columns.intersection(prots_to_keep)]
#     filtered_df = filtered_df.rename(index=lambda x: x + '_kin' if x in kinases else x + '_phos' if x in phosphatases else x)
#     print(f"Matrix filtered for kinases and phosphatases: {filtered_df}")
    # return filtered_df

def subset_highly_correlated_proteins(df, threshold):
    """Subset the correlation matrix for highly correlated proteins."""

    positive_corr = (df > 0).sum(axis=0) / len(df)
    negative_corr = (df < 0).sum(axis=0) / len(df)

    filtered_cols = df.columns[(positive_corr > threshold) | (negative_corr > threshold)]
    filtered_df = df.loc[filtered_cols, filtered_cols]
    print(f"Matrix filtered for proteins that highly correlate with >50% of proteins: {filtered_df}")
    return filtered_df



# ----------------- #

if __name__ == '__main__':

    args = parse_arguments()

    # print("Creating clustered matrix...")
    # corr_matrix = create_clustered_matrix(args.base_dir)

    corr_matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/correlation_matrix/correlation_matrix.csv', header=0, index_col=0)

    # df = pd.read_csv(f'{args.base_dir}/04_clustering/hierarchical_clustering/corr_ordered.csv', header=0, index_col=0)
    # print(df)
    
    print("Plotting clustered heatmap for all proteins...")
    plot_clustered_heatmap(
        base_dir=args.base_dir, 
        df=corr_matrix,
        output_file="corr_ordered",
        axis_labels=False
    )
    
    # print("Subsetting and plotting clustered heatmap for kinases and phosphatases...")
    # kin_phos_corr_matrix = subset_corr_matrix_for_kinases_phosphatases(
    #     base_dir=args.base_dir,
    #     df=corr_matrix)
    
    # plot_clustered_heatmap(
    #     base_dir=args.base_dir,
    #     df=kin_phos_corr_matrix,
    #     output_file="x_correlation_matrix_heatmap_kinases_phosphatases",
    #     axis_labels=True
    # )

    # print("Subsetting and plotting clustered heatmap for highly correlated proteins...")
    # highly_correlated_proteins = subset_highly_correlated_proteins(
    #     df=corr_matrix,
    #     threshold=0.3
    # )

    # plot_clustered_heatmap(
    #     base_dir=args.base_dir,
    #     df=highly_correlated_proteins,
    #     output_file="x_correlation_matrix_heatmap_highly_correlated_proteins",
    #     axis_labels=False
    # )

    # print("Subsetting and plotting clustered heatmap for highly correlated kinase or phosphatase proteins...")
    # highly_correlated_kin_phos_prots = subset_highly_correlated_proteins(
    #     df=kin_phos_corr_matrix,
    #     threshold=0.5
    # )

    # plot_clustered_heatmap(
    #     base_dir=args.base_dir,
    #     df=highly_correlated_kin_phos_prots,
    #     output_file="x_correlation_matrix_heatmap_highly_correlated_kinases_phosphatases",
    #     axis_labels=True
    # )
    
    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')

