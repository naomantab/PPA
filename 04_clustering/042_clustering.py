#!/bin/python

import time
start_time = time.time()

import pandas as pd
import numpy as np
import os
import sys
import random
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import logging

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_dataset_name_as_index(input_matrix):
    if 'DatasetName' in input_matrix.columns:
        input_matrix.set_index('DatasetName', inplace=True)
    return input_matrix

def process_prefix(prefix, sites, matrix, random_seed_list):
    if len(sites) <= 2:
        return None

    # Drop columns that are all NaN first
    new_df = matrix[sites].dropna(axis=1, how='all')

    # Impute remaining NaNs with column means
    new_df = new_df.apply(lambda x: x.fillna(x.mean()), axis=0)

    # Transpose and drop duplicates before clustering
    new_df = new_df.T.drop_duplicates()

    if new_df.isnull().values.any():
        logging.warning(f"NaNs still present in {prefix} data after imputation")

    row_data = {'Prefix': prefix}

    for random_seed in random_seed_list:
        silhouette_avg = []
        # cluster counts from 2 up to number of samples
        for i in range(2, len(new_df)):
            kmeans = KMeans(n_clusters=i, random_state=random_seed)
            labels = kmeans.fit_predict(new_df)
            if len(set(labels)) > 1:
                score = silhouette_score(new_df, labels, random_state=random_seed)
            else:
                score = -1
            silhouette_avg.append(score)

        if silhouette_avg and max(silhouette_avg) > -1:
            optimal_k = np.argmax(silhouette_avg) + 2
        else:
            optimal_k = 1

        row_data[f'RandomSeed:{random_seed}'] = optimal_k

    return row_data

def calculate_optimal_clusters(matrix, filename):
    random_seed_list = random.sample(range(1, 1000), 100)
    prefixes = {}
    for ppsite in matrix.columns:
        prefix = ppsite.split('_')[0]
        prefixes.setdefault(prefix, []).append(ppsite)

    results = []
    for prefix, sites in prefixes.items():
        row = process_prefix(prefix, sites, matrix, random_seed_list)
        if row is not None:
            results.append(row)

    clusters_df = pd.DataFrame(results)
    clusters_df.set_index('Prefix', inplace=True)

    optimal_clusters = clusters_df.mode(axis=1)
    if len(optimal_clusters.columns) > 1:
        optimal_k = pd.DataFrame(optimal_clusters.mean(axis=1)).astype(int)
    else:
        optimal_k = optimal_clusters

    optimal_k.columns = ['ModeClusters']

    clusters_output = pd.concat([clusters_df, optimal_k], axis=1)
    outpath = f'/data/home/bt23917/PPA/04_clustering/interim_data/{filename}_optimal_clusters.csv'
    clusters_output.to_csv(outpath, index=True)

    logging.info(f"Optimal clusters saved to {outpath}")

    return clusters_output

def create_clustered_matrix_from_normalised_matrix(normalised_matrix, optimal_clusters, filename):
    logging.info("Starting clustering of phosphosites based on optimal clusters...")

    optimal_clusters = optimal_clusters.reset_index()
    mode = optimal_clusters[['Prefix', 'ModeClusters']]

    prefixes = {}
    for ppsite in normalised_matrix.columns:
        prefix = ppsite.split('_')[0]
        prefixes.setdefault(prefix, []).append(ppsite)

    clustered_matrix = normalised_matrix.copy()
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

    new_cols = []

    for prefix, sites in prefixes.items():
        logging.info(f"Processing prefix: {prefix} with {len(sites)} sites")

        if len(sites) == 1:
            continue

        if len(sites) == 2:
            correlation = normalised_matrix[sites].corr().iloc[0, 1]
            if correlation > 0.6:
                average = normalised_matrix[sites].mean(axis=1)
                new_col_name = ' '.join(sites)
                new_cols.append(pd.DataFrame({new_col_name: average}))
                logging.info(f"Two-site group {prefix} merged with correlation {correlation:.2f}")
            continue

        try:
            optimal_k = mode.loc[mode['Prefix'] == prefix, 'ModeClusters'].iloc[0]
        except IndexError:
            logging.warning(f"No optimal cluster found for prefix {prefix}. Skipping.")
            continue

        if not isinstance(optimal_k, (int, np.integer)) or optimal_k < 2:
            logging.warning(f"Invalid cluster count ({optimal_k}) for prefix {prefix}. Skipping.")
            continue

        # Clean and prepare data
        new_df = normalised_matrix[sites].dropna(axis=1, how='all')
        if new_df.empty:
            logging.warning(f"No valid data for prefix {prefix} after dropping empty columns. Skipping.")
            continue

        new_df = new_df.apply(lambda x: x.fillna(x.mean()), axis=0).T
        new_df = new_df.drop_duplicates()

        if new_df.empty or new_df.shape[0] < optimal_k:
            logging.warning(f"Not enough unique samples for clustering in prefix {prefix}. Skipping.")
            continue

        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=random_seed)
            new_df['cluster'] = kmeans.fit_predict(new_df)

            new_df = new_df.reset_index()
            cluster_means = new_df.groupby('cluster').mean(numeric_only=True).sort_values(by='cluster')

            cluster_names = new_df[['index', 'cluster']].groupby('cluster')['index'].transform(lambda x: ' '.join(x))
            cluster_names_df = pd.concat([new_df['cluster'], cluster_names], axis=1).drop_duplicates().sort_values(by='cluster')
            cluster_names_df.set_index('cluster', inplace=True)

            averaged_groups = pd.concat([cluster_means, cluster_names_df], axis=1)
            averaged_groups.set_index('index', inplace=True)
            averaged_groups = averaged_groups.T

            new_cols.append(averaged_groups)
            clustered_matrix.drop(columns=sites, inplace=True)

            logging.info(f"Clustered phosphosites for {prefix} into {optimal_k} groups.")
        except Exception as e:
            logging.error(f"Clustering failed for prefix {prefix}: {e}")
            continue

    if not new_cols:
        logging.warning("No clustered phosphosites were generated. Returning unmodified matrix.")
        return clustered_matrix

    try:
        new_cols_df = pd.concat(new_cols, axis=1)
        clustered_matrix = pd.concat([clustered_matrix, new_cols_df], axis=1)
    except Exception as e:
        logging.error(f"Failed to concatenate new clustered columns: {e}")
        return clustered_matrix

    mean_tolerance = 1e-8
    for col in clustered_matrix.columns:
        col_mean = clustered_matrix[col].mean()
        clustered_matrix[col] = clustered_matrix[col].mask(
            (clustered_matrix[col] > col_mean - mean_tolerance) &
            (clustered_matrix[col] < col_mean + mean_tolerance)
        )

    clustered_array = np.array(clustered_matrix).astype(float)
    output_matrix = pd.DataFrame(clustered_array, index=clustered_matrix.index, columns=clustered_matrix.columns)

    output_matrix.reset_index(inplace=True)
    output_matrix.rename(columns={"index": "DatasetName"}, inplace=True)

    if output_matrix.columns[0] != 'DatasetName':
        phosphosite_ID = output_matrix.pop('DatasetName')
        output_matrix.insert(0, 'DatasetName', phosphosite_ID)

    outpath = f'/data/home/bt23917/PPA/04_clustering/interim_data/{filename}_clustered_matrix.csv'
    try:
        output_matrix.to_csv(outpath, index=False)
        logging.info(f'Matrix with clustered phosphosites saved to {outpath}')
    except Exception as e:
        logging.error(f"Failed to write clustered matrix to file: {e}")

    return output_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to input split matrix CSV')
    args = parser.parse_args()

    input_path = args.file
    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    logging.info(f"Loading matrix from: {input_path}")
    matrix = pd.read_csv(input_path, header=0)
    matrix = set_dataset_name_as_index(matrix)

    logging.info("Calculating the optimal number of clusters per protein over 100 random seeds...")
    optimal_clusters = calculate_optimal_clusters(matrix, filename=base_filename)

    logging.info("Performing clustering to group phosphosites...")
    clustered_matrix = create_clustered_matrix_from_normalised_matrix(matrix, optimal_clusters, filename=base_filename)

    logging.info("Clustering complete.")
    logging.info(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes.')
