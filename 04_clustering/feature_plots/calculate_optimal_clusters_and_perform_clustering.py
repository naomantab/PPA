#!/bin/python

import time
start_time = time.time()
import pandas as pd
import numpy as np
import os
import sys
import random
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# grandgrandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.append(grandgrandparent_dir)
# from funcs import generalfuncs

# add argparse function to allow for command line arguments and specifically, the base directory



# functions below

def set_dataset_name_as_index(input_matrix):
    """Set 'DatasetName' column to index.
    
    The DatasetName column contains string identifiers for each row's 
    source paper. When working with the matrix, this must be set as the 
    index to prevent errors with handling different data types (str/float).
    
    Input:
        input_matrix <pd.Dataframe>: Matrix with columns as features and rows as datasets
    
    Output:
        <pd.Dataframe> : Matrix with index set to 'DatasetName'
    """
    
    if 'DatasetName' not in input_matrix.index.names:
        input_matrix.set_index('DatasetName', inplace=True)

    return input_matrix


# ----------------- #

def calculate_optimal_clusters(matrix, filename):
    random_seed_list = random.sample(range(1, 1000), 100)
    prefixes = {}
    for ppsite in matrix.columns: # Loop over each column in the matrix
        prefix = ppsite.split('_')[0] # Get protein
        if prefix not in prefixes:
            prefixes[prefix] = [ppsite] #  Add prefix as dict key and phosphosite column name as values
        else:
            prefixes[prefix].append(ppsite) # Append phosphosite column name to prefix key if already exists in dict

    results = []
    for prefix, sites in prefixes.items(): # Iterate over keys and values in prefixes
        
        if len(sites) > 2: # If a protein has more than 2 phosphosites
            
            new_df = matrix[sites] #  Create df containing all data for given protein
            # impute missing values with mean of each column
            new_df = new_df.dropna(axis=1, how='all')
            new_df = new_df.apply(lambda x: x.fillna(x.mean()), axis=0)
            new_df = new_df.T # Transpose data for clustering
            new_df = new_df.drop_duplicates()

            #testing
            if new_df.isnull().values.any():
                print(f"NaNs still present in {prefix} data after imputation")
                print(new_df)
            
            row_data = {'Prefix': prefix}
            
            for random_seed in random_seed_list:
                # print(f'Current random seed:', random_seed)
                silhouette_avg = []
                
                # Iterate through possible number of clusters (i.e., total number of ppsites per protein)
                for i in range(2, len(new_df)):
                    kmeans = KMeans(n_clusters=i, random_state=random_seed) # Create KMeans object
                    kmeans.fit_predict(new_df)
                    if len(set(kmeans.labels_)) > 1: # Calculate silhouette scores
                        score = silhouette_score(new_df, kmeans.labels_, random_state=random_seed)
                    else:
                        print(f'Only one cluster was found for {prefix}. Silhouette score cannot be calculated.')
                        score = -1
                    silhouette_avg.append(score)# Store silhouette score in list
                    
                # check if silhouette_avg has any valid scores
                if silhouette_avg and max(silhouette_avg) > -1:
                    optimal_k = np.argmax(silhouette_avg) + 2
                    # print(f'Optimal number of clusters for {prefix} is {optimal_k}')
                else:
                    # handle the case where no valid clustering was found
                    print(f'No valid clustering results for {prefix}, setting default clusters to 1')
                    optimal_k = 1  # Default to 2 clusters if no valid silhouette score
                
                row_data[f'RandomSeed:{random_seed}'] = optimal_k    
                
            results.append(row_data)

    clusters_df = pd.DataFrame(results)
    clusters_df.set_index('Prefix', inplace=True)

    optimal_clusters = clusters_df.mode(axis=1)
    if len(optimal_clusters.columns) > 1:
        optimal_k = pd.DataFrame(optimal_clusters.mean(axis=1)).astype(int)
    else:
        optimal_k = optimal_clusters

    optimal_k.columns = ['ModeClusters']

    clusters_output = pd.concat([clusters_df, optimal_k], axis=1)
    # clusters_output.to_csv(f'/data/home/bt23917/PP/04_clustering/interim_data/{filename}.csv', index=True)
    clusters_output.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/interim_data/{filename}.csv', index=True)

    return clusters_output


# ----------------- #

def create_clustered_matrix_from_normalised_matrix(normalised_matrix, optimal_clusters, filename):
    optimal_clusters.reset_index(inplace=True)
    mode = optimal_clusters[['Prefix', 'ModeClusters']]
    
    prefixes = {}
    for ppsite in normalised_matrix.columns: # Loop over each column in the matrix
        prefix = ppsite.split('_')[0] # Get protein
        if prefix not in prefixes:
            prefixes[prefix] = [ppsite] # Add prefix as dict key and phosphosite column name as values
        else:
            prefixes[prefix].append(ppsite) # Append phosphosite column name to prefix key if already exists in dict

    clustered_matrix = normalised_matrix.copy()
    random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)

    new_cols = []
    for prefix, sites in prefixes.items(): # Iterate over keys and values in prefixes 

        if len(sites) == 1: # If a protein has only one phosphosite
            continue
        if len(sites) == 2: # If a protein has 2 phosphosites
            correlation = normalised_matrix[sites].corr().iloc[0, 1] # compute correlation between phosphosites
            if correlation > 0.6: # if correlation is greater than threshold
                average = normalised_matrix[sites].mean(axis=1) # create mean dataframe
                new_col_name = ' '.join(sites) # store ppsite names into one variable
                new_cols.append(pd.DataFrame({new_col_name: average})) # add new column to matrix
        if len(sites) > 2: # If a protein has more than 2 phosphosites
            optimal_k = mode.loc[mode['Prefix'] == prefix, 'ModeClusters'].iloc[0] # Get optimal cluster number for each protein  
            # print(f'Optimal_k for {prefix} is {optimal_k}. Imputing missing values...')    
            # Create df containing all data for given protein and transpose
            new_df = normalised_matrix[sites].apply(lambda x: x.fillna(x.mean()), axis=0).T
                
            # --------------- #
            # Create clusters and group phosphosites by cluster
            kmeans = KMeans(n_clusters=optimal_k, random_state=random_seed) # Create KMeans object with optimal number of clusters
            y = kmeans.fit_predict(new_df) # Fit data and get labels
            new_df['cluster'] = y # Create cluster column in existing df
            new_df = new_df.reset_index() # Move DatasetName from index to a column
            # Group phosphosites according to their cluster - removes DatasetName from output df
            ppsites_df = new_df.groupby('cluster').mean(numeric_only = True)
            ppsites_df = ppsites_df.sort_values(by='cluster') # Arrange df according to values in cluster column
                          
            # --------------- #
            # Create cluster names using phosphosites
            subset = new_df[['index', 'cluster']]
            # Concat phosphosites in same clusters with white space to form a cluster names
            cluster_names = pd.DataFrame(subset.groupby(['cluster'])['index'].transform(lambda x: ' '.join(x)))
            # Create 2-column df with cluster numbers and cluster names
            cluster_names_df = pd.concat([subset['cluster'], cluster_names], axis=1).drop_duplicates().sort_values(by='cluster')
            # Set cluster column to index
            cluster_names_df.set_index('cluster', inplace=True)
                
                
            # --------------- #
            # Create clustered matrix
            # ppsites_df and clusters_uniq are both indexed on cluster so can be concatenated
            averaged_groups = pd.concat([ppsites_df, cluster_names_df], axis=1, ignore_index=False)
            # Swap index from cluster numbers to DatasetNames
            averaged_groups.set_index('index', inplace=True)
            # Transpose so each cluster is a column of phosphorylation vals
            averaged_groups = averaged_groups.T
            new_cols.append(averaged_groups)
            # Remove phosphosites we've just clustered from initial matrix
            clustered_matrix.drop(columns=sites, inplace=True)
            print(f'Phosphosites for {prefix} have been clustered and added to matrix.')
        
    new_cols_df = pd.concat(new_cols, axis=1)     
    clustered_matrix = pd.concat([clustered_matrix, new_cols_df], axis=1)      

    mean_tolerance = 1e-8 # Remove any imputed values
    for col in clustered_matrix.columns:
        col_mean = clustered_matrix[col].mean() # calculate the column mean
        clustered_matrix[col] = clustered_matrix[col].mask((clustered_matrix[col] > col_mean - mean_tolerance) &
                                                        (clustered_matrix[col] < col_mean + mean_tolerance))
        
    clustered_array = np.array(clustered_matrix).astype(float)
    output_matrix = pd.DataFrame(clustered_array, index=clustered_matrix.index, columns=clustered_matrix.columns)

    output_matrix.reset_index(inplace=True)
    output_matrix.rename(columns={"index": "DatasetName"}, inplace=True)

    if output_matrix.columns[0] != 'DatasetName':
        phosphosite_ID = output_matrix.pop('DatasetName')
        output_matrix.insert(0, 'DatasetName', phosphosite_ID)
        
    # output_matrix.to_csv(f'/data/home/bt23917/PPA/04_clustering/interim_data/{filename}.csv', index=False)
    output_matrix.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/interim_data/{filename}.csv', index=False)
    
    print('Matrix with clustered phosphosites has been saved:', output_matrix.head())
    return output_matrix

# ----------------- #

if __name__ == "__main__":
    
    print("Loading normalised matrix...")
    # import the phosphosite-resolved and normalised matrix
    matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', header=0)
    # matrix = pd.read_csv('/data/home/bt23917/PPA/04_clustering/NormalisedMatrix(Quantile)(Z-score).csv', header=0)

    matrix = set_dataset_name_as_index(matrix)

    print("Calculating the optimal number of clusters per protein over 100 random seeds...")    
    optimal_clusters = calculate_optimal_clusters(matrix, 
                                                               filename="optimal_clusters")
    
    print("Performing clustering to group phosphosites...")
    clustered_matrix = create_clustered_matrix_from_normalised_matrix(matrix, 
                                                                                   optimal_clusters,
                                                                                   filename="clustered_matrix")
    print("Clustering complete. Clustered matrix has been saved.")


    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')
