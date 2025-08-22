#!/bin/python

import pandas as pd
import numpy as np
import random
import copy
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------- #    

def compute_fisher_scores(input_matrix, threshold):
    # warnings.filterwarnings('ignore', category=RuntimeWarning)
    """Computes Fisher scores and retains greatest X features.

    Selects each column individually, calculates the Fisher scores for all features
    with respect to that column, and retains a `threshold` number of features with 
    the greatest Fisher scores.
    
    Input:
        input_matrix <pd.Dataframe>: Matrix of phosphoproteomics data, from which `X` and `y` will be created
        threshold <int>: Threshold for how many predictive features to retain
    
    Output:
        list <[pd.DataFrame]>: List of dataframes containing top features and their Fisher scores for each feature of interest in the matrix
    """
    
    # list to store dataframes
    fisherScore_dfs = []

    cols_remaining = len(input_matrix.columns)

    # create X and y dataframes
    for column in input_matrix.columns:
        cols_remaining -= 1
        current_df = pd.DataFrame()
        # y = feature of interest
        y = input_matrix[column]
        feat_of_interest = y.name
        data_rem = input_matrix.drop(columns = column)
        for col in data_rem.columns:
            gene_name = col
            # X = features to compute Fisher scores for, with respect to y
            X = data_rem[col]
            
            in_zero = [0] if (y == 0).any() else [np.nan]
            in_low = [y[(y > 0) & (y <= 1/3)].mean()]
            in_medium = [y[(y > 1/3) & (y <= 2/3)].mean()]
            in_high = [y[(y > 2/3) & (y <= 1)].mean()]
            inputAvg = pd.DataFrame(data = {'zero': in_zero, 'low': in_low, 'medium': in_medium, 'high': in_high})
    
            ft_zero = [X[(y == 0)].mean()]
            ft_low = [X[(y > 0) & (y <= 1/3)].mean()]
            ft_medium = [X[(y > 1/3) & (y <= 2/3)].mean()]
            ft_high = [X[(y > 2/3) & (y <= 1)].mean()]
            featAvg = pd.DataFrame(data = {'zero': ft_zero, 'low': ft_low, 'medium': ft_medium, 'high': ft_high})
    
            ftstd_zero = [X[(y == 0)].std()]
            ftstd_low = [X[(y > 0) & (y <= 1/3)].std()]
            ftstd_medium = [X[(y > 1/3) & (y <= 2/3)].std()]
            ftstd_high = [X[(y > 2/3) & (y <= 1)].std()]
            featStd = pd.DataFrame(data = {'zero': ftstd_zero, 'low': ftstd_low, 'medium': ftstd_medium, 'high': ftstd_high})
    
            inputAvg = inputAvg.dropna(axis=1, how='all')
            featAvg = featAvg.dropna(axis=1, how='all')
            featStd = featStd.dropna(axis=1, how='all')
    
            num_cols = min(featAvg.shape[1], inputAvg.shape[1]) # get number of columns in both dataframes
            cols = list(range(num_cols)) # get list of column indices
                    
            avgSum = [((featAvg.iloc[:, i] - inputAvg.iloc[:, i]) ** 2).sum() for i in cols]
            
            # calculate the sum of the standard deviations          
            stdevSum = featStd.sum().sum()
            
            # divide sum of averages by sum of standard deviations
            score = sum(avgSum) / stdevSum 
        
            row = pd.DataFrame([[gene_name, score, feat_of_interest]], columns=['Feature', 'FisherScore', 'TargetFeature'])
            current_df = pd.concat([current_df, row], ignore_index=True)
            current_df = current_df[current_df['FisherScore'] != 0]
            
            # sort into descending order
            current_df.sort_values(by='FisherScore', ascending=False, inplace=True)
            
            # select specified number features with greatest Fisher scores
            current_df = current_df.head(threshold)
    
        # add individual dataframes to list
        fisherScore_dfs.append(current_df)
        print(f'Fisher scores calculated for {column}. Remaining features: {cols_remaining}')           
    
    # returns a list of multiple dataframes
    return fisherScore_dfs


# ----------------- #

def create_list_of_dataframes_from_fisher_scores(all_fscores):
    """Split Fisher score dataframe into a list of dataframes.
    
    Input:
        all_fscores <pd.DataFrame>: All Fisher scores concatenated into one
            dataframe (cols: 'Feature', 'FisherScore', 'TargetFeature')
    
    Output:
        <[pd.DataFrame]>: List of dataframes, split on the TargetFeature column
    """
    
    fisher_score_dfs = [group for _, group in all_fscores.groupby('TargetFeature', sort=False)]
    return fisher_score_dfs


# ----------------- #

def create_submatrix_from_clustered_matrix(prots_to_keep, subset_name, min_vals):
    """Creates a submatrix of clustered matrix containing only specified proteins.
    
    Inputs:
        full_matrix <pd.DataFrame>: Matrix containing all features (ie. ClusteredMatrix.csv)
        prots_to_keep <tuple of strings>: Feature names containing these proteins
            are extracted into the new submatrix
        subset_name <str>: Identifier for specific matrix subset (ie. 'FEARMEN')
        min_vals <int>: Threshold value to specify minimum values per feature
        
    Outputs:
        <pd.DataFrame>: Subset of the clustered matrix containing only specified features
    """
    full_matrix = pd.read_csv(f'/data/home/bt23917/PPA/04_clustering/interim_data/ClusteredMatrix_Min{min_vals}Vals.csv', header=0, index_col=0)
    # full_matrix = pd.read_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/interim_data/ClusteredMatrix_Min{min_vals}Vals.csv', header=0, index_col=0)
    
    # set index
    set_dataset_name_as_index(full_matrix)
    
    # keep only columns containing one of the specified proteins
    submatrix = full_matrix[[col for col in full_matrix.columns if col.split('_')[0] in prots_to_keep]]

    submatrix.to_csv(f'/data/home/bt23917/PPA/04_clustering/interim_data/ClusteredMatrix_{subset_name}_Min{min_vals}Vals.csv', index=False)
    # submatrix.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/ClusteredMatrix_{subset_name}_Min{min_vals}Vals.csv', index=False)
    return submatrix

def load_clustered_matrix_and_fisher_scores(base_dir, threshold):
    # load fisher scores
    fisher_scores = pd.read_csv(f'{base_dir}/04_clustering/top_500_fisher_scores_min{threshold}vals.csv')
    # group fisher scores into separate dataframes according to their target feature
    fisher_score_dfs = create_list_of_dataframes_from_fisher_scores(fisher_scores)
    # load matrix
    matrix = pd.read_csv(f'{base_dir}/04_clustering/clustered_matrix_min{threshold}vals.csv', header=0)
    matrix = set_dataset_name_as_index(matrix)
    return matrix, fisher_score


# ----------------- #

def load_clustered_matrix_and_fisher_score_files(base_dir, threshold):
    # load fisher scores
    fisher_scores = pd.read_csv(f'{base_dir}/05_feature_selection/top_500_fisher_scores_min{threshold}vals.csv')
    # group fisher scores into separate dataframes according to their target feature
    fisher_score_dfs = create_list_of_dataframes_from_fisher_scores(fisher_scores)

    # load matrix
    matrix = pd.read_csv(f'{base_dir}/04_clustering/clustered_matrix_min{threshold}vals.csv', header=0)
    matrix = set_dataset_name_as_index(matrix)
    return matrix, fisher_score_dfs


# ----------------- #

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
    clusters_output.to_csv(f'/data/home/bt23917/PP/04_clustering/interim_data/{filename}.csv', index=True)
    # clusters_output.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/interim_data/{filename}.csv', index=True)

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
        
    output_matrix.to_csv(f'/data/home/bt23917/PPA/04_clustering/interim_data/{filename}.csv', index=False)
    # output_matrix.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/interim_data/{filename}.csv', index=False)
    
    print('Matrix with clustered phosphosites has been saved:', output_matrix.head())
    return output_matrix
    