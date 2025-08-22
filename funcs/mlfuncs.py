#!/bin/python

import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from scipy.stats import spearmanr
pd.set_option('future.no_silent_downcasting', True)


# ----------------- #

def remove_cols_not_in_both(matrix, fisher_score_dfs):
    '''
    In some instances, the clusters in fisher_score_dfs and matrix.columns don't align.
    These need identifying and removing.
    '''
    # if matrix length is greater than fisher_score_dfs length, remove columns from matrix that aren't in fisher_score_dfs
    if len(matrix.columns) > len(fisher_score_dfs):
        print(f'Matrix (length: {len(matrix.columns)}) has {len(matrix.columns) - len(fisher_score_dfs)} more clusters than fisher_score_dfs (length: {len(fisher_score_dfs)}). Removing additional clusters from matrix...')
        
        lst = []
        # for each dataframe in the fisher_score_dfs list
        for i in fisher_score_dfs:
            # add the target feature to a list
            lst.append(i['TargetFeature'].drop_duplicates().values[0])
        
        # for each cluster column in the matrix
        for col in matrix.columns:
            # if the cluster column is not in lst (ie. the column from the matrix isn't in fisher_score_dfs)
            if col not in lst:
                # remove the column from the matrix
                matrix = matrix[matrix.columns.drop([col])]
                # print(f'The following clusters have been removed: {col}')
        print(f'Additional clusters from the matrix list have been removed. There are now {len(matrix.columns)} clusters in the matrix.')
        
    # if fisher_score_dfs length is greater than matrix length, remove columns from fisher_score_df that aren't in matrix
    elif len(matrix.columns) < len(fisher_score_dfs):
        print(f'Matrix (length: {len(matrix.columns)}) has {len(fisher_score_dfs) - len(matrix.columns)} fewer clusters than fisher_score_dfs (length: {len(fisher_score_dfs)}). Removing additional clusters from fisher_score_dfs...')
        
        df_lst = []
        for i in matrix.columns:
            df_lst.append(i)

        # list comprehension to remove dataframes from fisher_score_dfs list if cluster is not in df_list (ie. the cluster in fisher_score_dfs isn't in the matrix)
        fisher_score_dfs = [clust for clust in fisher_score_dfs if clust['TargetFeature'].iloc[0] not in df_lst]
        print(f'Additional clusters from fisher_score_dfs list have been removed. There are now {len(fisher_score_dfs)} clusters in fisher_score_dfs.')
                
    # if both have the same length, continue
    else:
        print(f'Matrix and fisher_score_dfs have the same number of clusters ({len(matrix.columns)} and {len(fisher_score_dfs)} respectively).')

    return matrix, fisher_score_dfs
    
    

# ----------------- #

# def select_specific_cluster(fisher_score_dfs, selection):
#     """Sets a specific cluster to be used in the script."""
#     target_id = str(selection)
#     current_cluster = None
#     for df in fisher_score_dfs:
#         if target_id in df['TargetFeature'].values:
#             current_cluster = df
#             break
#     return current_cluster


# ----------------- #

def choose_all_or_one_cluster(dfs, matrix_col_index):
    """Sets the cluster to be used in the script. Run 'all' for an array job.
    
    Args:
        selection: str ('all' or 'cluster name')
    """
    if isinstance(matrix_col_index, int):
        # select the cluster of interest from the dfs list
        current_cluster = dfs[matrix_col_index]
    else:
        target_id = str(matrix_col_index)
        current_cluster = None
        for df in dfs:
            if target_id in df['TargetFeature'].values:
                current_cluster = df
                break
    return current_cluster

# ----------------- #

def create_X_and_y_dfs(df, target_feature):
    # shuffle dataframe
    df = df.sample(frac=1, random_state=42)
    # X is all columns except the target feature
    X = df.drop([f'TargetFeature: {target_feature}'], axis=1)
    # y is only the target feature column
    y = df[f'TargetFeature: {target_feature}'].values
    return X, y


# ----------------- #

def format_predictive_feats_dfs(targetfeature_df, matrix):
    """Create df containing only relevant predictive features and target feature
    
        Columns 0:-1 <= predictive features
        Column -1 <= target feature
    
    Each output dataframe will be formatted as follows:
    feat1 | feat2 | feat3 | feat4 | feat5 | ... | TargetFeature: {tf}
     xxx  |  xxx  |  xxx  |  xxx  |  xxx  | ... |        xxx  
     xxx  |  xxx  |  xxx  |  xxx  |  xxx  | ... |        xxx  
     xxx  |  xxx  |  xxx  |  xxx  |  xxx  | ... |        xxx
    """
    # formatting for saving files
    # store target feature
    target_feature = targetfeature_df['TargetFeature'].iloc[0] 
    # set length limit for feature name
    feature_max_length = 20 
    # crop target feature to max length for saving to files
    truncated_target_feature = target_feature[:feature_max_length] 
    
    '''Identify how many features can be used for prediction. To use x features for prediction, 
    we need a minimum of x*10 data values for the target feature.'''
    
    # store target feature column
    target_col = matrix.loc[:, target_feature] 
    # count non-NA values in target feature column
    target_count = target_col.count() 
    # divide total features 10 to identify how many can be used for prediction
    num_features = int(target_count/10) 
    # select computed number of features from current cluster to be used for prediction
    cropped_fscores = targetfeature_df[0:num_features] # fisher scores are already ordered in descending order

    '''Format dataframe.'''
    
    # print(cropped_fscores)
    # create df with columns as predictive features
    new_df = pd.DataFrame(columns = cropped_fscores['PredictiveFeature'])
    ## feat1 | feat2 | feat3 | feat4 | feat5 | ...
    new_df[f'TargetFeature: {target_feature}'] = matrix.loc[:, target_feature] # add target feature column
            
    # add phosphoproteomics data from input matrix to new_df if column names match
    for col in new_df.columns: # loop over column names of new_df
        if col in matrix.columns: # loop over column names of input matrix
            new_df[col] = matrix[col] # add columns from input matrix to new dataframe
    new_df = new_df.fillna(0)

    # remove target feature from features if necessary
    if target_feature in new_df.columns:
        new_df = new_df.drop(f'{target_feature}', axis=1)
        print(f'{target_feature} was both the TargetFeature and in features columns. It has been removed from features columns.')
    else:
        print(f'{target_feature} is only the TargetFeature column.')
    
    return new_df, target_feature, truncated_target_feature, cropped_fscores


# ----------------- #

def compute_spearman_corr(best_model, X, y):
    kf = KFold(n_splits=10)
    y_pred = cross_val_predict(best_model, X, y, cv=kf)
    mask = y != 0
    y_no_impute = y[mask]
    y_pred_no_impute = y_pred[mask]
    spearman_corr, _ = spearmanr(y_no_impute, y_pred_no_impute)
    return spearman_corr


# ----------------- #

def concat_best_models_all_clusters(loss_function, min_vals, model_type):
    """Concatenates the best models into a single file.
    
    Iterates through files in a directory, selecting the optimal model 
    parameters from each, and saves these to a new file. The output 
    CSV contains one row per feature, with optimal model parameters 
    stored in columns.
    
    Input:
        loss_function <str>: One of 'mse', 'mae', or 'r2'
        min_vals <int>: Threshold value to specify minimum values per feature
        model_type <str>: One of 'cnn' or 'xgboost'
        
    Output:
        CSV file <csv>: Concatenated file of best models and parameters
    """
    if model_type == 'cnn':
        df = pd.DataFrame(columns=['TargetFeature', 'LR', 'epochs', 'num_filters', 
                               'num_layers', 'mean_mse', 'mean_mae', 'mean_r2', 
                               'spearman_corr_mse', 'spearman_corr_mae', 'spearman_corr_r2'])
    elif model_type == 'xgboost':
        df = pd.DataFrame(columns=['TargetFeature', 'colsample_bytree', 'gamma', 'max_depth', 
                               'min_child_weight', 'n_estimators', 'subsample', 'mean_mse', 
                               'mean_mae', 'mean_r2', 'spearman_corr_mse', 'spearman_corr_mae',
                               'spearman_corr_r2'])
    
    for filename in os.listdir(f"/data/Blizard-ZabetLab/CM/{model_type}/results_files"):
        if f"Min{min_vals}Vals" in filename:
            with open(f'/data/Blizard-ZabetLab/CM/{model_type}/results_files/{filename}', 'r') as FILE:
                current_file = pd.read_csv(FILE)
                if loss_function == 'mse' or loss_function == 'mae':
                    idx = current_file[f'mean_{loss_function}'].idxmin() # get index of row with greatest value
                elif loss_function =='r2' or loss_function == 'spcorr':
                    idx = current_file[f'mean_{loss_function}'].idxmax() # get index of row with lowest value
                current_row = pd.DataFrame(current_file.loc[idx]).T
                df = pd.concat([df, current_row], ignore_index=True)

    df.to_csv(f'/data/home/bty449/ExplainableAI/06_models/{model_type}/params/{model_type}_optimised_parameters_{loss_function}_min{min_vals}vals.csv', index=False)
    return df


# ----------------- #

def identify_models_for_retraining(base_dir, threshold, model_type):
    """Identifies models that need retraining.
    
    Two step function:
    1. Checks if clusters in fisher_score_dfs and matrix.columns align. 
        If they don't, clusters not in both are removed.
    2. Compares remaining clusters with those in optimised_parameters file. 
        Clusters missing from the optimised parameters file need training 
        again with the specified model and threshold.
   
    Input:
        model_type <str>: One of 'cnn' or 'xgboost'
        min_vals <int>: Threshold value to specify minimum values per feature
    """
    
    # load required files
    fscores = pd.read_csv(f'{base_dir}/05_feature_selection/interim_data/top_500_fisher_scores_min{threshold}vals.csv', header=0)
    fisher_score_dfs = [group for _, group in fscores.groupby('TargetFeature', sort=False)]

    matrix = pd.read_csv(f'{base_dir}/04_clustering/interim_data/clustered_matrix_min{threshold}vals.csv', header=0)

    optimised_models = pd.read_csv(f'{base_dir}/06_models/{model_type}/params/{model_type}_nested_cv_master_results_file_min{threshold}vals.csv', header=0) # mse, mae and r2 files have the same clusters
    
    # ~~~~~~~~~~~~ 1) check for correct rows
    lst = []
    for i in fisher_score_dfs: # for each dataframe in the fisher_score_dfs list
        lst.append(i['TargetFeature'].drop_duplicates().values[0]) # add the target feature to a list
        
    non_existent_cols = []    
    for col in matrix.columns:
        if col not in lst:
            non_existent_cols.append(col)
            
    for clust in lst:
        if clust not in matrix.columns:
            non_existent_cols.append(col)

    # ~~~~~~~~~~~~ 2) identify clusters for retraining
    matrix_clusters = set(matrix.columns)
    optimised_clusters = set(optimised_models['TargetFeature']) # all mse, mae, spcorr and r2 files have the same clusters

    missing_clusters = matrix_clusters - optimised_clusters # find difference between sets

    if len(missing_clusters) == 0 or (len(missing_clusters) == 1 and 'DatasetName' in missing_clusters):
        print('All models have been trained.')
    else:
        print(f'Clusters to be retrained for {model_type} at the {threshold} threshold:')
        for cluster in missing_clusters:
            if cluster != 'DatasetName' and cluster not in non_existent_cols:
                print(f'{cluster}\n') # print missing clusters
       

# ----------------- #     
            
def compute_global_shap_values_from_local_values(local_values_df, cropped_fisher_scores):
    """Compute a global SHAP value per feature by averaging local values.
    
    SHAP values are initially computed per dataset (d) for each predictive 
    feature (p), meaning there are d * p local SHAP values. This function
    computes global values by taking the mean of local values for each
    feature.
    
    Input:
        local_values_df <pd.DataFrame>: Local SHAP values (one value for every
            row in every column)
        cropped_fisher_scores <pd.DataFrame>: Top X features with greatest Fisher
            scores from current cluster (X is 10x smaller than the number of 
            datasets with quantification values for that feature)
    
    Output:
        <pd.DataFrame>: Global SHAP values (one value per feature)
    """
    
    global_values_df = local_values_df.abs().mean().to_frame().T
    
    # rename columns to match the feature names
    global_values_df.columns = cropped_fisher_scores['PredictiveFeature'].tolist()
    return global_values_df


# ----------------- #  

def perform_grid_search(instantised_model, parameters_to_search, X, y, scoring):
    """Performs grid search to find the best hyperparameters for the model.
    
    Args:
        instantised_model: KerasRegressor object
        parameters_to_search: dict
        X: np.array
        y: np.array
        scoring: str
    """
    grid_search = GridSearchCV(estimator=instantised_model, param_grid=parameters_to_search, cv=10, scoring=scoring)
    grid_search_result = grid_search.fit(X, y, verbose=0)
    best_model = grid_search_result.best_estimator_
    best_params = grid_search_result.best_params_
    if scoring == 'neg_mean_squared_error' or scoring == 'neg_mean_absolute_error':
        values = -grid_search_result.cv_results_['mean_test_score']
    elif scoring == 'r2':
        values = grid_search_result.cv_results_['mean_test_score']
    mean_value = np.mean(values)
    return grid_search_result, best_model, best_params, mean_value


# ----------------- #

def concat_and_log_all_shap_files(threshold, model_type):
    """Concatenates all SHAP files and computes logged values.
    
    Iterates through global SHAP files in a directory, and concatenates
    into a single file. Then computes log of SHAP values and add this column
    to the dataframe.
    
    Input:
        min_vals <int>: Threshold value to specify minimum values per feature
        model_type <str>: One of 'cnn' or 'xgboost'
        
    Output:
        CSV files <csv>: Concatenated file of SHAP values for each feature with 
            respect to each target feature (cols = 'PredictiveFeature', 
            'TargetFeature', 'SHAPvalue', 'LogSHAPValue')
    """
    df = pd.DataFrame(columns=['PredictiveFeature', 'TargetFeature', 'SHAPValue'])
    
    for filename in os.listdir(f"/data/Blizard-ZabetLab/CM/{model_type}/nested_cv_global_shaps"):
        # specify only certain files
        if f"min{threshold}vals" in filename and f"_global" in filename:
            truncated_feat = filename.split("_global")[0]
            
            with open(f'/data/Blizard-ZabetLab/CM/{model_type}/nested_cv_global_shaps/{filename}', 'r') as FILE:
                current_file = pd.read_csv(FILE)
                transformed_file = current_file.T.reset_index()
                transformed_file.columns = ['PredictiveFeature', 'SHAPValue']
                # create new column in file to store the target feature
                transformed_file['TargetFeature'] = truncated_feat
                transformed_file = transformed_file[['PredictiveFeature', 'TargetFeature', 'SHAPValue']]
                # add current file contents to growing df
                df = pd.concat([df, transformed_file], ignore_index=True)

    
    
    # logged SHAP values column
    df['LogSHAPValue'] = np.log(df['SHAPValue'].replace(0, np.nan))

    # many of the SHAP values are 0, so remove these
    df = df[df['SHAPValue'] != 0] # remove rows with SHAP value of 0

    df.to_csv(f"/data/home/bty449/ExplainableAI/08_results/{model_type}/nested_cv_master_shaps/{model_type}_master_shap_file_cluster_level_min{threshold}vals.csv", index=False)
    
    # reduce clusters to protein level
    df['PredictiveFeature'] = [i.split("_")[0] for i in df['PredictiveFeature']]
    df['TargetFeature'] = [j.split("_")[0] for j in df['TargetFeature']]

    df['Coeff*R2'] = df['LogSHAPValue'] * df['MeanR2']


    df.to_csv(f"/data/home/bty449/ExplainableAI/08_results/{model_type}/nested_cv_master_shaps/{model_type}_master_shap_file_protein_level_min{threshold}vals.csv", index=False)

    return df


# ----------------- #

def extract_specified_shap_values_and_return_log_shap_array(min_vals, model_type, network_name, prots):
    """Concatenates specified SHAP files and computes logged values.
    
    Iterates through global SHAP files in a directory, and concatenates
    specified clusters/proteins into a single file. Then computes log of 
    SHAP values and add this column to the dataframe.
    
    Input:
        min_vals <int>: Threshold value to specify minimum values per feature
        model_type <str>: One of 'cnn' or 'xgboost'
        network_name <str>: Name of network
        prots <tuple>: Tuple of protein names
        
    Output:
        CSV files <csv>: Concatenated file of SHAP values for each feature with 
            respect to each target feature (cols = 'PredictiveFeature', 
            'TargetFeature', 'SHAPvalue', 'LogSHAPValue')
    """
    df = pd.read_csv(f"/data/Blizard-ZabetLab/CM/{model_type}/concatenated_shaps/{model_type}_concatenated_all_shap_values_cluster_level_min{min_vals}vals.csv", header=0)
    print("Original df:")
    print(df.head())
    
    df_subset = df[df.apply(lambda row: any(prot in row['PredictiveFeature'] for prot in prots) or any(prot in row['TargetFeature'] for prot in prots), axis=1)]
    print("Subset df:")
    print(df_subset.head())
    
    df_subset.to_csv(f"/data/Blizard-ZabetLab/CM/{model_type}/concatenated_shaps/{model_type}_concatenated_{network_name}_shap_values_cluster_level_min{min_vals}vals.csv", index=False)
    df_subset.to_csv(f"/data/home/bty449/ExplainableAI/07_results/{model_type}/concatenated_shaps/{model_type}_concatenated_{network_name}_shap_values_cluster_level_min{min_vals}vals.csv", index=False)
    
    # reduce clusters to protein level
    df_subset.loc[:, 'PredictiveFeature'] = df_subset['PredictiveFeature'].apply(lambda x: x.split("_")[0])
    df_subset.loc[:, 'TargetFeature'] = df_subset['TargetFeature'].apply(lambda x: x.split("_")[0])
    print("Protein df:")
    print(df_subset.head())

    df_subset.to_csv(f"/data/Blizard-ZabetLab/CM/{model_type}/concatenated_shaps/{model_type}_concatenated_{network_name}_shap_values_protein_level_min{min_vals}vals.csv", index=False)
    df_subset.to_csv(f"/data/home/bty449/ExplainableAI/07_results/{model_type}/concatenated_shaps/{model_type}_concatenated_{network_name}_shap_values_protein_level_min{min_vals}vals.csv", index=False)
    
    log_shap_vals = df_subset['LogSHAPValue'].values
    log_shap_vals = log_shap_vals[~np.isnan(log_shap_vals)]
    print(log_shap_vals)
    
    if np.isnan(log_shap_vals).any():
        print("log_shap_vals contains NaNs")
    else:
        print("log_shap_vals does not contain NaNs")
    
    return log_shap_vals


# ----------------- #
