#!/bin/python

import time
start_time = time.time()
import pandas as pd
import numpy as np
import sys
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import xgboost
from xgboost import XGBRegressor
print(xgboost.__version__)
import shap

grandgrandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(grandgrandparent_dir)
from funcs import generalfuncs, mlfuncs
print('Dependencies loaded.')


# ----------------- #

def int_or_str(value):
    try:
        # Try to convert to an integer
        return int(value)
    except ValueError:
        # If it fails, return the value as a string
        return value

def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/PPA/xgboost', help='Base directory for the project')
    parser.add_argument('--threshold', type=int, default=50, help='Threshold to compute')
    parser.add_argument('--matrix_col_index', type=int_or_str, help='Column in clustered matrix to perform XGBoost on.')

    return parser.parse_args()

def perform_nested_cross_validation(X, y, model, param_grid, cropped_fscores):
    """Performs nested cross-validation for XGBoost model.
    
    For each of the 5 outer folds:
    - Hyperparameters are tuned using 10-fold cross-validation on train data
    - SHAP values are computed using 10-fold cross-validation on test data
    - Median SHAP value is computed across the 10 folds for each outer fold
    - The median SHAP values across the 5 outer folds is returned
    """
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # split data fives times for outer cross-validation
    global_shap_df = []
    r2_scores = []

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
        grid_search_result = grid_search.fit(X_train, y_train)
        best_model = grid_search_result.best_estimator_
        best_params = grid_search_result.best_params_

        neg_mse = -grid_search_result.cv_results_['mean_test_score']
        mean_mse = np.mean(neg_mse)

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        
        def model_predict(X):
            return best_model.predict(X) 
        explainer = shap.KernelExplainer(model_predict, X_train)

        shap_inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        shapley_arrays = []

        for inner_fold_idx, (shap_train_idx, shap_test_idx) in enumerate(shap_inner_cv.split(X_test)):
            X_shap_test = X_test.iloc[shap_test_idx]
            # one shap value per feature per each of the 10 folds
            shap_values = explainer.shap_values(X_shap_test)
            # print(shap_values)
            shapley_arrays.append(shap_values)
        
        concatenated_array = np.concatenate(shapley_arrays, axis=0)
        shaps_df = pd.DataFrame(concatenated_array, columns=cropped_fscores['Feature'].tolist())
        local_global_values_df = shaps_df.abs().median().to_frame().T
        local_global_values_df.columns = cropped_fscores['Feature'].tolist()

        global_shap_df.append(local_global_values_df)

    global_shap_df = pd.concat(global_shap_df, ignore_index=True)
    global_shap_df = global_shap_df.abs().median().to_frame().T

    return grid_search_result, best_params, mean_mse, global_shap_df, r2_scores
    

# ----------------- #

if __name__ == '__main__':

    args = parse_arguments()
    
    print("Loading clustered matrix and fisher scores files...")
    clust_matrix, fisher_score_dfs = generalfuncs.load_clustered_matrix_and_fisher_score_files(
        base_dir=args.base_dir,
        threshold=args.threshold
    )

    print("Removing columns not in both clustered matrix and fisher scores...")
    clust_matrix, fisher_score_dfs = mlfuncs.remove_cols_not_in_both(clust_matrix, fisher_score_dfs)

    print("Selecting cluster to be target feature for prediction...")
    current_cluster = mlfuncs.choose_all_or_one_cluster(
        dfs=fisher_score_dfs,
        matrix_col_index=args.matrix_col_index
    )

    print("Formatting dataframes of predictive and target features...")
    new_df, target_feature, truncated_target_feature, cropped_fscores = mlfuncs.format_predictive_feats_dfs(
        targetfeature_df=current_cluster, 
        matrix=clust_matrix
    )
    
    X, y = mlfuncs.create_X_and_y_dfs(
         df=new_df,
         target_feature=target_feature
    )

    # parameter search grid
    param_grid = {
        'n_estimators': [2, 4, 6, 8],
        'max_depth': [1, 2, 3, 4],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],}
    

    print("Performing nested cross-validation...")
    model = XGBRegressor()
    grid_search_result, best_params, mean_score, global_shap_df, r2_scores = perform_nested_cross_validation(
        X, y, model, param_grid, cropped_fscores
    )
    shap_file = f"{truncated_target_feature}_global_shaps_min{args.threshold}vals.csv"
    global_shap_df.to_csv(f"/data/home/bt23917/PPA/xgboost/nested_cv_global_shaps/{shap_file}", index=False)
    print(global_shap_df)
    print("R² scores for each outer fold:", r2_scores)
    print("Mean R2 score across outer folds:", np.mean(r2_scores))

    results_file = f"{truncated_target_feature}_grid_search_results_xgboost_min{args.threshold}vals.csv"
    print(f"Saving results to {results_file}...")
    results_df = pd.DataFrame(columns=['TargetFeature'] + list(grid_search_result.cv_results_['params'][0].keys()) + ['mean_mse'] + ['mean_r2'])
    results_df.loc[len(results_df)] = [target_feature] + list(best_params.values()) + [mean_score] + [np.mean(r2_scores)]
    results_df.to_csv(f'/data/home/bt23917/PPA/xgboost/nested_cv_results_files/{results_file}', index=False)
    print(results_df)

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')