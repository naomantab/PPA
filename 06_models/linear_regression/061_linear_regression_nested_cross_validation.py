#!/bin/python

import time
start_time = time.time()
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import os
import sys
import argparse

grandgrandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(grandgrandparent_dir)
from funcs import generalfuncs, mlfuncs
print('Dependencies loaded.')


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    parser.add_argument('--base_dir', type=str, default='/data/home/bt23917/PPA/06_models', help='Base directory for the project')
    parser.add_argument('--threshold', type=int, default=200, help='Threshold to compute')

    return parser.parse_args()

def linear_regression_cv(X, y, model, clust_matrix, i):
    """Perform linear regression."""

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    coefficients_dfs = []
    r2_scores = []
    mse_scores = []

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

        coef_df = pd.DataFrame({'PredictiveFeature': X.columns, 
                                'Coefficient': model.coef_,
                                'TargetFeature': clust_matrix.columns[i]})
        
        coefficients_dfs.append(coef_df)
    
    all_coeffs = pd.concat(coefficients_dfs, ignore_index=True)
    all_coeffs = all_coeffs.groupby('PredictiveFeature')['Coefficient'].median().reset_index()
    
    return all_coeffs, r2_scores, mse_scores

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

    corr_list = []
    all_coefficients_list = []

    try:
        # csv = pd.read_csv(f'{args.base_dir}/08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min{args.threshold}vals.csv', header=0)
        csv = pd.read_csv(f'{args.base_dir}/results/linear_regression_cv_coefficients_min{args.threshold}vals.csv', header=0)
    except:
        csv = pd.DataFrame(columns=['PredictiveFeature', 'TargetFeature', 'FisherScore', 'MedianCoeff', 'MeanR2', 'MeanMSE'])

    count = len(clust_matrix.columns)

    for i in range(len(clust_matrix.columns)):
        print('Target feature:', clust_matrix.columns[i])

        current_cluster = fisher_score_dfs[i]

        if clust_matrix.columns[i] != current_cluster['TargetFeature'].values[0]:
            print(f"Target feature {clust_matrix.columns[i]} does not match current cluster {current_cluster['TargetFeature'].values[0]}.")
            break

        print("Formatting dataframes of predictive and target features...")
        new_df, target_feature, truncated_target_feature, cropped_fscores = mlfuncs.format_predictive_feats_dfs(
            targetfeature_df=current_cluster, 
            matrix=clust_matrix
        )

        X, y = mlfuncs.create_X_and_y_dfs(
            df=new_df,
            target_feature=target_feature
        )

        model = LinearRegression()
        all_coeffs, r2_scores, mse_scores = linear_regression_cv(X, y, model, clust_matrix, i)

        for col, row in all_coeffs.iterrows():
            pred_feat = row.iloc[0]
            coeff = row.iloc[1]

            # fisher score for the target feature
            fisher_score = current_cluster[current_cluster['PredictiveFeature'] == pred_feat]['FisherScore'].values[0]

            new_row_dict = {
                'PredictiveFeature': pred_feat,
                'TargetFeature': clust_matrix.columns[i],
                'FisherScore': fisher_score,
                'MedianCoeff': coeff,
                'MeanR2': np.mean(r2_scores),
                'MeanMSE': np.mean(mse_scores)
            }

            if ((csv['TargetFeature'] == clust_matrix.columns[i]) & (csv['PredictiveFeature'] == pred_feat)).any():
                csv = csv.drop(csv[(csv['TargetFeature'] == clust_matrix.columns[i]) & (csv['PredictiveFeature'] == pred_feat)].index)
                print(f'{clust_matrix.columns[i]} and {pred_feat} were already in the csv file. The old entry has been removed.')

            new_row_df = pd.DataFrame([new_row_dict])
            csv = pd.concat([csv, new_row_df], ignore_index=True)

        # drop rows with NaN in 'MedianCoeff' before applying z-score
        csv = csv.dropna(subset=['MedianCoeff'])

        # remove rows where 'MedianCoeff' has a z-score < 3 across all values in the column
        csv = csv[(np.abs(zscore(csv['MedianCoeff'])) < 3)]

        count -= 1
        print(f'{count} clusters left to process.')

    csv['Coeff*R2'] = csv['MedianCoeff'] * csv['MeanR2']
        
    # csv.to_csv(f'{args.base_dir}/08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min{args.threshold}vals.csv', index=False)
    csv.to_csv(f'{args.base_dir}/results/linear_regression_cv_coefficients_min{args.threshold}vals.csv', index=False)

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')