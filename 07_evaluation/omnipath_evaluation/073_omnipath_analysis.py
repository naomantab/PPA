#!/bin/python

# Run from command line:
# python3 omnipath_evaluation.py --base_dir /path/to/project --network_name my_network --model_types linear_regression xgboost --thresholds 50 100
# or just
# python3 omnipath_evaluation.py 
# for default parameters

import time
start_time = time.time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import os
import argparse


# ----------------- #

def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against OmniPath reference.')
    parser.add_argument('--base_dir', type=str, default='C:/Users/tnaom/OneDrive/Desktop/PPA', help='Base directory for the project')
    parser.add_argument('--network_name', type=str, default=None, help='Network name if analyzing specific network')
    parser.add_argument('--model_types', nargs='+', default=['linear_regression', 'xgboost'], 
                        help='Model types to evaluate')
    parser.add_argument('--thresholds', nargs='+', default=[50, 100, 150, 200], 
                        help='Thresholds to evaluate')
    return parser.parse_args()

def get_file_path(base_dir, network_name, model_type, threshold, file_type, int_type):
    """Construct file paths based on parameters."""
    if file_type == 'input_preds':
        if model_type == 'linear_regression':
            if network_name is None:
                return f'{base_dir}/08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min{threshold}Vals.csv'
            else:
                return f'{base_dir}/08_results/linear_regression/coefficients/linear_regression_{network_name}_coefficients_protein_level_min{threshold}vals.csv'
        else:
            if network_name is None:
                return f'{base_dir}/08_results/{model_type}/concatenated_shaps/{model_type}_master_shap_file_protein_level_min{threshold}vals.csv'
            # else:
            #     return f'{base_dir}/08_results/{model_type}/concatenated_shaps/{model_type}_concatenated_{network_name}_shap_values_protein_level_min{threshold}vals.csv'
    elif file_type == 'csv_output':
        filename = f'omnipath_aucs_{int_type}'
        if network_name is not None:
            filename = f'{filename}_{network_name}'
        return f'{base_dir}/08_results/omnipath_results/{filename}.csv'
    else:
        # For plot outputs
        filename = f'omnipath_aucs_{int_type}'
        if network_name is not None:
            filename = f'{filename}_{network_name}'
        return f'{base_dir}/08_results/omnipath_results/{filename}.png'

def load_reference_data(base_dir):
    """Load and format the OmniPath reference dataset."""
    df = pd.read_csv(f"{base_dir}/07_evaluation/omnipath_evaluation/omnipath_mouse_human_homologue_interactions.csv")
    df = df[['mouse_source_gene_name', 'mouse_target_gene_name']]
    df.columns = ['PredictiveFeature', 'TargetFeature']
    print('Reference dataframe:', df)
    return df

def load_prediction_data(base_dir, network_name, model_type, threshold):
    """Load and format the predicted interactions data."""
    file_path = get_file_path(base_dir=base_dir, 
                              network_name=network_name, 
                              model_type=model_type, 
                              threshold=threshold,
                              file_type='input_preds', 
                              int_type=None)
    df = pd.read_csv(file_path, header=0)
    
    if model_type == 'linear_regression':
        df['TargetFeature'] = df['TargetFeature'].str.split('_').str[0]
        df['PredictiveFeature'] = df['Feature'].str.split('_').str[0]
        # df = df[['PredictiveFeature', 'TargetFeature', 'Coefficient']]
        df = df[['PredictiveFeature', 'TargetFeature', 'Coeff*R2']]
        # value_col = 'Coefficient'
        value_col = 'Coeff*R2'
    else:
        df = df[['PredictiveFeature', 'TargetFeature', 'SHAPValue']]
        value_col = 'SHAPValue'
    
    # Remove duplicates
    df.dropna(subset=[value_col], inplace=True)
    df.drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'], inplace=True)
    print('Predicted dataframe:', df)
    return df, value_col

def filter_for_common_proteins(predictions, reference):
    """Retain only proteins present in both datasets."""
    # Standardize protein names
    for df in [predictions, reference]:
        df['PredictiveFeature'] = df['PredictiveFeature'].str.upper().str.strip()
        df['TargetFeature'] = df['TargetFeature'].str.upper().str.strip()
    
    # Get common proteins
    pred_prots = set(predictions['PredictiveFeature']) | set(predictions['TargetFeature'])
    ref_prots = set(reference['PredictiveFeature']) | set(reference['TargetFeature'])
    common_prots = pred_prots & ref_prots
    print(f'Number of common proteins: {len(common_prots)}')
    
    # Filter both datasets
    mask_preds = (predictions['PredictiveFeature'].isin(common_prots)) & (predictions['TargetFeature'].isin(common_prots))
    mask_ref = (reference['PredictiveFeature'].isin(common_prots)) & (reference['TargetFeature'].isin(common_prots))
    
    return predictions[mask_preds].copy(), reference[mask_ref].copy()

def format_df_per_model_and_threshold(base_dir, network_name, model_types, thresholds):
    """Main function to run the evaluation process."""
    
    # Ensure output directory exists
    os.makedirs(f'{base_dir}/08_results/omnipath_results', exist_ok=True)
    
    print("Loading reference data...")
    reference_data = load_reference_data(base_dir)
    
    # Load and process predictions for all models and thresholds
    print("Loading and processing prediction data...")
    predictions_dict = {model: {} for model in model_types}
    value_cols = {model: None for model in model_types}
    reference_filtered_dict = {model: {} for model in model_types}

    print('Creating dictionaries of predictions and reference dataframes...')
    for model in model_types:
        for threshold in thresholds:
            # Load predictions
            predictions, value_col = load_prediction_data(base_dir, network_name, model, threshold)
            value_cols[model] = value_col
            
            # Filter for common proteins
            filtered_preds, filtered_ref = filter_for_common_proteins(predictions, reference_data)
            predictions_dict[model][threshold] = filtered_preds
            reference_filtered_dict[model][threshold] = filtered_ref

    return predictions_dict, reference_filtered_dict, value_cols

def calculate_aucs(preds_dict, ref_dict, value_cols, base_dir, network_name, model_types, thresholds, int_type):
    """Calculate PR-AUC and ROC-AUC scores."""
    auc_results = []
    for model in model_types:
        model_results = []
        for threshold in thresholds:
            preds_dict[model][threshold] = preds_dict[model][threshold].drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'])
            ref_dict[model][threshold] = ref_dict[model][threshold].drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'])

            y_scores = preds_dict[model][threshold][value_cols[model]].tolist()

            merged_df = pd.merge(preds_dict[model][threshold], ref_dict[model][threshold], on=['PredictiveFeature', 'TargetFeature'],
                                how='left', indicator=True)
            print("the merged df is: ", merged_df)
            y_true = (merged_df['_merge'] == 'both').astype(int).tolist()

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(y_true, y_scores)
            print(f"Positive vs negative examples ({model}, {threshold}): {sum(y_true)} vs {len(y_true) - sum(y_true)}")
            
            unique_labels = set(y_true)
            pos_count = sum(y_true)
            neg_count = len(y_true) - pos_count

             # Check if y_true contains both classes
            if len(unique_labels) < 2:
                print(f"Skipping threshold {threshold} for model {model} because y_true has only one class: {unique_labels}")
                pr_auc = None
                roc_auc = None
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                pr_auc = auc(recall, precision)
                roc_auc = roc_auc_score(y_true, y_scores)

            model_results.append({
                'model_type': model,
                'threshold': threshold,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'positive_examples': sum(y_true),
                'negative_examples': len(y_true) - sum(y_true)
            })
        auc_results.extend(model_results)

    auc_df = pd.DataFrame(auc_results)
    output_result_path = get_file_path(
        base_dir=base_dir,
        network_name=network_name, 
        model_type=None, 
        threshold=None, 
        file_type='csv_output',
        int_type=int_type
    )
    auc_df.to_csv(output_result_path, index=False)
    print("AUC results:", auc_df)

    return auc_df

def plot_aucs(auc_df, base_dir, network_name, int_type, title_suffix):
    """Plot PR-AUC and ROC-AUC scores."""
    thresholds = auc_df['threshold'].unique()
    model_types = auc_df['model_type'].unique()
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Bar settings
    bar_width = 0.2
    bar_positions = np.arange(len(model_types))
    colors = {
        50: '#f9993c',
        100: '#b95461',
        150: '#7985a5',
        200: '#2e86de'
    }
    
    # Plot PR-AUC
    for i, threshold in enumerate(thresholds):
        subset_df = auc_df[auc_df['threshold'] == threshold]
        for j, model in enumerate(model_types):
            model_df = subset_df[subset_df['model_type'] == model]
            axs[0].bar(bar_positions[j] + i * bar_width, model_df['pr_auc'], 
                      bar_width, label=threshold if j == 0 else "", color=colors[threshold])
    
    # Format PR-AUC plot
    axs[0].set_xlabel('Model Type', fontsize=14)
    axs[0].set_ylabel('PR AUC Scores', fontsize=14)
    network_text = f" ({network_name} proteins)" if network_name else ""
    axs[0].set_title(f'PR AUC scores{network_text} ({title_suffix})', fontsize=16)
    axs[0].set_xticks(bar_positions + bar_width * (len(thresholds) - 1) / 2)
    axs[0].set_xticklabels(model_types, fontsize=12)
    axs[0].legend(title='Min Vals', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.3)
    
    # Plot ROC-AUC
    for i, threshold in enumerate(thresholds):
        subset_df = auc_df[auc_df['threshold'] == threshold]
        for j, model in enumerate(model_types):
            model_df = subset_df[subset_df['model_type'] == model]
            axs[1].bar(bar_positions[j] + i * bar_width, model_df['roc_auc'], 
                      bar_width, label=threshold if j == 0 else "", color=colors[threshold])
    
    # Format ROC-AUC plot
    axs[1].set_xlabel('Model Type', fontsize=14)
    axs[1].set_ylabel('ROC AUC Scores', fontsize=14)
    axs[1].set_title(f'ROC AUC scores{network_text} ({title_suffix})', fontsize=16)
    axs[1].set_xticks(bar_positions + bar_width * (len(thresholds) - 1) / 2)
    axs[1].set_xticklabels(model_types, fontsize=12)
    axs[1].legend(title='Min Vals', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.3)
    
    # Match y-axis limits
    max_y = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim(0, max_y)
    axs[1].set_ylim(0, max_y)
    
    plt.tight_layout()
    output_path = get_file_path(base_dir=base_dir, 
                                network_name=network_name, 
                                model_type=None, 
                                threshold=None,
                                file_type=None, 
                                int_type=int_type)
    plt.savefig(output_path)
    plt.close()

def filter_by_conservation(predictions_dict, filter_type, model_types, thresholds):
    """Filter interactions based on conservation across models or thresholds."""
    filtered_dict = {model: {} for model in model_types}
    
    for model in model_types:
        for threshold in thresholds:
            predictions = predictions_dict[model][threshold]
            
            if filter_type == 'model':
                # Keep interactions found in at least one other model
                other_models = [m for m in model_types if m != model]
                combined_other_ints = set()
                
                for other_model in other_models:
                    other_preds = predictions_dict[other_model][threshold]
                    other_ints = set(zip(other_preds['PredictiveFeature'], other_preds['TargetFeature']))
                    combined_other_ints.update(other_ints)
                
            elif filter_type == 'threshold':
                # Keep interactions found in at least one other threshold
                other_thresholds = [t for t in thresholds if t != threshold]
                combined_other_ints = set()
                
                for other_threshold in other_thresholds:
                    other_preds = predictions_dict[model][other_threshold]
                    other_ints = set(zip(other_preds['PredictiveFeature'], other_preds['TargetFeature']))
                    combined_other_ints.update(other_ints)
            
            # Filter predictions
            filtered_dict[model][threshold] = predictions[
                predictions[['PredictiveFeature', 'TargetFeature']].apply(tuple, axis=1).isin(combined_other_ints)
            ]
    
    return filtered_dict


# ----------------- #  

if __name__ == '__main__':

    args = parse_arguments()

    preds_dict, ref_dict, value_cols = format_df_per_model_and_threshold(
        base_dir=args.base_dir,
        network_name=args.network_name, 
        model_types=args.model_types, 
        thresholds=args.thresholds
    )

    print("Computing AUCs for all models and thresholds...")
    auc_results = calculate_aucs(
        preds_dict=preds_dict,
        ref_dict=ref_dict,
        value_cols=value_cols,
        base_dir=args.base_dir,
        network_name=args.network_name,
        model_types=args.model_types, 
        thresholds=args.thresholds,
        int_type='all_interactions'
    )

    # print("Plotting AUCs for all models and thresholds...")
    # plot_aucs(
    #     auc_df=auc_results, 
    #     base_dir=args.base_dir, 
    #     network_name=args.network_name,
    #     int_type='all_interactions',
    #     title_suffix='all interactions'
    # )
    # # ----------------- #

    # print("Evaluating interactions conserved across models...")
    # filtered_by_model = filter_by_conservation(preds_dict, 'model', args.model_types, args.thresholds)

    # conserved_model_auc_results = calculate_aucs(
    #     preds_dict=filtered_by_model,
    #     ref_dict=ref_dict,
    #     value_cols=value_cols,
    #     base_dir=args.base_dir,
    #     network_name=args.network_name,
    #     model_types=args.model_types, 
    #     thresholds=args.thresholds,
    #     int_type='conserved_by_model'
    # )

    # plot_aucs(
    #     auc_df=conserved_model_auc_results, 
    #     base_dir=args.base_dir, 
    #     network_name=args.network_name,
    #     int_type='conserved_by_model',
    #     title_suffix='conserved interactions by models'
    # )

    # # ----------------- #

    # print("Evaluating interactions conserved across thresholds...")
    # filtered_by_threshold = filter_by_conservation(preds_dict, 'threshold', args.model_types, args.thresholds)

    # conserved_threshold_auc_results = calculate_aucs(
    #     preds_dict=filtered_by_threshold,
    #     ref_dict=ref_dict,
    #     value_cols=value_cols,
    #     base_dir=args.base_dir,
    #     network_name=args.network_name,
    #     model_types=args.model_types, 
    #     thresholds=args.thresholds,
    #     int_type='conserved_by_threshold'
    # )

    # plot_aucs(
    #     auc_df=conserved_threshold_auc_results, 
    #     base_dir=args.base_dir, 
    #     network_name=args.network_name,
    #     int_type='conserved_by_threshold',
    #     title_suffix='conserved interactions by thresholds'
    # )

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')