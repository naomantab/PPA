#!/bin/python

# Run from command line:
# python omnipath_evaluation.py --base_dir /path/to/project --network_name my_network --model_types linear_regression xgboost --thresholds 50 100
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
    parser = argparse.ArgumentParser(description='Evaluate protein interaction predictions against Biogrid reference.')
    parser.add_argument('--base_dir', type=str, default='/Users/caitlinmellor/Documents/GitHub/ExplainableAI', help='Base directory for the project')
    parser.add_argument('--network_name', type=str, default=None, help='Network name if analyzing specific network')
    parser.add_argument('--model_types', nargs='+', default=['linear_regression', 'xgboost', 'cnn'], 
                        help='Model types to evaluate')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[50, 100, 150, 200], 
                        help='Thresholds to evaluate')
    return parser.parse_args()

def get_file_path(base_dir, network_name, model_type, threshold, file_type, int_type):
    """Construct file paths based on parameters."""
    if file_type == 'input_preds':
        if model_type == 'linear_regression':
            if network_name is None:
                return f'{base_dir}/08_results/linear_regression/coefficients/LR_Coefficients_Min{threshold}Vals.csv'
            else:
                return f'{base_dir}/08_results/linear_regression/coefficients/LR_{network_name}_coefficients_protein_level_min{threshold}vals.csv'
        else:
            if network_name is None:
                return f'{base_dir}/08_results/{model_type}/concatenated_shaps/{model_type}_concatenated_all_shap_values_protein_level_min{threshold}vals.csv'
            else:
                return f'{base_dir}/08_results/{model_type}/concatenated_shaps/{model_type}_concatenated_{network_name}_shap_values_protein_level_min{threshold}vals.csv'
    elif file_type == 'csv_output':
        filename = f'biogrid_aucs_{int_type}'
        if network_name is not None:
            filename = f'{filename}_{network_name}'
        return f'{base_dir}/08_results/biogrid_results/{filename}.csv'
    else:
        # For csv outputs
        filename = f'biogrid_aucs_{int_type}'
        if network_name is not None:
            filename = f'{filename}_{network_name}'
        return f'{base_dir}/08_results/biogrid_results/{filename}.png'
    
def load_reference_data(base_dir):
    """Load and format the Biogrid reference dataset."""
    df = pd.read_csv(f"{base_dir}/01_input_data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-4.4.243.tab3.txt", sep="\t", header=0, low_memory=False)
    df = df[['Official Symbol Interactor A', 'Official Symbol Interactor B']]
    df2 = df[['Official Symbol Interactor B', 'Official Symbol Interactor A']]
    df2.columns = ['Official Symbol Interactor A', 'Official Symbol Interactor B']
    df = pd.concat([df, df2], axis=0)
    df.columns = ['PredictiveFeature', 'TargetFeature']
    df.drop_duplicates(inplace=True)
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
        df = df[['PredictiveFeature', 'TargetFeature', 'Coefficient']]
        value_col = 'Coefficient'
    else:
        df = df[['PredictiveFeature', 'TargetFeature', 'LogSHAPValue']]
        value_col = 'LogSHAPValue'
    
    # Remove duplicates
    df.dropna(subset=[value_col], inplace=True)
    df.drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'], inplace=True)
    print(f'Predicted dataframe ({model_type}, {threshold}): {df}')
    return df

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
    """Creates dictionaries of the prediction and reference datasets.
    
    Outputs:
    predictions_dict:   Dictionary of filtered predicted interactions {'model':{threshold: pd.DataFrame}} 
    reference_dict:     Dictionary of filtered reference interactions {'model':{threshold: pd.DataFrame}}
    value_cols:         Dictionary of the evaluation metric per model {'model': 'metric'}
    """

    # Ensure output directory exists
    os.makedirs(f'{base_dir}/08_results/biogrid_results', exist_ok=True)
    
    print("Loading reference data...")
    reference_data = load_reference_data(base_dir)
    
    # Load and process predictions for all models and thresholds
    print("Loading and processing prediction data...")
    predictions_dict = {model: {} for model in model_types}
    reference_filtered_dict = {model: {} for model in model_types}

    print('Creating dictionaries of predictions and reference dataframes...')
    for model in model_types:
        for threshold in thresholds:
            # Load predictions
            predictions = load_prediction_data(base_dir, network_name, model, threshold)
            # value_cols[model] = value_col
            
            # Filter for common proteins
            filtered_preds, filtered_ref = filter_for_common_proteins(predictions, reference_data)
            predictions_dict[model][threshold] = filtered_preds
            reference_filtered_dict[model][threshold] = filtered_ref

            # min-max normalise coefficients or logged SHAP values
            df = predictions_dict[model][threshold]
            if 'Coefficient' in df.columns:
                column_to_normalise = 'Coefficient'
            elif 'LogSHAPValue' in df.columns:
                column_to_normalise = 'LogSHAPValue'

            min_value = df[column_to_normalise].min()
            max_value = df[column_to_normalise].max()
            if max_value > min_value:  # Avoid division by zero
                df[column_to_normalise] = (df[column_to_normalise] - min_value) / (max_value - min_value)
            else:
                df[column_to_normalise] = 0.0  # If all values are the same, set them to 0

            df.columns = ['PredictiveFeature', 'TargetFeature', 'NormalisedCoeffOrSHAP']
            predictions_dict[model][threshold] = df

    return predictions_dict, reference_filtered_dict

def calculate_aucs(preds_dict, ref_dict, base_dir, network_name, model_types, thresholds, int_type):
    """Calculate PR and ROC AUC scores."""
    auc_results = []
    for model in model_types:
        model_results = []
        for threshold in thresholds:
            preds_dict[model][threshold] = preds_dict[model][threshold].drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'])
            ref_dict[model][threshold] = ref_dict[model][threshold].drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'])
            
            y_scores = preds_dict[model][threshold]['NormalisedCoeffOrSHAP'].tolist()

            # identify predicted interactions also found in reference dataset
            # all rows from predictions are in output (df length is same as predictions)
            # rows in both dfs = 'both' / rows just in predictions = 'left_only'
            merged_df = pd.merge(preds_dict[model][threshold], ref_dict[model][threshold], on=['PredictiveFeature', 'TargetFeature'],
                                how='left', indicator=True)
            print('merged_df:', merged_df.head())

            y_true = (merged_df['_merge'] == 'both').astype(int).tolist()
            print('y_true:', y_true[:5])
            print('y_scores:', y_scores[:5])

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(y_true, y_scores)
            print(f"Positive vs negative examples ({model}, {threshold}): {sum(y_true)} vs {len(y_true) - sum(y_true)}")
            
            model_results.append({
                'model_type': model,
                'threshold': threshold,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'positive_examples': sum(y_true),
                'negative_examples': len(y_true) - sum(y_true)
            })

        # if value_cols is not None:
        auc_results.extend(model_results)
        # else:
        #     continue

    # if value_cols is None:
        # auc_results.append({
        #     'model_type': 'all_models',
        #     'threshold': 'all_thresholds',
        #     'pr_auc': pr_auc,
        #     'roc_auc': roc_auc,
        #     'positive_examples': sum(y_true),
        #     'negative_examples': len(y_true) - sum(y_true)
        # })

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

    return auc_df

# ----------------- #
# Plots #

def plot_aucs_per_model_per_threshold(auc_df, base_dir, network_name, int_type, title_suffix):
    """Plot PR and ROC AUC scores across all models and thresholds
    
    Inputs:
    auc_df: A 12-row df of AUC scores, one row for each model and threshold.

    Outputs:
    Two separate bar plots - one for PR AUCs and one for ROC AUCs. Each plot has a 
    separate bar for each model and threshold (3 models & 4 thresholds = 12 bars). 
    """

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
    axs[0].legend(title='Min Vals', fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
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
    axs[1].legend(title='Min Vals', fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
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

def filter_for_all_models_and_thresholds(predictions_dict, model_types, thresholds):
    """
    Retain only interactions that are found in all models at all thresholds.
    
    Inputs:
    predictions_dict: Dictionary of predictions for each model and threshold.
    model_types: List of model types.
    thresholds: List of thresholds.
    
    Outputs:
    filtered_df: Filtered df containing only interactions found in all models and thresholds.
    """

    # initialise empty variable to store interactions
    common_interactions = None

    for model in model_types:
        for threshold in thresholds:
            # Get the interactions for the current model and threshold
            current_interactions = set(zip(
                predictions_dict[model][threshold]['PredictiveFeature'],
                predictions_dict[model][threshold]['TargetFeature'],
                predictions_dict[model][threshold]['NormalisedCoeffOrSHAP']
            ))
            print('current_interactions:', len(current_interactions))
            
            # Intersect with the existing common interactions
            # Intersect with the existing common interactions
            if common_interactions is None:
                # Initialize with only PredictiveFeature and TargetFeature
                common_interactions = current_interactions
            else:
                # Intersect based only on PredictiveFeature and TargetFeature
                common_interactions &= set((pf, tf) for pf, tf, _ in current_interactions)

            print('common_interactions:', len(common_interactions))
    # Convert the common interactions back to a DataFrame
    if common_interactions:
        filtered_df = pd.DataFrame(list(common_interactions), columns=['PredictiveFeature', 'TargetFeature', 'NormalisedCoeffOrSHAP'])
    else:
        filtered_df = pd.DataFrame(columns=['PredictiveFeature', 'TargetFeature', 'NormalisedCoeffOrSHAP'])  # Empty DataFrame if no common interactions

    return filtered_df

def plot_aucs_ints_in_all_models_and_thresholds(auc_df, base_dir, network_name, int_type, title_suffix):
    """Plot PR AUC and ROC AUC on a single graph for one model and one threshold.
    
    Inputs:
    auc_df: A 1-row df containing AUC scores for interactions in all models and thresholds.

    Outputs:
    A bar plot with two bars - one for PR AUC and one for ROC AUC.
    """

    # Extract PR AUC and ROC AUC values
    pr_auc = auc_df['pr_auc'].iloc[0]
    roc_auc = auc_df['roc_auc'].iloc[0]

    # Bar settings
    bar_labels = ['PR', 'ROC']
    bar_values = [pr_auc, roc_auc]
    bar_colors = ['#7985a5', '#f9993c']

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bar_labels, bar_values, color=bar_colors, width=0.5)

    # Format the plot
    ax.set_ylim(0, 1)  # AUC values range from 0 to 1
    ax.set_ylabel('AUC Score', fontsize=14)
    ax.set_xlabel('Classification evaluation metric', fontsize=14)
    network_text = f"{network_name} protein" if network_name else "protein"
    ax.set_title(f'AUC Scores for {network_text} {title_suffix}', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Save the plot
    plt.tight_layout()
    output_path = get_file_path(base_dir=base_dir, 
                                network_name=network_name, 
                                model_type=None, 
                                threshold=None,
                                file_type=None, 
                                int_type=int_type)
    plt.savefig(output_path)
    plt.close()

def filter_by_union_of_models(predictions_dict, model_types, thresholds):
    """
    Retain only interactions that are found in all models at all thresholds.
    
    Args:
        predictions_dict (dict): Dictionary of predictions for each model and threshold.
        model_types (list): List of model types.
        thresholds (list): List of thresholds.
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing interactions found in all models and thresholds.
    """
    # Initialize a set with interactions from the first model and threshold
    common_interactions = None

    for model in model_types:
        for threshold in thresholds:
            # Get the interactions for the current model and threshold
            current_interactions = set(zip(
                predictions_dict[model][threshold]['PredictiveFeature'],
                predictions_dict[model][threshold]['TargetFeature']
            ))
            
            # Intersect with the existing common interactions
            if common_interactions is None:
                common_interactions = current_interactions
            else:
                common_interactions &= current_interactions  # Keep only interactions present in all sets

    # Convert the common interactions back to a DataFrame
    if common_interactions:
        filtered_df = pd.DataFrame(list(common_interactions), columns=['PredictiveFeature', 'TargetFeature'])
    else:
        filtered_df = pd.DataFrame(columns=['PredictiveFeature', 'TargetFeature'])  # Empty DataFrame if no common interactions

    return filtered_df
  
def filter_by_minimum_two_models_or_thresholds(predictions_dict, filter_type, model_types, thresholds):
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
    
    # combine the interactions across models at each threshold
    combined_dict = {}
    for threshold in thresholds:
        # combine dfs for all models at the current threshold
        combined_df = pd.concat(
            [filtered_dict[model][threshold] for model in model_types if threshold in filtered_dict[model]],
            ignore_index=True
        )
        unique_interactions = combined_df.drop_duplicates(subset=['PredictiveFeature', 'TargetFeature'])
        if 'LogSHAPValue' in unique_interactions.columns and unique_interactions['LogSHAPValue'].isna().all():
            unique_interactions = unique_interactions.drop(columns=['LogSHAPValue'])
        combined_dict[threshold] = unique_interactions
        print(f'Combined dict {threshold}', combined_dict[threshold].head())
        print(len(combined_dict[threshold]))


    print(combined_dict)
    
    return filtered_dict

def plot_aucs_by_model_type(auc_df, base_dir, network_name, int_type, title_suffix):
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

def plot_aucs_by_threshold(auc_df, base_dir, network_name, int_type, title_suffix):
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


# ----------------- #  
    
if __name__ == '__main__':

    args = parse_arguments()

    print("Formatting reference and prediction data for all models and thresholds...")
    preds_dict, ref_dict = format_df_per_model_and_threshold(
        base_dir=args.base_dir,
        network_name=args.network_name, 
        model_types=args.model_types, 
        thresholds=args.thresholds
    )
    print("Predictions dictionary:", preds_dict)

    # print("Computing AUCs - one per model per threshold...")
    # auc_results = calculate_aucs(
    #     preds_dict=preds_dict,
    #     ref_dict=ref_dict,
    #     base_dir=args.base_dir,
    #     network_name=args.network_name,
    #     model_types=args.model_types, 
    #     thresholds=args.thresholds,
    #     int_type='all_interactions'
    # )
    # print("AUC results:", auc_results)


    # print("Plotting AUCs for all models and thresholds...")
    # plot_aucs_per_model_per_threshold(
    #     auc_df=auc_results, 
    #     base_dir=args.base_dir, 
    #     network_name=args.network_name,
    #     int_type='all_interactions',
    #     title_suffix='all interactions'
    # )

    # # ----------------- #

    print("Evaluating interactions conserved across all models and thresholds...")
    filtered_for_all = filter_for_all_models_and_thresholds(preds_dict, args.model_types, args.thresholds)
    print('Filtered for all:', filtered_for_all)

    # conserved_all_auc_results = calculate_aucs(
    #     preds_dict={model: {threshold: filtered_for_all for threshold in args.thresholds} for model in args.model_types},
    #     ref_dict=ref_dict,
    #     value_cols=None,
    #     base_dir=args.base_dir,
    #     network_name=args.network_name,
    #     model_types=args.model_types, 
    #     thresholds=args.thresholds,
    #     int_type='conserved_by_all'
    # )

    # plot_aucs_ints_in_all_models_and_thresholds(
    #     auc_df=conserved_all_auc_results, 
    #     base_dir=args.base_dir, 
    #     network_name=args.network_name,
    #     int_type='conserved_by_all',
    #     title_suffix='interactions\npredicted in all models and thresholds'
    # )

    # # ----------------- #

    # print("Evaluating interactions predicted in at least 2 models...")
    # filtered_by_model = filter_by_minimum_two_models_or_thresholds(preds_dict, 'model', args.model_types, args.thresholds)
    # print(filtered_by_model)

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

    # # # ----------------- #

    # print("Evaluating interactions conserved across thresholds...")
    # filtered_by_threshold = filter_by_minimum_two_models_or_thresholds(preds_dict, 'threshold', args.model_types, args.thresholds)

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

    # # plot_aucs(
    # #     auc_df=conserved_threshold_auc_results, 
    # #     base_dir=args.base_dir, 
    # #     network_name=args.network_name,
    # #     int_type='conserved_by_threshold',
    # #     title_suffix='conserved interactions by thresholds'
    # # )

    # # ----------------- #


    
    

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')