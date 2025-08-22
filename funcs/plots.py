#!/bin/python

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyvis.network import Network
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


# ----------------- #

def scatterplot_phos_vals(dataset, save_file_as):
    plt.figure()
    # create a long-form dataframe
    dataset_melt = dataset.melt(id_vars='DatasetName', var_name='Phosphosite', value_name='Value')

    dataset_num = len(dataset)
    # create viridis colourmap 
    viridis = plt.get_cmap('viridis')
    colours = viridis(np.linspace(0, 0.85, dataset_num))
    colours = [mcolors.to_hex(c) for c in colours]

    # create scatter plot
    sns.stripplot(x='DatasetName', y='Value', hue='DatasetName', data=dataset_melt, jitter=True, 
                  size = 1.7, alpha = 0.9, 
                  palette=colours)

    # plot settings
    plt.xlabel('Dataset')
    plt.ylabel('Log2 abundance values')
    plt.title(f'Phosphoproteomics matrix log2-transformed data visualisation')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='-')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    plt.savefig(f'/data/home/bty449/ExplainableAI/PlotsGraphsPNGs/{save_file_as}.png', dpi=300, bbox_inches='tight')
    print(f'Scatter plot of values saved successfully!')
    
    
# ----------------- #  

def lineplot_raw_phos_means(dataset, save_file_as):
    plt.figure()
    # plot mean value for each dataset
    means = dataset.set_index('DatasetName').mean(1)
    means.plot(kind='line')

    plt.xlabel('Dataset')
    plt.ylabel('Mean value')
    plt.title('Phosphoproteomics matrix mean dataset values')
    plt.xticks(rotation=90)

    # remove x-axis labels
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.savefig(f'/data/home/bty449/ExplainableAI/PlotsGraphsPNGs/{save_file_as}.png', dpi=300, bbox_inches='tight')
    print('Line plot of mean values saved successfully!')
    
    

# ----------------- #  
    
def lineplot_norm_phos_means(dataset, save_file_as):
    plt.figure()
    # plot mean value for each dataset
    means = dataset.set_index('DatasetName').mean(1)
    means.plot(kind='line')

    plt.xlabel('Dataset')
    plt.ylabel('Mean value')
    plt.title('Phosphoproteomics matrix mean dataset values')
    plt.xticks(rotation=90)
    
    # Set the y scale between 0 and 1
    plt.ylim(0, 1)

    # remove x-axis labels
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.savefig(f'/data/home/bty449/ExplainableAI/PlotsGraphsPNGs/{save_file_as}.png', dpi=300, bbox_inches='tight')
    print('Line plot of mean values saved successfully!')
    
    

# ----------------- #  

def plot_CNN_or_XGB_predicted_network(base_dir, model, df, strong_percentile, medium_percentile, 
                                      weak_percentile, selected_prots, save_as_filename):
    """Plots network of selected proteins.
    
    Inputs:
    df <dataframe>: 4 column df - PredictiveFeature, TargetFeature, SHAPValue,
        and LogSHAPValue columns
    strong_boundary <int>: value to set at strongest log SHAP value cutoff
    medium_boundary <int>: value to set at medium log SHAP value cutoff
    weak_boundary <int>: value to set at weakest log SHAP value cutoff 
    selected_prots <tuple>: Tuple of string proteins to select for
    save_as_filename <str>: Filename to save output network to
    """
    
    df['SHAP*R2'] = pd.to_numeric(df['SHAP*R2'], errors='coerce')
    strong_boundary = np.percentile(df['SHAP*R2'].dropna(), strong_percentile)
    medium_boundary = np.percentile(df['SHAP*R2'].dropna(), medium_percentile)
    weak_boundary = np.percentile(df['SHAP*R2'].dropna(), weak_percentile)

    strong_boundary = strong_boundary
    medium_boundary = medium_boundary
    weak_boundary = weak_boundary

    print(f"Strong boundary: {strong_boundary}, Medium: {medium_boundary}, Weak: {weak_boundary}")
        
    nt = Network('1000px', '1000px', directed=True)
    for _, row in df.iterrows():
        PredictiveFeature = row['PredictiveFeature']
        TargetFeature = row['TargetFeature']
        logSHAPvalue = row['SHAP*R2']

        # Set default colors
        feature_color = '#a7b5e0'
        targetfeat_color = '#a7b5e0'

        # Update colors if features are in selected_prots
        if PredictiveFeature in selected_prots:
            feature_color = '#d16002'
        if TargetFeature in selected_prots:
            targetfeat_color = '#d16002'

        strong_color = '#311432'
        medium_color = '#a84296'
        weak_color = '#d8bfd8'

        node_fontsize = 25

        # for feedback loops
        if PredictiveFeature == TargetFeature:
            if logSHAPvalue >= strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=strong_color, value=2)
            elif logSHAPvalue >= medium_boundary and logSHAPvalue < strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=medium_color, value=2)
            elif logSHAPvalue >= weak_boundary and logSHAPvalue < medium_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=weak_color, value=2)
        # for all other edges
        else:
            if logSHAPvalue >= strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=strong_color, value=2)
            elif logSHAPvalue >= medium_boundary and logSHAPvalue < strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=medium_color, value=2)
            elif logSHAPvalue >= weak_boundary and logSHAPvalue < medium_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=weak_color, value=2)
                
    # Save the network graph

    nt.save_graph(f'{base_dir}/08_results/{model}/predicted_networks/{save_as_filename}')
    print('Predicted network graph saved!')
    
# ----------------- #  

def plot_two_way_XGB_predicted_network(base_dir, model, df, strong_percentile, medium_percentile, 
                                       weak_percentile, selected_prots, save_as_filename):
    """Plots a directional network of selected proteins, using SHAP*R2 values to encode both strength and direction.
    
    Inputs:
    - df <DataFrame>: Must contain columns 'PredictiveFeature', 'TargetFeature', 'SHAPValue', and 'SHAP*R2'
    - strong_percentile, medium_percentile, weak_percentile <int>: Cutoffs for edge strength
    - selected_prots <tuple>: Proteins to highlight in orange
    - save_as_filename <str>: Filename to save the resulting HTML network graph
    """
    
    df['SHAP*R2'] = pd.to_numeric(df['SHAP*R2'], errors='coerce')
    df['SHAPValue'] = pd.to_numeric(df['SHAPValue'], errors='coerce')

    abs_shap_r2 = df['SHAP*R2'].abs().dropna()
    strong_boundary = np.percentile(abs_shap_r2, 100 - strong_percentile)
    medium_boundary = np.percentile(abs_shap_r2, 100 - medium_percentile)
    weak_boundary = np.percentile(abs_shap_r2, 100 - weak_percentile)

    nt = Network('1000px', '1000px', directed=True)

    # Color palettes
    strong_positive = '#36441d'
    medium_positive = '#708d3e'
    weak_positive = '#d4e2bd'

    strong_negative = '#311432'
    medium_negative = '#a84296'
    weak_negative = '#d8bfd8'

    node_fontsize = 25

    for _, row in df.iterrows():
        PredictiveFeature = row['PredictiveFeature']
        TargetFeature = row['TargetFeature']
        shap_value = row['SHAPValue']
        shap_r2 = row['SHAP*R2']

        # Skip NaNs or zeros
        if pd.isna(shap_value) or pd.isna(shap_r2) or shap_r2 == 0:
            continue

        # Determine node colors
        feature_color = '#a7b5e0'
        targetfeat_color = '#a7b5e0'
        if PredictiveFeature in selected_prots:
            feature_color = '#d16002'
        if TargetFeature in selected_prots:
            targetfeat_color = '#d16002'

        abs_val = abs(shap_r2)

        # Assign edge color by sign and strength
        if shap_value > 0:
            if abs_val >= strong_boundary:
                edge_color = strong_positive
            elif abs_val >= medium_boundary:
                edge_color = medium_positive
            elif abs_val >= weak_boundary:
                edge_color = weak_positive
            else:
                edge_color = '#eef6dd'  # very light green for very weak pos
        elif shap_value < 0:
            if abs_val >= strong_boundary:
                edge_color = strong_negative
            elif abs_val >= medium_boundary:
                edge_color = medium_negative
            elif abs_val >= weak_boundary:
                edge_color = weak_negative
            else:
                edge_color = '#f1e2f4'  # very light purple for very weak neg
        else:
            continue  # skip truly neutral SHAPs

        # Add nodes and edge
        if PredictiveFeature not in [n['id'] for n in nt.nodes]:
            nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
        if TargetFeature not in [n['id'] for n in nt.nodes]:
            nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
        nt.add_edge(PredictiveFeature, TargetFeature, color=edge_color, value=2)


    # Save graph
    nt.save_graph(f'{base_dir}/08_results/{model}/predicted_networks/{save_as_filename}')
    print('Predicted network graph saved!')

# ----------------- #

def plot_predicted_network(base_dir, model_type, df, strong_percentile, medium_percentile,
                              weak_percentile, selected_prots, save_as_filename):
    """Plots network of selected proteins.
    
    Inputs:
    df <dataframe>: 4 column df - PredictiveFeature, TargetFeature, SHAPValue,
        and LogSHAPValue columns
    strong_boundary <int>: value to set at strongest log SHAP value cutoff
    medium_boundary <int>: value to set at medium log SHAP value cutoff
    weak_boundary <int>: value to set at weakest log SHAP value cutoff
    selected_prots <tuple>: Tuple of string proteins to select for
    save_as_filename <str>: Filename to save output network to
    """
    if model_type == 'linear_regression':
        abs_value = df['MedianCoeff'].abs()
    else:
        try:
            abs_value = df['SHAP*R2'].abs()
        except KeyError:
            abs_value = df['EdgeScore'].abs() # change to 'MeanEdgeWeight' if needed
    strong_boundary = np.percentile(abs_value, 100 - strong_percentile)
    medium_boundary = np.percentile(abs_value, 100 - medium_percentile)
    weak_boundary = np.percentile(abs_value, 100 - weak_percentile)
 
    nt = Network('1000px', '1000px', directed=True)
    for row in df.iterrows():
        PredictiveFeature = row[1][0]
        TargetFeature = row[1][1]
        coeff_or_shap = row[1][4] # change to row[1][2] if using abs_value of 'MeanEdgeWeight'
 
        # Set default colors
        feature_color = '#a7b5e0'
        targetfeat_color = '#a7b5e0'
 
        # Update colors if features are in selected_prots
        if PredictiveFeature in selected_prots:
            feature_color = '#d16002'
        if TargetFeature in selected_prots:
            targetfeat_color = '#d16002'
 
        strong_positive = '#36441d'
        medium_positive = '#708d3e'
        weak_positive = '#d4e2bd'
        
        strong_negative = "#9B3F8B"  # Lighter strong negative color
        medium_negative = "#B963AC"  # Keeping the medium as is (it’s already a balanced shade)
        weak_negative = "#F2D8F2" 
 
        node_fontsize = 25
    
        # for feedback loops
        if PredictiveFeature == TargetFeature:
            if coeff_or_shap <= -strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=strong_negative, value=2)
            elif coeff_or_shap <= -medium_boundary and coeff_or_shap > -strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=medium_negative, value=2)
            elif coeff_or_shap <= -weak_boundary and coeff_or_shap > -medium_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=weak_negative, value=2)
            elif coeff_or_shap >= strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=strong_positive, value=2)
            elif coeff_or_shap >= medium_boundary and coeff_or_shap < strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=medium_positive, value=2)
            elif coeff_or_shap >= weak_boundary and coeff_or_shap < medium_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=weak_positive, value=2)
        # for all other edges
        else:
            if coeff_or_shap <= -strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=strong_negative, value=2)
            elif coeff_or_shap <= -medium_boundary and coeff_or_shap > -strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=medium_negative, value=2)
            elif coeff_or_shap <= -weak_boundary and coeff_or_shap > -medium_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=weak_negative, value=2)
            elif coeff_or_shap >= strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=strong_positive, value=2)
            elif coeff_or_shap >= medium_boundary and coeff_or_shap < strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=medium_positive, value=2)
            elif coeff_or_shap >= weak_boundary and coeff_or_shap < medium_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=weak_positive, value=2)
            
    nt.save_graph(f'{base_dir}/08_results/{model_type}/predicted_networks/{save_as_filename}')
    print('Predicted network graph saved!')

# ----------------- #  

def plot_LR_predicted_network(base_dir, df, strong_percentile, medium_percentile, 
                              weak_percentile, selected_prots, save_as_filename):
    """Plots network of selected proteins.
    
    Inputs:
    df <dataframe>: 4 column df - PredictiveFeature, TargetFeature, SHAPValue,
        and LogSHAPValue columns
    strong_boundary <int>: value to set at strongest log SHAP value cutoff
    medium_boundary <int>: value to set at medium log SHAP value cutoff
    weak_boundary <int>: value to set at weakest log SHAP value cutoff 
    selected_prots <tuple>: Tuple of string proteins to select for
    save_as_filename <str>: Filename to save output network to
    """
    abs_coeff = df['Coeff*R2']
    strong_boundary = np.percentile(abs_coeff, 100 - strong_percentile)
    medium_boundary = np.percentile(abs_coeff, 100 - medium_percentile)
    weak_boundary = np.percentile(abs_coeff, 100 - weak_percentile)

    nt = Network('1000px', '1000px', directed=True)
    for row in df.iterrows():
        PredictiveFeature = row[1][0]
        TargetFeature = row[1][1]
        Coefficient = row[1][2]

        # Set default colors
        feature_color = '#a7b5e0'
        targetfeat_color = '#a7b5e0'

        # Update colors if features are in selected_prots
        if PredictiveFeature in selected_prots:
            feature_color = '#d16002'
        if TargetFeature in selected_prots:
            targetfeat_color = '#d16002'

        strong_positive = '#36441d'
        medium_positive = '#708d3e'
        weak_positive = '#d4e2bd'
        
        strong_negative = '#873c07'
        medium_negative = '#ca5a0b'
        weak_negative = '#f8bb8f'

        node_fontsize = 25
    
        # for feedback loops
        if PredictiveFeature == TargetFeature:
            if Coefficient <= -strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=strong_negative, value=2)
            elif Coefficient <= -medium_boundary and Coefficient > -strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=medium_negative, value=2)
            elif Coefficient <= -weak_boundary and Coefficient > -medium_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=weak_negative, value=2)
            elif Coefficient >= strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=strong_positive, value=2)
            elif Coefficient >= medium_boundary and Coefficient < strong_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=medium_positive, value=2)
            elif Coefficient >= weak_boundary and Coefficient < medium_boundary:
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, PredictiveFeature, color=weak_positive, value=2)
        # for all other edges
        else:
            if Coefficient <= -strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=strong_negative, value=2)
            elif Coefficient <= -medium_boundary and Coefficient > -strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=medium_negative, value=2)
            elif Coefficient <= -weak_boundary and Coefficient > -medium_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=weak_negative, value=2) 
            elif Coefficient >= strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=strong_positive, value=2)
            elif Coefficient >= medium_boundary and Coefficient < strong_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=medium_positive, value=2)
            elif Coefficient >= weak_boundary and Coefficient < medium_boundary:
                if TargetFeature not in nt.nodes:
                    nt.add_node(TargetFeature, color=targetfeat_color, font={'size': node_fontsize})
                if PredictiveFeature not in nt.nodes:
                    nt.add_node(PredictiveFeature, color=feature_color, font={'size': node_fontsize})
                nt.add_edge(PredictiveFeature, TargetFeature, color=weak_positive, value=2)
            
    nt.save_graph(f'{base_dir}/08_results/linear_regression/predicted_networks/{save_as_filename}')
    print('Predicted network graph saved!')
    
    
# ----------------- #  

def coefficients_histogram_for_specific_proteins(min_vals, proteins, network, strong_boundary, medium_boundary, weak_boundary):
    """Plots a histogram of coefficients against the number of predictive features for specified proteins."""
    
    df = pd.read_csv(f'/data/home/bty449/ExplainableAI/LinearRegression/LR_Coefficients_Min{min_vals}Vals.csv', header=0) 

    # subset dataframe to only include rows with specified proteins
    dfs = []
    for i in proteins:
        target_feat_in_prots = df[df['TargetFeature'].str.contains(i)]
        dfs.append(target_feat_in_prots)
        predictive_feat_in_prots = df[df['Feature'].str.contains(i)]
        dfs.append(predictive_feat_in_prots)

    dfs = pd.concat(dfs)
    dfs = dfs[['TargetFeature', 'Feature', 'Coefficient']]
    sorted_df = dfs.sort_values(by=['Coefficient'], ascending=False)
    lr_coeff_vals = sorted_df['Coefficient']

    plt.figure()
    plt.hist(lr_coeff_vals, bins=100, log=True)
    plt.axvline(x=-strong_boundary, ls='--', color='#873c07')
    plt.axvline(x=-medium_boundary, ls='--', color='#ca5a0b')
    plt.axvline(x=-weak_boundary, ls='--', color='#f8bb8f')
    plt.axvline(x=strong_boundary, ls='--', color='#36441d')
    plt.axvline(x=medium_boundary, ls='--', color='#708d3e')
    plt.axvline(x=weak_boundary, ls='--', color='#d4e2bd')
    # plt.text(-6.8,10000,'strong\nnegative', fontsize=8, color='#873c07')
    # plt.text(-3.3, 10000,'medium\nnegative', fontsize=8, color='#ca5a0b')
    # plt.text(-2.3, 4500, 'weak\nnegative', fontsize=8, color='#f8bb8f')
    # plt.text(7.2, 10000,'strong\npositive', fontsize=8, color='#36441d')
    # plt.text(3.7, 4500,'medium\npositive', fontsize=8, color='#708d3e')
    # plt.text(2.7, 10000, 'weak\npositive', fontsize=8, color='#d4e2bd')
    plt.xlabel('Coefficient values')
    plt.ylabel('Frequency of features (log scale)')
    plt.title(f'Frequency distribution for LR \nCoefficient values ({network} proteins)')
    plt.savefig(f'/data/home/bty449/ExplainableAI/PlotsGraphsPNGs/LR_CoefficientValues_{network}_Histogram_Min{min_vals}Vals.png', dpi=300, bbox_inches='tight')
    print(f"Histogram distribution of coefficients for Min{min_vals}Vals saved successfully!")
    

# ----------------- # 

def plot_logged_SHAP_values_for_each_threshold(model_type, network_name, 
                                                             log_shap_vals_50, log_shap_vals_100, 
                                                             log_shap_vals_150, log_shap_vals_200):
    """Plot 4 histograms (one figure) of logged SHAP values, one plot per threshold.
    
    This plot DOESN'T have distribution (Gaussian or normal) lines.
    """
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    data = [log_shap_vals_50, log_shap_vals_100, log_shap_vals_150, log_shap_vals_200]
    titles = ['Threshold: 50', 'Threshold: 100', 'Threshold: 150', 'Threshold: 200']
    for i, ax in enumerate(axs.flat):
        ax.hist(data[i], bins=100)
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Logged SHAP values', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, linestyle='-', alpha=0.3)
        # ax.grid(True, linestyle='-', alpha=0.3)
    # Set the overall figure title
    fig.suptitle(f'Comparison of {model_type} logged SHAP values across\ndifferent thresholds (models optimised for MSE)')
    fig.subplots_adjust(hspace=0.35)
    plt.savefig(f'/data/home/bty449/ExplainableAI/07_results/{model_type}/figures/{model_type}_loggedSHAPvalues_{network_name}_histogram_all_thresholds.png', dpi=300, bbox_inches='tight')
    
    
# ----------------- #  
    
def plot_logged_SHAP_values_distribution_for_one_threshold(min_vals, model_type, network_name, log_shap_vals):
    """Plot a histogram of logged SHAP values for a specific threshold."""
        
    gmm = GaussianMixture(n_components=2)
    log_shap_vals_reshaped = log_shap_vals.reshape(-1, 1)
    gmm.fit(log_shap_vals_reshaped)
    mu1, mu2 = gmm.means_.flatten()
    std1, std2 = np.sqrt(gmm.covariances_).flatten()
    weight1, weight2 = gmm.weights_
    
    plt.figure()
    count, bins, ignored = plt.hist(log_shap_vals, bins=100, density=False)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p1 = norm.pdf(x, mu1, std1)
    p2 = norm.pdf(x, mu2, std2)
    p1_scaled = p1 * weight1 * len(log_shap_vals) * (bins[1] - bins[0])
    p2_scaled = p2 * weight2 * len(log_shap_vals) * (bins[1] - bins[0])
    p_combined = p1_scaled + p2_scaled

    # combine the two normal distributions
    plt.plot(x, p_combined, 'k', linewidth=2)
    plt.xlabel('Logged SHAP values')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.title(f'Distribution for {model_type} logged SHAP values\n({network_name} proteins) - threshold: {min_vals}, optimised for MSE')
    plt.savefig(f'/data/home/bty449/ExplainableAI/07_results/{model_type}/figures/{model_type}_loggedSHAPvalues_{network_name}_histogram_min{min_vals}vals.png', dpi=300, bbox_inches='tight')
    
    print(f'Plot of logged CNN SHAP values (threshold: {min_vals}) saved successfully!')
    
    
# ----------------- #

def plot_logged_SHAP_values_distribution_for_all_thresholds(model_type, network_name, 
                                                             log_shap_vals_50, log_shap_vals_100, 
                                                             log_shap_vals_150, log_shap_vals_200):
    """Plot 4 individual histograms of logged SHAP values, one plot per threshold.
    
    distributions (list): List of distribution types for each dataset. Options: 'normal', 'gaussian', or None.
    """
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    data = [log_shap_vals_50, log_shap_vals_100, log_shap_vals_150, log_shap_vals_200]
    titles = ['Threshold: 50', 'Threshold: 100', 'Threshold: 150', 'Threshold: 200']
    
    for i, ax in enumerate(axs.flat):
        # Plot the histogram
        count, bins, _ = ax.hist(data[i], bins=100, density=False)

        gmm = GaussianMixture(n_components=2)
        reshaped_data = np.array(data[i]).reshape(-1, 1)
        gmm.fit(reshaped_data)
        mu1, mu2 = gmm.means_.flatten()
        std1, std2 = np.sqrt(gmm.covariances_).flatten()
        weight1, weight2 = gmm.weights_
        
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        bin_width = bins[1] - bins[0]
        p1 = norm.pdf(x, mu1, std1) * weight1
        p2 = norm.pdf(x, mu2, std2) * weight2
        p_combined = (p1 + p2) * bin_width * len(data[i])
        ax.plot(x, p_combined, 'k', linewidth=2, label=f'GMM: μ1={mu1:.2f}, μ2={mu2:.2f}')
        
        # Add labels and titles
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Logged SHAP values', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, linestyle='-', alpha=0.3)
    
    # Set the overall figure title
    fig.suptitle(f'Comparison of {model_type} logged SHAP values for {network_name}\nnetwork across different thresholds (models optimised for MSE)', fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(f'/data/home/bty449/ExplainableAI/07_results/{model_type}/figures/{model_type}_loggedSHAPvalues_{network_name}_histogram_all_thresholds.png', dpi=300, bbox_inches='tight')
    plt.show()
    