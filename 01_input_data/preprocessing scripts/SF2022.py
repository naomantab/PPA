#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np

grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

dataset = 'AW2014'

print('Loading raw data for', dataset, '...')
file_name = "elife-68648-supp1-v1.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name="AZ20_4HR_QuadrFiltered", header=0)
print('Raw data loaded.')


data = data[data['BestProb'] >= 0.85] # filter localization probability

# filter data
data['Sequence_window'] = data['Sequence_window'].str.replace('_', '')

data.rename(columns={'Sequence_window': 'Sequence'}, inplace=True) # fix column name to match preprocessing func

# Below temporarily removed as it may not be needed with the match_seq_to_genename() func
# data.rename(columns={'UniProtID': 'GeneName'}, inplace=True) # fix column name to match preprocessing func

preprocessing.match_seq_to_genename(data, 'Sequence')
print('Amino acid sequences matched to gene names.')

preprocessing.find_position_in_gene(data, 'Sequence')
print('Position Discovered')

preprocessing.get_position_and_gene(data, 'Sequence', 'StartPosition')

data['Phosphosite'] = data['Reisdue'].astype(str) + '_(' + data['StartPosition'].astype(str) + ')'

data = preprocessing.create_phos_ID(data)

column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file

print(dataset, 'has been saved to CSV successfully!', data)