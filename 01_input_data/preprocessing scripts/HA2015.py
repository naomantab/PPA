#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np
import requests

grandparent_dir = os.path.abspath(os.getcwd())
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

dataset = 'HA2015'

print('Loading raw data for', dataset, '...')
file_name = "Phospho (STY)Sites_HA2015.txt"
data = pd.read_csv(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", delimiter='\t')
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter localization probability

# Extract columns using list comprehension
ratio_columns = data[[col for col in data.columns if 'Ratio' in col]]

# Display the filtered columns
filtered_columns = ['Gene names', 'Amino acid', "Position"]
filtered_columns.extend(list(ratio_columns.columns))

data = data[filtered_columns].copy()

# create phosphosite column
data['Phosphosite'] = data['Amino acid'].astype(str) + '_(' + data['Position'].astype(str) + ')'

# rename GeneName column and remove blanks
data.rename(columns={'Gene names': 'GeneName'}, inplace=True)
data = data.dropna(subset=['GeneName'])

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]
# drop columns
data.drop('Amino acid', axis=1, inplace=True)
data.drop('Position', axis=1, inplace=True)

# remove columns where all cells are empty
data = data.dropna(subset=data.columns[1:], how='all')

# drop rows where it contains non-conformant char
data = data[~data['phosphosite_ID'].str.contains((';|:|-'))]

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)