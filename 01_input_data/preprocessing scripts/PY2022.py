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

dataset = 'PY2022'

print('Loading raw data for', dataset, '...')
file_name = "X17742_phosphoproteins1113.csv"
data = pd.read_csv(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}")
print('Raw data loaded.')


# Display the filtered columns
filtered_columns = [
    'Gene Symbol',
    'Modifications',
    'Abundances (by Bio. Rep.): OVX,1',
    'Abundances (by Bio. Rep.): OVX,2',
    'Abundances (by Bio. Rep.): OVX,3',
    'Abundances (by Bio. Rep.): OVX,4',
    'Abundances (by Bio. Rep.): Sham,5',
    'Abundances (by Bio. Rep.): Sham,6',
    'Abundances (by Bio. Rep.): Sham,7',
    'Abundances (by Bio. Rep.): Sham,8'
]

data = data[filtered_columns].copy()

# Clean up data
data = data[~data['Modifications'].str.contains(';', na=False)] # drop rows with multiple pos
data[['Amino acid', 'Position', 'Localization prob']] = data['Modifications'].str.extract(r'\[([A-Z])(\d+)\(([\d.]+)\)\]')
data['Position'] = data['Position'].astype(float).astype('Int64')  # Optional: convert to integer
data['Localization prob'] = data['Localization prob'].astype(float)

# filter probability
data = data[data['Localization prob'] >= 0.85] 

# rename GeneName column and remove blanks
data.rename(columns={'Gene Symbol': 'GeneName'}, inplace=True)
data = data[~data['GeneName'].str.contains(';|NA', na=False)]


# create phosphosite column
data['Phosphosite'] = data['Amino acid'].astype(str) + '(' + data['Position'].astype(str) + ')'

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]
# drop columns
data.drop('Amino acid', axis=1, inplace=True)
data.drop('Position', axis=1, inplace=True)
data.drop('Modifications', axis=1, inplace=True)
data.drop('Localization prob', axis=1, inplace=True)

# remove columns where all cells are empty
data = data.dropna(subset=data.columns[1:], how='all')

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# log the data
data= preprocessing.log2_transform(data)

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)