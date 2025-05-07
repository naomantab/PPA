#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np
import requests
import re

grandparent_dir = os.path.abspath(os.getcwd())
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

dataset = 'KM2022'

print('Loading raw data for', dataset, '...')
file_name = "cells-1620011-supplementary.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Table S2_Perseus Phospho', header=0)
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter data

# Extract columns using list comprehension
intensity_columns = data[[col for col in data.columns if 'Intensity' in col]]

# Display the filtered columns
filtered_columns = ['Gene names', "Position", 'Amino acid']
filtered_columns.extend(list(intensity_columns.columns))
data = data[filtered_columns].copy()

# get phosphosite ID
data['Phosphosite'] = data['Amino acid'] + '(' + data['Position'].astype(str) + ')'
data = data.dropna(subset=['Gene names'])
data = data[~data['Gene names'].str.contains('/',  na=False)]
data = data.rename(columns={'Gene names': 'GeneName'})
data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data.drop(columns=['Position', 'Amino acid'], inplace=True)

# log the data
data= preprocessing.log2_transform(data)

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)