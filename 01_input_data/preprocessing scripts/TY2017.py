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

dataset = 'TY2017'

print('Loading raw data for', dataset, '...')
file_name = "mmc6.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='S5B',header=3)
print('Raw data loaded.')

# data = data[data['Localization prob'] >= 0.85] # filter localization probability

data.rename(columns={'Unnamed: 2': 'GeneName'}, inplace=True)
data.rename(columns={'Unnamed: 6': 'Phosphosite'}, inplace=True)
# Manually fix the ambiguous column names
new_column_names = list(data.columns)
new_column_names[9]  = 'WT 0h.a'
new_column_names[10] = 'WT 0h.b'
new_column_names[11] = 'WT 16h.a'
new_column_names[12] = 'WT 16h.b'
new_column_names[13] = 'KO 0h.a'
new_column_names[14] = 'KO 0h.b'
new_column_names[15] = 'KO 16h.a'
new_column_names[16] = 'KO 16h.b'

# Assign them back to the dataframe
data.columns = new_column_names

filtered_columns = ['GeneName', 'Phosphosite',
                    'WT 0h.a', 'WT 0h.b', 'WT 16h.a', 'WT 16h.b',
                    'KO 0h.a', 'KO 0h.b', 'KO 16h.a', 'KO 16h.b']
data = data[filtered_columns].copy()

data = data[~data['Phosphosite'].str.contains(',', case=False, na=False)]
data['Phosphosite'] = data['Phosphosite'].str.replace(r'^([A-Z])(\d+)$', r'\1(\2)', regex=True)

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data = preprocessing.log2_transform(data)

# final clean up
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)