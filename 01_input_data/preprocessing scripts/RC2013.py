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

dataset = 'RC2013'

print('Loading raw data for', dataset, '...')
file_name = "tabless1tos8.xls"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='table S1-all phosphopeptides')
print('Raw data loaded.')

filtered_columns = ['Protein Descriptions', 'Amino Acid','Modified Sequence', 'Phospho (ST) Probabilities',
                    'M/L Log2 Normalized', 'H/L Log2 Normalized', 'H/M Log2 Normalized']
data = data[filtered_columns].copy()

data['GeneName'] = data['Protein Descriptions'].str.extract(r'GN=([\w\-]+)')
data.drop('Protein Descriptions', axis=1, inplace=True)

data['Phospho (ST) Probabilities'] = data['Phospho (ST) Probabilities'].str.replace(r"\([^)]*\)", "", regex=True)
data.rename(columns={'Phospho (ST) Probabilities': 'Sequence'}, inplace=True)

data = preprocessing.find_position_in_gene(data, 'Sequence')

data = data.dropna(subset=['StartPosition'])
data['StartPosition'] = data['StartPosition'].round(0).astype(int)
data = data[data['StartPosition'] != 0]
data['Phosphosite'] = data['Amino Acid'] + '(' + data['StartPosition'].astype(str) + ')'

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data.drop(columns=['StartPosition', 'Amino Acid', 'Modified Sequence', 'Sequence'], inplace=True)

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# final clean up
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)