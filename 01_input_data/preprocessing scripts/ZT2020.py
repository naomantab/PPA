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

dataset = 'ZT2020'

print('Loading raw data for', dataset, '...')
file_name = "pr0c00028_si_003.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='PH-WTvsSham-WT sorted')
print('Raw data loaded.')

# localization prob is >0.75 per paper

# drop columns
data.drop(columns=['Protein accession', 'PH-WT/Sham-WT Ratio', 'Regulated Type', 'PH-WT/Sham-WT P value', 'Protein description'], inplace=True)

data = data.rename(columns={'Gene name': 'GeneName'}) #rename GeneName
data.dropna(subset=['GeneName'], inplace=True)

data['Phosphosite'] = data['Amino acid'] + '(' + data['Position'].astype(str) + ')'

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data.drop(columns=['Position', 'Amino acid'], inplace=True)

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