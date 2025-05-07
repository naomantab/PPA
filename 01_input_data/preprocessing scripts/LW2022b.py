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

dataset = 'LW2022B'

print('Loading raw data for', dataset, '...')
file_name = "STIM1KOvsWT_proteomics+phosphoproteomics_2022 .xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Quant Phos SUMMARY')
print('Raw data loaded.')

# localization probabilty confirmed over 90% in paper

# format the Phosphosites column
data = data.rename(columns={'Residue_phosphorylated': 'Phosphosite'})
data = data[~data['Phosphosite'].str.contains(';',  na=False)]
data.dropna(subset=['Phosphosite'])
data['Phosphosite'] = data['Phosphosite'].str.replace(r'([STY])(\d+)', r'\1(\2)', regex=True)

# Extract columns using list comprehension
abundance_columns = data[[col for col in data.columns if 'abundance' in col]]

# Display the filtered columns
filtered_columns = ['gene_symbol','Phosphosite']
filtered_columns.extend(list(abundance_columns.columns))
data = data[filtered_columns].copy()

data = data.rename(columns={'gene_symbol': 'GeneName'}) #rename GeneName
data.dropna(subset=['GeneName'], inplace=True)

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

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
