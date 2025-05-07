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

dataset = 'MD2022'

print('Loading raw data for', dataset, '...')
file_name = "13041_2022_945_MOESM1_ESM.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Phosphoproteins', header=0)
print('Raw data loaded.')

# Probablity over 0.75

# Extract columns using list comprehension
abundance_columns = data[[col for col in data.columns if 'Abundance' in col]]

# Display the filtered columns
filtered_columns = ['Accession', 'Modifications']
filtered_columns.extend(list(abundance_columns.columns))
data = data[filtered_columns].copy()

# format the modifications column
data = data[~data['Modifications'].str.contains(';',  na=False)]
data = data.dropna(subset=['Modifications'])
data['Modifications'] = data['Modifications'].str.replace(
    r'Phospho\s*\[([STY])(\d+)\]', r'\1(\2)', regex=True)
data = data.rename(columns={'Modifications': 'Phosphosite'})

# convert accession ID to GeneName
data = data.rename(columns={'Accession': 'UniProtID'})
data = preprocessing.map_uniprot_to_gene(data)

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]
# drop columns
data = data[data['UniProtID'] != 'Gene name not found']
data.drop('UniProtID', axis=1, inplace=True)
data = data[~data['phosphosite_ID'].str.startswith('_', na=False)]

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