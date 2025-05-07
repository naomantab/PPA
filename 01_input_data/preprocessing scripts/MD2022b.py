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

dataset = 'MD2022B'

print('Loading raw data for', dataset, '...')
file_name = "tjp14743-sup-0010-s10.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='DA_ZT4_PO', header=0)
print('Raw data loaded.')

# Probablity not given in paper

# Extract columns using list comprehension
d19_columns = data[[col for col in data.columns if 'D19' in col]]

# Display the filtered columns
filtered_columns = ['geneName','motifX','modifCoord']
filtered_columns.extend(list(d19_columns.columns))
data = data[filtered_columns].copy()

mod_aa = data['motifX'].str.extract(r'([STY])\*')  # Gets the modified amino acid
mod_pos = data['modifCoord']                      # Gets the position in protein
data['Phosphosite'] = mod_aa[0] + '(' + mod_pos.astype(str) + ')'

# format GeneName column
data.dropna(subset=['geneName'])
data = data.rename(columns={'geneName': 'GeneName'})

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

# clean up data
data.drop('motifX', axis=1, inplace=True)
data.drop('modifCoord', axis=1, inplace=True)

# log the data
data= preprocessing.log2_transform(data)

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)