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

dataset = 'LY2022'

print('Loading raw data for', dataset, '...')
file_name = "Appendix 3 -Phosphorylation modification sites identification table.xlsx"

data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}")
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter data

intensity_columns = data[[col for col in data.columns if 'Intensity' in col]]
filtered_columns = ['Gene names','Positions within proteins', 'Phospho (STY) Probabilities']
filtered_columns.extend(list(intensity_columns.columns))
data = data[filtered_columns].copy()
data.drop('Intensity', axis=1, inplace=True)


# Extract STY amino acids and probabilities using regex
data['STY_Probabilities'] = data['Phospho (STY) Probabilities'].apply(
    lambda x: re.findall(r'([STY])\((\d+\.\d+|\d+)\)', x))
# Find the STY amino acid with the highest probability and store it in 'Amino Acid' column
data['Amino Acid'] = data['STY_Probabilities'].apply(
    lambda x: max(x, key=lambda item: float(item[1]))[0] if x else None)


data['Phosphosite'] = data['Amino Acid'] + '(' + data['Positions within proteins'].astype(str) + ')'

#drop the unneeded columns
data.drop('STY_Probabilities', axis=1, inplace=True)
data.drop('Phospho (STY) Probabilities', axis=1, inplace=True)
data.drop('Amino Acid', axis=1, inplace=True)
data.drop('Positions within proteins', axis=1, inplace=True)
data = data[~data['Gene names'].str.contains(';', na=False)]
data = data.rename(columns={'Gene names': 'GeneName'})

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

# log the data
data= preprocessing.log2_transform(data)

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)