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

dataset = 'SY2019'

print('Loading raw data for', dataset, '...')
file_name = "Table S1 Intensity and knownsites information.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", header=0)
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter localization probability

data = data.rename(columns={'Gene names': 'GeneName'}) # rename column

# create new phosposite column
data['Phosphosite'] = data['Amino acid'].astype(str) + '(' + data['Position'].astype(str) + ')'

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# remove unnecessary columns
data.drop('Protein', axis=1, inplace=True)
data.drop('Protein names', axis=1, inplace=True)
data.drop('Proteins', axis=1, inplace=True)
data.drop('Positions within proteins', axis=1, inplace=True)
data.drop('Known site', axis=1, inplace=True)
data.drop('Sequence window', axis=1, inplace=True)
data.drop('PhosphoSitePlus window', axis=1, inplace=True)
data.drop('PhosphoSitePlus kinase', axis=1, inplace=True)
data.drop('Localization prob', axis=1, inplace=True)
data.drop('Amino acid', axis=1, inplace=True)
data.drop('Position', axis=1, inplace=True)

data = preprocessing.log2_transform(data) # log2 transform the data
print('Log2 transformation complete.')

# clean up command
data = preprocessing.clean_phosID_col(data)
print('Phosphosite IDs cleaned.')

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)