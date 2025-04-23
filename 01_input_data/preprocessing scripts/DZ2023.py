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

dataset = 'DZ2023'

print('Loading raw data for', dataset, '...')
file_name = "1-s2.0-S0944711323002581-mmc3.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Significant',header=2)
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter data

#filtered columns
data = data.iloc[:, [3, 5, 6] + list(range(9, 27))]

data = data.rename(columns={'Gene names': 'GeneName'}) #rename GeneName
data.dropna(subset=['GeneName'], inplace=True)

data['Phosphosite'] = data['Amino acid'] + '(' + data['Position'].astype(str) + ')'

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data.drop(columns=['Position', 'Amino acid'], inplace=True)

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)
