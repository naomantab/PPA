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

dataset = 'WN2010'

print('Loading raw data for', dataset, '...')
file_name = "pr1002214_si_001.xls"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Localization Prob. >=0.57', header=0)
print('Raw data loaded.')

print(data.columns)

data = data[data['Best Localization Probability '] >= 0.85] # filter data

# Display the filtered columns
filtered_columns = ['Gene Names', "Position", 'Amino Acid',
                    'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 
                    'Unnamed: 14','Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
                    'Unnamed: 19', 'Intensity']

data = data[filtered_columns].copy()

data = data.rename(columns={
    'Unnamed: 9': 'FACE experiment A',
    'Unnamed: 10': 'FACE experiment B',
    'Unnamed: 11': 'SAX experiment C',
    'Unnamed: 12': 'SAX experiment D',
    'Unnamed: 13': 'SAX experiment A',
    'Unnamed: 14': 'SAX experiment A second enrichment',
    'Unnamed: 15': 'SAX experiment B',
    'Unnamed: 16': 'SAX experiment B second enrichment',
    'Unnamed: 17': 'SEC1',
    'Unnamed: 18': 'SEC2',
    'Unnamed: 19' : 'SEC3'
})

data = data.dropna(subset=['Position'])
data['Position'] = data['Position'].astype(int)

data = data.dropna(subset=['Gene Names'])
data = data[~data['Gene Names'].str.contains(';',  na=False)]
data = data.rename(columns={'Gene Names': 'GeneName'})

data['Phosphosite'] = data['Amino Acid'] + '(' + data['Position'].astype(str) + ')'

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

data.drop(columns=['Position', 'Amino Acid'], inplace=True)

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)
