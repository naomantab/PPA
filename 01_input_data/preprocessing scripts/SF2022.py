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

dataset = 'SF2022'

print('Loading raw data for', dataset, '...')
file_name = "elife-68648-supp1-v1.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name="AZ20_4HR_QuadrFiltered", header=0)
print('Raw data loaded.')


data = data[data['BestProb'] >= 0.85] # filter localization probability

# remove unnecessary columns
data.drop('All_Parent_Proteins', axis=1, inplace=True)
data.drop('BestProb', axis=1, inplace=True)
data.drop('Sequence_window', axis=1, inplace=True)
data.drop('P.1', axis=1, inplace=True)
data.drop('P.2', axis=1, inplace=True)
data.drop('P.3', axis=1, inplace=True)
data.drop('Total_ExpObservations', axis=1, inplace=True)
data.drop('AZ20_ExpObservations', axis=1, inplace=True)
data.drop('RAD1_ExpObservations', axis=1, inplace=True)
data.drop('AZ20_Ratio_Avg', axis=1, inplace=True)
data.drop('AZ20_pval', axis=1, inplace=True)
data.drop('RAD1_Ratio_Avg', axis=1, inplace=True)
data.drop('RAD1_pval', axis=1, inplace=True)
data.drop('quadr', axis=1, inplace=True)
data.drop('STY.Q.X', axis=1, inplace=True)


# removes all multi-position rows
data = data[~data['ProtName_site'].str.contains(' ')]
data.drop('ProtName_site', axis=1, inplace=True)

# renames column
data = data.rename(columns={'UniprotID': 'Phosphosite'})

# create new phosposite column
data['Phosphosite'] = data['Unnamed: 6'].astype(str) + '(' + data['position'].astype(str) + ')'

data = preprocessing.map_uniprot_to_gene(data)

# filter out results where gene name isnt found
data = data[~data['GeneName'].str.startswith('Gene name not found')].reset_index(drop=True)

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]
# remove incorrect amino acids
data = data[~data['Unnamed: 6'].isin(['G', 'D', 'A', 'P', 'R'])]
# remove incorrect IDs
data = data[~data['phosphosite_ID'].str.startswith(('_S_', '_T_'))]
# drop columns
data.drop('UniProtID', axis=1, inplace=True)
data.drop('position', axis=1, inplace=True)
data.drop('Unnamed: 6', axis=1, inplace=True)
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




