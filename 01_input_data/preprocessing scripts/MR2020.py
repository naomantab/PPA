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

dataset = 'MR2020'

print('Loading raw data for', dataset, '...')
file_name = "msb20209819-sup-0002-datasetev1.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name="CK-p25", header=0)
print('Raw data loaded.')

# drop columns
data.drop(['Proteins', 'Uniprot Accessions', 'Sequence', '# Samples Measured', 'Cell Type Prediction', '# PSMs', 
           'CaMKII Motif (# residues)', 'CDK Motif (# residues)'], axis=1, inplace=True) 

data.drop([
    "CK-p25 vs CK 2wk + 7wk Hip + Cortex-FC","CK-p25 vs CK 2wk + 7wk Hip + Cortex-p",
    "CK-p25 vs CK 2wk Hip-FC","CK-p25 vs CK 2wk Hip-p",
    "CK-p25 vs CK 2wk Cortex-FC","CK-p25 vs CK 2wk Cortex-p",
    "CK-p25 vs CK 2wk Cere-FC","CK-p25 vs CK 2wk Cere-p",
    "CK-p25 vs CK 7wk Hip-FC","CK-p25 vs CK 7wk Hip-p"], axis=1, inplace=True)

# remove non conformat modifications
data = data[~data['Modifications'].str.contains(',',  na=False)]
data = data[~data['Modifications'].str.contains('/',  na=False)]

# make phosphosite column
data["Modifications"] = data["Modifications"].str.replace(r"p([A-Z])(\d+)", r"\1(\2)", regex=True)
data = data.rename(columns={"Modifications": "Phosphosite"})

# rename GeneName column and remove blanks
data.rename(columns={'Genes': 'GeneName'}, inplace=True)
data = data.dropna(subset=['GeneName'])

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# cleaning up the dataframe final time
# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()
# reposition column
column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

# remove columns where all cells are empty
data = data.dropna(subset=data.columns[1:], how='all')

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# clean up data
data = data[~data['phosphosite_ID'].str.contains('/', na=False)]
data = data[~data['phosphosite_ID'].str.contains('NAN', na=False)]

# clean up command
data = preprocessing.clean_phosID_col(data)

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)