#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np

grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

dataset = 'AW2014'

print('Loading raw data for', dataset, '...')
file_name = "13058_2014_437_MOESM2_ESM.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", header=6)
print('Raw data loaded.')


data = data[data['Localization Prob'] >= 0.85] # filter data

data['Phosphosite'] =  'Y' + '(' + data['pY site'].astype(str) + ')' # combine Y and position

data.rename(columns={'Entry name': 'GeneName'}, inplace=True) # fix column name to match preprocessing func

# remove unnecessary columns
data.drop('Leading Proteins', axis=1, inplace=True)
data.drop('Protein name', axis=1, inplace=True)
data.drop('Localization Prob', axis=1, inplace=True)
data.drop('pY site', axis=1, inplace=True)



data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

data = preprocessing.log2_transform(data)
print('Data has been log2 transformed.')

column_name = 'phosphosite_ID'
data = data[[column_name] + [col for col in data.columns if col != column_name]]

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

    
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)
