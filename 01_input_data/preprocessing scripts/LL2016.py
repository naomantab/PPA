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

dataset = 'LL2016'

print('Loading raw data for', dataset, '...')
file_name = "15357163mct150692-sup-154388_1_supp_3290495_j07cjp.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='FullDataset', header=0)
print('Raw data loaded.')

# Drop the first row if it contains "Survival (Days)"
data = data[data.iloc[:, 0] != "Survival (Days)"].reset_index(drop=True)
# Drop the first column
data = data.iloc[:, 1:]

data = data[~data['Phosphosite'].str.contains(',', na=False)]
data = data[~data['AltProtName'].str.contains('', na=False)]

filtered_columns = ['Gene', 'Phosphosite', 'Brain2', 'Brain3', 'Brain4', 'Brain5', 'Brain6', 'Brain7', 'Brain8',
                    'Tumor1','Tumor2','Tumor3','Tumor4','Tumor5','Tumor6','Tumor7','Tumor8',
                    'Tumor9','Tumor10','Tumor11','Tumor12','Tumor13','Tumor14']
data = data[filtered_columns].copy()

data = data.rename(columns={'Gene': 'GeneName'})
data['Phosphosite'] = data['Phosphosite'].str.replace(r'([YST])(\d+)', r'\1(\2)', regex=True)

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# capitalise the first col
data['phosphosite_ID'] = data['phosphosite_ID'].str.upper()

data = preprocessing.log2_transform(data)

# clean up command
data = preprocessing.clean_phosID_col(data)

# append dataset name
new_columns = [data.columns[0]] + [f"{dataset}_{col}" for col in data.columns[1:]]
data.columns = new_columns

# export the file
data.to_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{dataset}.csv', index = False) # save processed data to csv file
print(dataset, 'has been saved to CSV successfully!', data)
