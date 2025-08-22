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

dataset = 'VW2021'

print('Loading raw data for', dataset, '...')
file_name = "Supplemental table S1 for Viengkhou et al_v7.xlsx"

# ----------------- #

data1 = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Supp Tab 8', header=0)
print('Raw data loaded.')

data1 = data1[data1['Phosphosite'].str.count('_') <= 1]

#filter columns
abundance_columns = data1[[col for col in data1.columns if 'Abundance' in col]]
filtered_columns = ['Phosphosite']
filtered_columns.extend(list(abundance_columns.columns))
data1 = data1[filtered_columns].copy()

# Extract UniProt ID
data1['UniProt_ID'] = data1['Phosphosite'].str.split('_').str[0]
data1['Phosphosite'] = data1['Phosphosite'].str.split('_').str[1]
data1['Phosphosite'] = data1['Phosphosite'].str.replace('Ser', 'S').str.replace('Thr', 'T').str.replace('Tyr', 'Y').str.replace(r'([STY])(\d+)', r'\1(\2)', regex=True)
preprocessing.map_uniprot_to_gene(data1, 'UniProt_ID', 'Gene_Symbol')
data1 = data1.rename(columns={'Gene_Symbol': 'GeneName'})
data1 = data1[data1['GeneName'].notna() & (data1['GeneName'] != '')]

data1 = preprocessing.create_phos_ID(data1) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# drop columns
data1.drop('UniProt_ID', axis=1, inplace=True)
# reposition column
column_name = 'phosphosite_ID'
data1 = data1[[column_name] + [col for col in data1.columns if col != column_name]]

# log2 transform
data1= preprocessing.log2_transform(data1)

# clean up command
data1 = preprocessing.clean_phosID_col(data1)




# ----------------- #

data2 = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Supp Tab 12', header=0)
print('Raw data loaded.')

data2 = data2[data2['Phosphosite'].str.count('_') <= 1]

# filter columns
intensity_columns = data2[[col for col in data2.columns if 'Intensity' in col]]
filtered_columns = ['Phosphosite']
filtered_columns.extend(list(intensity_columns.columns))
data2 = data2[filtered_columns].copy()
data2 = data2.rename(columns={
    'Summed Intensity (NTC)': 'Astrocytes Summed Intensity (NTC)',
    'Summed Intensity (5 mins)': 'Astrocytes Summed Intensity (5 mins)',
    'Summed Intensity (15 mins)': 'Astrocytes Summed Intensity (15 mins)',
    'Summed Intensity (30 mins)': 'Astrocytes Summed Intensity (30 mins)'
})

# Extract UniProt ID
data2['UniProt_ID'] = data2['Phosphosite'].str.split('_').str[0]
data2['Phosphosite'] = data2['Phosphosite'].str.split('_').str[1]
data2['Phosphosite'] = data2['Phosphosite'].str.replace('Ser', 'S').str.replace('Thr', 'T').str.replace('Tyr', 'Y').str.replace(r'([STY])(\d+)', r'\1(\2)', regex=True)
preprocessing.map_uniprot_to_gene(data2, 'UniProt_ID', 'Gene_Symbol')
data2 = data2.rename(columns={'Gene_Symbol': 'GeneName'})
data2 = data2[data2['GeneName'].notna() & (data2['GeneName'] != '')]

data2 = preprocessing.create_phos_ID(data2)  # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# drop columns
data2.drop('UniProt_ID', axis=1, inplace=True)
# reposition column
column_name = 'phosphosite_ID'
data2 = data2[[column_name] + [col for col in data2.columns if col != column_name]]

# log2 transform
data2 = preprocessing.log2_transform(data2)

# clean up command
data2 = preprocessing.clean_phosID_col(data2)




# ----------------- #

data3 = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Supp Tab 16', header=0)
print('Raw data loaded.')

data3 = data3[data3['Phosphosite'].str.count('_') <= 1]

# filter columns
intensity_columns = data3[[col for col in data3.columns if 'Intensity' in col]]
filtered_columns = ['Phosphosite']
filtered_columns.extend(list(intensity_columns.columns))
data3 = data3[filtered_columns].copy()
data3 = data3.rename(columns={
    'Summed Intensity (NTC)': 'Microglia Summed Intensity (NTC)',
    'Summed Intensity (5 mins)': 'Microglia Summed Intensity (5 mins)',
    'Summed Intensity (15 mins)': 'Microglia Summed Intensity (15 mins)',
    'Summed Intensity (30 mins)': 'Microglia Summed Intensity (30 mins)'
})

# Extract UniProt ID
data3['UniProt_ID'] = data3['Phosphosite'].str.split('_').str[0]
data3['Phosphosite'] = data3['Phosphosite'].str.split('_').str[1]
data3['Phosphosite'] = data3['Phosphosite'].str.replace('Ser', 'S').str.replace('Thr', 'T').str.replace('Tyr', 'Y').str.replace(r'([STY])(\d+)', r'\1(\2)', regex=True)
preprocessing.map_uniprot_to_gene(data3, 'UniProt_ID', 'Gene_Symbol')
data3 = data3.rename(columns={'Gene_Symbol': 'GeneName'})
data3 = data3[data3['GeneName'].notna() & (data3['GeneName'] != '')]

data3 = preprocessing.create_phos_ID(data3)  # call function to create phosphosite_ID column
print('Phosphosite IDs created.')

# drop columns
data3.drop('UniProt_ID', axis=1, inplace=True)
# reposition column
column_name = 'phosphosite_ID'
data3 = data3[[column_name] + [col for col in data3.columns if col != column_name]]

# log2 transform
data3 = preprocessing.log2_transform(data3)

# clean up command
data3 = preprocessing.clean_phosID_col(data3)



# ----------------- #

data = pd.merge(data1, data2, on='phosphosite_ID', how='outer')
data = pd.merge(data, data3, on='phosphosite_ID', how='outer')

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