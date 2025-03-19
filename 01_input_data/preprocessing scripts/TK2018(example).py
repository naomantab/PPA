#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np

grandparent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 
sys.path.append(grandparent_dir)
from funcs import preprocessing

# ----------------- #
# LOAD & CLEAN DATA
# ----------------- #

dataset = 'TK2018'

print('Loading raw data for', dataset, '...')
data = pd.read_csv('/data/home/bty449/ExplainableAI/RawData/EMBJ-37-e98745-s005.csv', header = 0)
print('Raw data loaded.')

data = data[data['Localization prob'] >= 0.85] # filter data

# filter data
data['Sequence window'] = data['Sequence window'].str.replace('_', '')

preprocessing.match_seq_to_genename(data, 'Sequence window')
print('Amino acid sequences matched to gene names.')

data['Phosphosite'] = data['Amino acid'].astype(str) + '(' + data['Positions within proteins'].astype(str) + ')'

keepcols = [38, 37] + [x for x in range(0, 10)] # columns to keep
data = data.iloc[:, keepcols] # keep only specified columns
data.iloc[:, 2:12] = data.iloc[:, 2:12].astype(float) # convert from string float

data = preprocessing.create_phos_ID(data) # call function to create phosphosite_ID column
print('Phosphosite IDs created.')
data = preprocessing.log2_transform(data)
print('Data has been log2 transformed.')


data = preprocessing.clean_phosID_col(data)

    
    
data.to_csv(f'/data/home/bty449/ExplainableAI/PreprocessedDatasets/{dataset}.csv', index = False) # save processed data to csv file

print(dataset, 'has been saved to CSV successfully!', data)

