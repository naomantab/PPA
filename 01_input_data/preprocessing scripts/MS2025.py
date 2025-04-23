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

dataset = 'MS2025'

print('Loading raw data for', dataset, '...')
file_name = "11357_2025_1601_MOESM2_ESM.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Table S5', header=1)
print('Raw data loaded.')

# Probablity over 0.75

# Extract columns using list comprehension
intensity_columns = data[[col for col in data.columns if 'Intensity' in col]]

# Display the filtered columns
filtered_columns = ['Genes Names', 'Amino acid', "Position"]
filtered_columns.extend(list(intensity_columns.columns))