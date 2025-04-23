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

dataset = 'SP2013'

print('Loading raw data for', dataset, '...')
file_name = "jah3274-sup-0001-tabless1-s2.xlsx"
data = pd.read_excel(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/raw_data/{file_name}", sheet_name='Table S2 P-sites', header=2)
print('Raw data loaded.')

filtered_columns = ['Gene Name(s)', 'Protein Start', 'Peptide Site',
                    'WT 0h.a', 'WT 0h.b', 'WT 16h.a', 'WT 16h.b',
                    'KO 0h.a', 'KO 0h.b', 'KO 16h.a', 'KO 16h.b']
data = data[filtered_columns].copy()