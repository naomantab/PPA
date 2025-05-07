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


file_name = "RawMatrix.csv"
data = pd.read_csv(f"C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/{file_name}")
print('Raw data loaded.')

data.describe()
data.info()