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
# IDENTIFY IQR OUTLIERS
# ----------------- #

# raw_matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/MatrixCSVs/AlignedRawMatrix.csv', header = 0)
raw_matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/RawMatrix.csv', header = 0)

raw_matrix_indexed = raw_matrix.set_index('DatasetName')
raw_matrix_outliers = preprocessing.find_outliers_IQR(raw_matrix_indexed)
print('Number of IQR outliers per phosphosite:', raw_matrix_outliers.count())



# ----------------- #
# REMOVE IQR OUTLIERS
# ----------------- #

raw_matrix_no_outliers = preprocessing.drop_outliers_IQR(raw_matrix_indexed)
raw_matrix_no_outliers.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/RawMatrix_NoOutliers.csv', index=True)
print(f'Raw matrix with no IQR outliers has been saved:', raw_matrix_no_outliers)



# ----------------- #

print('Done!')