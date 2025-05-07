#!/bin/python

import pandas as pd
import numpy as np
import copy
from functools import reduce


# ----------------- #

def create_dataframe_per_dataset(dataframe): # create new dataframe for each column in a larger dataset
    if 'phosphosite_ID' not in dataframe.index.names:
        dataframe.set_index('phosphosite_ID', inplace=True)
        
    dataset_list = [dataframe[[column]].dropna() for column in dataframe.columns]
    return dataset_list



# ----------------- #

def MinMax_normalize_and_merge(dictionary, scalar): # MinMax normalise all dataframes in a dictionary and return a merged dataframe
    MinMax_dict = copy.deepcopy(dictionary) # copy dictionary
    for dataset in MinMax_dict:
        if 'phosphosite_ID' not in MinMax_dict[dataset].index.names:
            MinMax_dict[dataset].set_index('phosphosite_ID', inplace=True)
        MinMax_dict[dataset][MinMax_dict[dataset].columns] = scalar.fit_transform(MinMax_dict[dataset][MinMax_dict[dataset].columns]) # MinMax normalise each dataset
        MinMax_dict[dataset].reset_index(inplace=True) # reset index
    MinMax_itr = MinMax_dict.values() # convert dict values (dfs) to list
    MinMax_merged = reduce(lambda  left,right: pd.merge(left,right,on=['phosphosite_ID'], how='outer'), MinMax_itr) # merge datasets on column
    return MinMax_merged