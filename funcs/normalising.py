#!/bin/python

import pandas as pd
import numpy as np
import copy
from functools import reduce


# ----------------- #

# orignal function
# def create_dataframe_per_dataset(dataframe): # create new dataframe for each column in a larger dataset
#     if 'phosphosite_ID' not in dataframe.index.names:
#         dataframe.set_index('phosphosite_ID', inplace=True)
        
#     dataset_list = [dataframe[[column]].dropna() for column in dataframe.columns]
#     return dataset_list

def create_dataframe_per_dataset(dataframe): # create new dataframe for each column in a larger dataset
    if 'DatasetName' not in dataframe.index.names:
        dataframe.set_index('DatasetName', inplace=True)
        
    dataset_list = [dataframe[[column]].dropna() for column in dataframe.columns]
    return dataset_list


# ----------------- #

# orignal function
# def MinMax_normalize_and_merge(dictionary, scalar): # MinMax normalise all dataframes in a dictionary and return a merged dataframe
#     MinMax_dict = copy.deepcopy(dictionary) # copy dictionary
#     for dataset in MinMax_dict:
#         if 'phosphosite_ID' not in MinMax_dict[dataset].index.names:
#             MinMax_dict[dataset].set_index('phosphosite_ID', inplace=True)
#         MinMax_dict[dataset][MinMax_dict[dataset].columns] = scalar.fit_transform(MinMax_dict[dataset][MinMax_dict[dataset].columns]) # MinMax normalise each dataset
#         MinMax_dict[dataset].reset_index(inplace=True) # reset index
#     MinMax_itr = MinMax_dict.values() # convert dict values (dfs) to list
#     MinMax_merged = reduce(lambda  left,right: pd.merge(left,right,on=['phosphosite_ID'], how='outer'), MinMax_itr) # merge datasets on column
#     return MinMax_merged

def MinMax_normalize_and_merge(dictionary, scalar): # MinMax normalise all dataframes in a dictionary and return a merged dataframe
    MinMax_dict = copy.deepcopy(dictionary) # copy dictionary
    for dataset in MinMax_dict:
        if 'DatasetName' not in MinMax_dict[dataset].index.names:
            MinMax_dict[dataset].set_index('DatasetName', inplace=True)
        MinMax_dict[dataset][MinMax_dict[dataset].columns] = scalar.fit_transform(MinMax_dict[dataset][MinMax_dict[dataset].columns]) # MinMax normalise each dataset
        MinMax_dict[dataset].reset_index(inplace=True) # reset index
    MinMax_itr = MinMax_dict.values() # convert dict values (dfs) to list
    MinMax_merged = reduce(lambda  left,right: pd.merge(left,right,on=['DatasetName'], how='outer'), MinMax_itr) # merge datasets on column
    return MinMax_merged









import pandas as pd
import copy

def normalize_and_concatenate_all(dictionary, scaler):
    all_normalized_rows = []

    for dataset_name, df in dictionary.items():
        df = df.copy()

        # Clean DatasetName
        if 'DatasetName' in df.index.names:
            df.reset_index(inplace=True)
        df['DatasetName'] = df['DatasetName'].astype(str).str.strip()

        # Normalize numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Add a source tag so we can track where each row came from
        df['SourceDataset'] = dataset_name

        # Reorder columns: SourceDataset, DatasetName, then samples
        df = df[['SourceDataset', 'DatasetName'] + list(numeric_cols)]

        all_normalized_rows.append(df)

    # Concatenate all rows (stack vertically)
    merged = pd.concat(all_normalized_rows, axis=0, ignore_index=True)

    return merged
