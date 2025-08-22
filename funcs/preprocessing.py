#!/bin/python

import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import re
from functools import reduce
import requests


# ----------------- #

def match_seq_to_genename(dataset, seq_column):
    '''
    Maps amino acid sequences to gene names using the loaded fasta file.
    
    args:
    =====
    dataset: <pd.Dataframe> with a column of amino acid sequences
    seq_column: <str> column name containing amino acid sequences
    
    out:
    ====
    dataset: <pd.Dataframe> with an additional column containing gene names
    '''    
    
    fasta_sequence = list(SeqIO.parse(open(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/reference_seq/uniprotkb_proteome_UP000000589.fasta"), "fasta"))
    
    gene_dict = {}
    
    # iterate over rows in seq_column
    for i in dataset[seq_column]:
        i_str = str(i)
        for seq_record in fasta_sequence:
            matches = re.findall(i_str, str(seq_record.seq))
            if matches:
                gene_name_match = seq_record.description.split(' ')[1].split(' ')[0]
                # gene_name_match = re.search("GN=(\w+)", seq_record.description)
                if gene_name_match:
                    gene_dict[i] = gene_name_match
    
    # map sequences to gene names           
    dataset['GeneName'] = dataset[seq_column].map(gene_dict) 
    print('Amino acid sequences matched to gene names.')
    
    

# ----------------- #

def find_position_in_gene(dataset, seq_column):
    positions_dict = {}
    
    fasta_sequence = list(SeqIO.parse(open(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/reference_seq/uniprotkb_proteome_UP000000589.fasta"), 'fasta'))

    # iterate over rows in the Sequence Window column of GG2009
    for i in dataset['Sequence']:
        # iterate over entries (gene ids and sequences) in the fasta_sequence object
        for seq_record in fasta_sequence:
            # find matches between the sequence in the dataset and the sequence in the fasta file
            matches = re.findall(i, str(seq_record.seq))
            # if matches are found, print the gene id, the sequence, and the length of the sequence
            if matches:
                # find the position of i within the seq_record.seq
                start_position = str(seq_record.seq).find(i)
                #print("Starting position of i within seq_record.seq:", start_position)
                positions_dict[i] = start_position # add start position to dictionary

    dataset['StartPosition'] = dataset[seq_column].map(positions_dict)
    
    return dataset
    
    
# ----------------- #

def get_position_and_gene(dataset, seq_column, position_column):
    gene_dict = {}
    residues_dict = {}

    fasta_sequence = list(SeqIO.parse(open(f"C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/uniprotkb_proteome_UP000000589.fasta"), 'fasta'))
        
    # get the gene name and amino acid from the fasta file
    for index, row in dataset.iterrows():  # iterate over rows in the DataFrame
        sequence = row[seq_column]
        position = int(row[position_column]) - 1  # Python uses 0-based indexing
        for seq_record in fasta_sequence:
            if sequence in str(seq_record.seq):  # if sequence matches
                gene_name_match = seq_record.description.split(' ')[1].split(' ')[0]
                # gene_name_match = re.search("GN=(\w+)", seq_record.description)
                if gene_name_match:
                    gene_dict[sequence] = gene_name_match
                    # Translate the sequence into amino acids
                    protein_sequence = seq_record.seq
                    # Find the amino acid at the given position
                    residue = protein_sequence[position]
                    residues_dict[sequence] = residue

    dataset['GeneName'] = dataset['Sequence'].map(gene_dict)  # map sequences to gene names
    dataset['Residue'] = dataset['Sequence'].map(residues_dict)  # map sequences to amino acids
    
    

# ----------------- #
    
def create_phos_ID(dataset):
    '''
    Concatenates GeneName and Phosphosite columns.
    
    args:
    =====
    dataset: <pd.Dataframe> with columns 'GeneName' and 'Phosphosite'
    
    out:
    ====
    dataset: <pd.Dataframe> with 'phosphosite_ID' column and 'GeneName' + 'Phosphosite' columns dropped
    '''
    dataset.loc[:, 'phosphosite_ID'] = dataset['GeneName'].astype(str) + '_' + dataset['Phosphosite'].astype(str)
    dataset = dataset.drop(columns=['Phosphosite', 'GeneName'])
    print('Phosphosite IDs created.')
    return dataset



# ----------------- #

def log2_transform(dataset):
    '''
    Log2 transform a dataset.
    
    args:
    =====
    dataset: <pd.Dataframe>
    
    out:
    ====
    dataset: <pd.Dataframe> with log2 transformed values

    '''
    cols_to_transform = dataset.columns.drop('phosphosite_ID')
    dataset[cols_to_transform] = dataset[cols_to_transform].astype(float).apply(np.log2)
    print('Data has been log2 transformed.')
    return dataset



# ----------------- #

def rename_col_by_index(dataframe, index_to_name_mapping):
    '''
    Rename columns by index
    
    args:
    =====
    dataframe: <pd.Dataframe>
    index_to_name_mapping: <dict> mapping of index (key) to column name (str value)
    
    out:
    ====
    dataframe: <pd.Dataframe> with renamed columns
    '''
    dataframe.columns = [index_to_name_mapping.get(i, col) for i, col in enumerate(dataframe.columns)]
    return dataframe



# ----------------- #

def get_ens_dict(file_path): 
    '''
    Create dictionary of gene names and gene IDs from Ensembl gtf file
    '''
    with open(file_path) as f:
        gtf = [x for x in f if not x.startswith('#') and 'gene_id "' in x and 'gene_name "' in x]
    if len(gtf) == 0:
        print('you need to change gene_id ' and 'gene_name "')
    gtf = dict(set(map(lambda x: (x.split('gene_id "')[1].split('"')[0], x.split('gene_name "')[1].split('"')[0]), gtf)))
    print('number of ID pairs retrieved from database: ', len(gtf))
    return gtf

    # explanation of function at: https://www.youtube.com/watch?v=ve_BoDw1s7I



# ----------------- #

def create_dict_per_dataset(file_names):
    files_dict = {}
    for file in file_names:
        files_dict[file] = pd.read_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{file}.csv', header=0)
        print(f"{file} added to dict")
    print('Datasets have been loaded into dictionary.')
    return files_dict

# def create_dict_per_dataset(file_names):
#     files_dict = {}
#     for file in file_names:
#         # Read the CSV
#         df = pd.read_csv(f'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets/{file}.csv', header=0)

#         # Rename columns, skipping the first column (index 0)
#         new_columns = [df.columns[0]] + [f"{file}_{col}" for col in df.columns[1:]]
#         df.columns = new_columns

#         files_dict[file] = df
#         print(f"{file} added to dict with renamed columns.")

#     print('Datasets have been loaded into dictionary.')
#     return files_dict

# ----------------- #

def create_matrix_header(files_dict):
    files_merged = reduce(lambda left,right:pd.merge(left,right, on=['phosphosite_ID'], how='outer'), files_dict.values())
    print('Datasets have been merged on phosphosite_ID column.')
    
    phos_id = files_merged['phosphosite_ID'].unique()
    matrix_cols = pd.DataFrame(columns = phos_id) 
    matrix_cols.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/raw-matrix-header.csv', index = False)
    print('Unique phosphosite_IDs saved.')
    return matrix_cols

# def create_matrix_header(files_dict):
#     files = list(files_dict.values())
#     files_merged = files[0]

#     for df in files[1:]:
#         files_merged = pd.merge(files_merged, df, on=['phosphosite_ID'], how='outer')

#     print('Datasets have been merged on phosphosite_ID column.')
    
#     phos_id = files_merged['phosphosite_ID'].unique()
#     matrix_cols = pd.DataFrame(columns=phos_id)
#     matrix_cols.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/raw-matrix-header.csv', index=False)
#     print('Unique phosphosite_IDs saved.')
    
#     return matrix_cols

# ----------------- #

# def add_rows_to_matrix(matrix, files_datasets, files_dict):
#     new_rows = []
    
#     for dataset_key, dataset_names in files_datasets:
#         if 'DictKey' in matrix.columns and dataset_key in matrix['DictKey'].values:
#             print(f"Dataset key {dataset_key} is already in the matrix.")
#             continue
    
#         matrix.columns = pd.Index(matrix.columns).drop_duplicates()
#         # Iterate over the columns of the dataset
#         for i in range (1, files_dict[dataset_key].shape[1]):
#             if i-1 >= len(dataset_names):
#                 print(f"Error: dataset_names doesn't have an element at index {i-1}")
#                 return matrix
#             # Convert the column to a dictionary
#             data_dict = files_dict[dataset_key].set_index(files_dict[dataset_key].columns[0]).to_dict()[files_dict[dataset_key].columns[i]]
#             # Create a new row dataframe from the dictionary
#             new_row = pd.DataFrame(data_dict, index = [0])
#             new_row = new_row.loc[:, ~new_row.columns.duplicated()]
#             # Reindex the new row to match the columns of the phosphoproteomics_matrix
#             new_row = new_row.reindex(columns = matrix.columns)
#             # Add the dataset name as a column to the new row
#             if 'DatasetName' in matrix.columns:
#                 new_row['DatasetName'] = dataset_names[i-1]
#             else:
#                 new_row.insert(0, "DatasetName", dataset_names[i-1])
#             new_row['DictKey'] = dataset_key
#             new_rows.append(new_row)
#             # Concatenate the new row to the phosphoproteomics_matrix
#             # matrix = pd.concat([matrix, new_row], ignore_index = True)
#             print(f"{new_row['DictKey']} added to matrix")
    
#     if new_rows:
#         matrix = pd.concat([matrix] + new_rows, ignore_index=True)
#     matrix.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/intermediary-raw-matrix.csv', index = False)
#     print('Intermediary raw matrix saved.')
#     return matrix

def add_rows_to_matrix(matrix, files_datasets, files_dict):
    new_rows = []

    for dataset_key, dataset_names in files_datasets:
        print(f"Processing: {dataset_key}")

        dataset = files_dict.get(dataset_key)
        if dataset is None:
            print(f"Warning: {dataset_key} not found in files_dict")
            continue

        # Assume first column is phosphosite_ID
        phosphosite_col = dataset.columns[0]
        dataset = dataset.set_index(phosphosite_col)

        for i in range(1, dataset.shape[1] + 1):  # loop over all data columns
            try:
                col_name = dataset.columns[i - 1]  # MS2024_4hr_intensity
                series = dataset[col_name]

                # Convert to row: phosphosite_IDs become column names
                row_dict = series.to_dict()

                # Add feature name
                row_dict['DatasetName'] = col_name

                new_rows.append(row_dict)

            except Exception as e:
                print(f"Failed on {dataset_key} col {i}: {e}")
                continue

    # Create dataframe
    matrix = pd.DataFrame(new_rows)

    # Move DatasetName to first column
    cols = matrix.columns.tolist()
    cols = ['DatasetName'] + [c for c in cols if c != 'DatasetName']
    matrix = matrix[cols]

    print(f"Rows added: {len(new_rows)}")
    return matrix




# ----------------- #

def find_outliers_IQR(df): # identify outliers from a dataframe using the Interquartile Range (IQR)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    return outliers



# ----------------- #

def drop_outliers_IQR(df): # drop outliers from a dataframe using the Interquartile Range (IQR)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    not_outliers = df[~((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    return not_outliers



# ----------------- #

def clean_phosID_col(data):
    data = data[~data.phosphosite_ID.str.contains('nan', case = False)]
    data = data[~data.phosphosite_ID.str.contains(';', case = False)] # remove rows containing ';' in phosphosite_ID column
    data = data[~data.phosphosite_ID.str.contains('-', case = False)] # remove rows containing '-' in phosphosite_ID column
    
    # check whether there are any phosphosites with multiple measurements
    data_grouped = data.groupby(by = 'phosphosite_ID')
    if len(data) != len(data_grouped):
        data = data_grouped.mean()
        data.reset_index(inplace=True) # reset index
        print('Phosphosites with multiple measurements have been averaged')
    else:
        print('There are no phosphosites with multiple measurements')
        
    print(data)
        
    data = data.replace([np.inf, -np.inf], np.nan)
        
    if data.columns[0] != 'phosphosite_ID':
        phosphosite_ID = data.pop('phosphosite_ID')
        data.insert(0, 'phosphosite_ID', phosphosite_ID)
    return data



# ----------------- #

def map_uniprot_to_gene(data, id_column='UniProtID', gene_column='GeneName'):
    """
    Maps UniProt IDs to gene names using the UniProt REST API.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing UniProt IDs.
    - id_column (str): Column name with UniProt IDs.
    - gene_column (str): Column name to store gene names (default: 'GeneName').

    Returns:
    - pd.DataFrame: Updated DataFrame with gene names.
    """
    def fetch_gene_mapping(uniprot_ids):
        url = "https://rest.uniprot.org/uniprotkb/search"
        query = ' OR '.join([f'accession:{uid}' for uid in uniprot_ids])
        params = {
            'query': query,
            'fields': 'accession,gene_primary',
            'format': 'tsv',
            'size': len(uniprot_ids)
        }
        response = requests.get(url, params=params)
        mapping = {}
        if response.status_code == 200:
            lines = response.text.strip().split('\n')[1:]  # skip header
            for line in lines:
                parts = line.split('\t')
                if len(parts) == 2:
                    acc, gene = parts
                    mapping[acc] = gene
                elif len(parts) == 1:
                    acc = parts[0]
                    mapping[acc] = "Gene name not found"
        else:
            print(f"API request failed with status code {response.status_code}")
        return mapping

    # Get unique UniProt IDs
    unique_ids = data[id_column].dropna().unique().tolist()

    # Fetch mappings in batches
    batch_size = 500
    all_mappings = {}
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]
        batch_map = fetch_gene_mapping(batch)
        all_mappings.update(batch_map)

    # Apply the mapping to the DataFrame
    data[gene_column] = data[id_column].map(all_mappings).fillna("Gene name not found")
    
    return data

# ----------------- #

def get_csv_file_names_as_tuple(folder_path):
    """
    Gives all the files in  a folder
    """
    file_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            name_without_ext = os.path.splitext(filename)[0]
            file_names.append(name_without_ext)
    return tuple(file_names)

# ----------------- #