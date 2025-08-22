# Import dependencies
import time 
start_time = time.time()
import omnipath as op
import pandas as pd
import numpy as np

# # ------------------ #

if __name__ == '__main__':
    
    print("Loading human OmniPath interactions...")
    interactions = op.interactions.PostTranslational.get()
    interactions = interactions.iloc[:, [0, 1]]
    interactions.rename(columns={"source": "human_source_uniprot_id", "target": "human_target_uniprot_id"}, inplace=True)
    print(interactions)

    homologues = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/uniprot_and_ensembl_human_mouse_homologues.csv', header=0)

    print(homologues.columns)
    print("Converting human UniProt IDs to human & mouse gene names...")
    human_uniprot_to_gene_name = homologues.set_index('human_uniprot_id')['human_gene_name'].to_dict()
    print("human_uniprot_to_gene_name", human_uniprot_to_gene_name)
    human_gene_to_mouse_gene = homologues.set_index('human_gene_name')['Mouse gene name'].to_dict()

    interaction_homologues = interactions.copy()
    interaction_homologues['human_source_gene_name'] = interaction_homologues['human_source_uniprot_id'].map(human_uniprot_to_gene_name)
    interaction_homologues['human_target_gene_name'] = interaction_homologues['human_target_uniprot_id'].map(human_uniprot_to_gene_name)
    interaction_homologues['mouse_source_gene_name'] = interaction_homologues['human_source_gene_name'].map(human_gene_to_mouse_gene)
    interaction_homologues['mouse_target_gene_name'] = interaction_homologues['human_target_gene_name'].map(human_gene_to_mouse_gene)
    print(interaction_homologues)

    interaction_homologues = interaction_homologues.dropna(subset=['mouse_source_gene_name', 'mouse_target_gene_name', 'human_source_gene_name', 'human_target_gene_name'], how='any')

    print("Saving mouse homologues of OmniPath interactions...")
    interaction_homologues.to_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/omnipath_mouse_human_homologue_interactions.csv", index=False)
    print(interaction_homologues)

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')


# def clean_uniprot_id(uid):
#     if pd.isna(uid):
#         return uid
#     return uid.split('-')[0].strip().upper()

# if __name__ == '__main__':
#     start_time = time.time()
    
#     print("Loading human OmniPath interactions...")
#     interactions = op.interactions.PostTranslational.get()
#     interactions = interactions.iloc[:, [0, 1]]
#     interactions.rename(columns={"source": "human_source_uniprot_id", "target": "human_target_uniprot_id"}, inplace=True)
#     print(interactions)
    
#     # Clean UniProt IDs by removing isoform suffixes (e.g., P12345-2 -> P12345)
#     interactions['human_source_uniprot_id'] = interactions['human_source_uniprot_id'].apply(clean_uniprot_id)
#     interactions['human_target_uniprot_id'] = interactions['human_target_uniprot_id'].apply(clean_uniprot_id)
    
#     print(f"Total interactions loaded: {interactions}")
    
#     print("Loading homologues mapping...")
#     homologues = pd.read_csv(
#         'C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/uniprot_and_ensembl_human_mouse_homologues.csv',
#         header=0
#     )
    
#     print(f"Total homologues entries: {homologues}")
    
#     print("Creating mapping dictionaries...")
#     # UniProt ID -> human gene name
#     human_uniprot_to_gene_name = homologues.set_index('human_uniprot_id')['human_gene_name'].to_dict()
#     # human gene name -> mouse gene name
#     human_gene_to_mouse_gene = homologues.set_index('human_gene_name')['Mouse gene name'].to_dict()
    
#     print("Mapping human UniProt IDs to human gene names...")
#     interactions = interactions.copy()
#     interactions['human_source_gene_name'] = interactions['human_source_uniprot_id'].map(human_uniprot_to_gene_name)
#     interactions['human_target_gene_name'] = interactions['human_target_uniprot_id'].map(human_uniprot_to_gene_name)
    
#     print("Mapping human gene names to mouse gene names...")
#     interactions['mouse_source_gene_name'] = interactions['human_source_gene_name'].map(human_gene_to_mouse_gene)
#     interactions['mouse_target_gene_name'] = interactions['human_target_gene_name'].map(human_gene_to_mouse_gene)
#     print(interactions)
#     print(f"Before filtering, interactions with mouse homologues (source and target) present:")
#     # print(interactions[['mouse_source_gene_name', 'mouse_target_gene_name']].notna().all(axis=1).sum())
    
#     # Filter out interactions where either mouse homologue is missing
#     interactions_filtered = interactions.dropna(subset=['mouse_source_gene_name', 'mouse_target_gene_name'], how='any')
    
#     print(f"After filtering, interactions count: {len(interactions_filtered)}")
    
#     print("Saving mouse homologues of OmniPath interactions...")
#     interactions_filtered.to_csv(
#         "C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/omnipath_mouse_human_homologue_interactions.csv",
#         index=False
#     )
    
#     print(f"Execution time: {time.time() - start_time:.2f} seconds")
