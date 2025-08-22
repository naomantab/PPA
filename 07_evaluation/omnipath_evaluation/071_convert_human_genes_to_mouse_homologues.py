# Import dependencies
import time
start_time = time.time()
import pandas as pd
import numpy as np

# ------------------ #

if __name__ == '__main__':
    
    print("Loading mouse homologues of human genes...")
    human_mouse_homologues = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/ensembl_human_mouse_homologues.csv")
    
    # Remove rows where Mouse gene stable ID is empty or NaN
    # human_mouse_homologues = human_mouse_homologues[human_mouse_homologues['Mouse gene stable ID'].notna() & (human_mouse_homologues['Mouse gene stable ID'] != '')]

    human_mouse_homologues.rename(columns={"Gene stable ID": "human_ens_code"}, inplace=True)
    human_mouse_homologues.rename(columns={"Gene name": "human_gene_name"}, inplace=True)
    human_mouse_homologues.rename(columns={"Mouse gene stable ID": "mouse_systematic_name", "%id. query gene identical to target Mouse gene": "percentage_similarity"}, inplace=True)
    human_mouse_homologues.rename(columns={"Mouse orthology confidence [0 low, 1 high]": "confidence0low1high", "Mouse homology type": "homology_type"}, inplace=True)
    print(human_mouse_homologues)

    print("Retaining only rows with highest percentage similarity...")
    idx = human_mouse_homologues.groupby(['Mouse gene name'])['percentage_similarity'].idxmax()
    filtered_homologues = human_mouse_homologues.loc[idx]
    filtered_homologues = filtered_homologues[filtered_homologues['confidence0low1high'] == 1]
    print(filtered_homologues)

    print("Matching mouse gene names to UniProt IDs...")
    mouse_uniprot_db = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/uniprot_mouse_idmapping.dat", sep='\t', header=0)
    print("mouse_uniprot_db columns:", mouse_uniprot_db.columns.tolist())
    filtered_homologues['mouse_uniprot_id'] = filtered_homologues['Mouse gene name'].apply(
        lambda x: (
        mouse_uniprot_db.loc[
            (mouse_uniprot_db['Gene Names'] == x), 'Entry'
        ].iloc[0]
        if pd.notna(x) and not mouse_uniprot_db[
            (mouse_uniprot_db['Gene Names'] == x)
        ].empty else np.nan))
    print(filtered_homologues)

    final_homologues = filtered_homologues.dropna(subset=['mouse_uniprot_id'])
    # final_homologues = final_homologues.iloc[:, [0, 1, 2, 3, 7]]

    print("Matching human gene names to human UniProtIDs...")
    human_uniprot_db = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/uniprot_human_idmapping.dat", sep='\t', header=0)
    final_homologues['human_uniprot_id'] = final_homologues['human_gene_name'].apply(
        lambda x: (
        human_uniprot_db.loc[
            (human_uniprot_db['Gene Names'] == x), 'Entry'
        ].iloc[0]
        if pd.notna(x) and not human_uniprot_db[
            (human_uniprot_db['Gene Names'] == x)
        ].empty else np.nan))
    final_homologues.to_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/uniprot_and_ensembl_human_mouse_homologues.csv', index=False)
    print(final_homologues)

    print(f'Execution time: {time.time() - start_time:.2f} seconds, {(time.time() - start_time)/60:.2f} minutes, {(time.time() - start_time)/3600:.2f} hours.')
