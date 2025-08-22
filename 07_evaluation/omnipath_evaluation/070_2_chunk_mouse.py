import pandas as pd

# Load your list of IDs (e.g., from a column in a CSV)
df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/ensembl_human_mouse_homologues.csv")  # or change to .txt if needed
ids = df['Mouse gene stable ID'].unique()

# Split into chunks
chunk_size = 60000
for i in range(0, len(ids), chunk_size):
    chunk = ids[i:i+chunk_size]
    pd.Series(chunk).to_csv(f"C:/Users/tnaom/OneDrive/Desktop/PPA/07_evaluation/omnipath_evaluation/uniprot_ids_chunk_{i//chunk_size + 1}_mouse.txt", index=False, header=False)
