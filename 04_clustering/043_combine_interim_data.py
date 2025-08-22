import pandas as pd
import glob
import os
from functools import reduce

# Define input and output paths
input_dir = "/data/home/bt23917/PPA/04_clustering/interim_data"
output_file = "/data/home/bt23917/PPA/04_clustering/results/combined_data.csv"

# Find all CSV files
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    if "DatasetName" in df.columns:
        dfs.append(df)
        print(f"✅ Included: {os.path.basename(file)}")
    else:
        print(f"⚠️ Skipped (missing 'DatasetName'): {os.path.basename(file)}")

# Merge all DataFrames on 'DatasetName'
if dfs:
    combined_df = reduce(lambda left, right: pd.merge(left, right, on='DatasetName', how='outer'), dfs)
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Successfully merged {len(dfs)} files.")
    print(f"📁 Output saved to: {output_file}")
else:
    print("\n❌ No valid CSVs found to merge.")
