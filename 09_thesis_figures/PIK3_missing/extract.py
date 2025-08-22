import pandas as pd

# # Load the data
# data = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/06_models/xgboost/master_shaps_files/xgboost_master_shap_file_cluster_level_min50vals_with_shapxr2.csv')

# # List of proteins to filter
prots = (
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
    "PIK3R1", "PIK3R2",
    "PIK3C2A", "PIK3C2B", "PIK3C2G",   # Class II PI3Ks
    "PI3KC2A", "PI3KC2B", "PI3KC2G",    # Class II PI3Ks (alternative names)
)

# # Filter rows based on the presence of proteins in the TargetFeature column
# filtered = data[data['TargetFeature'].str.contains('|'.join(prots), case=False, na=False)]

# # Clean 'TargetFeature' to only keep protein names
# # First remove post-translational modifications like '_S(261)', then split the string at spaces to keep only the first name
# filtered['TargetFeature'] = filtered['TargetFeature'].str.replace(r'(_S\(\d+\))', '', regex=True)  # Remove _S(261)
# filtered['TargetFeature'] = filtered['TargetFeature'].str.split().str[0]  # Keep only the first word (protein name)

# # Display the cleaned 'TargetFeature' and other relevant columns
# print(filtered[['TargetFeature', 'SHAPValue', 'SHAP*R2']])


###############################################################################

# data2 = pd.read_csv('C:/Users/tnaom/Downloads/clust_matrix.csv')
# # Print columns whose names start with 'PIK3'
# matching_cols = [col.split("_")[0] for col in data2.columns if col.startswith("PIK3")]

# # Remove duplicates and print the result
# matching_cols = list(set(matching_cols))  # Remove duplicates
# print(matching_cols)

###############################################################################

# data3 = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/05_feature_selection/interim_data/top_500_fisher_scores_min50vals.csv')

# # # Print the dataframe ordered by FisherScore in descending order
# # ordered_data = data3.sort_values(by='FisherScore', ascending=False)

# # # Print the ordered dataframe
# # print(ordered_data)

# prots = (
#     "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
#     "PIK3R1", "PIK3R2",
#     "PIK3C2A", "PIK3C2B", "PIK3C2G",   # Class II PI3Ks
#     "PI3KC2A", "PI3KC2B", "PI3KC2G",    # Class II PI3Ks (alternative names)
# )

# filtered = data3[data3['Feature'].str.contains('|'.join(prots), case=False, na=False)]
# print(filtered.sort_values(by='FisherScore', ascending=False))


#########################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # For log transformation

# Load your data
data3 = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/05_feature_selection/interim_data/top_500_fisher_scores_min50vals.csv')

# Convert FisherScore to numeric (if it's not already)
data3['FisherScore'] = pd.to_numeric(data3['FisherScore'], errors='coerce')

# Apply log transformation (avoid log(0) by adding a small value)
data3['LogFisherScore'] = np.log10(data3['FisherScore'] + 1)  # Adding 1 to avoid log(0)

# Filter data for rows that contain the proteins of interest in the TargetFeature column
prots = (
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
    "PIK3R1", "PIK3R2",
    "PIK3C2A", "PIK3C2B", "PIK3C2G",   # Class II PI3Ks
    "PI3KC2A", "PI3KC2B", "PI3KC2G",    # Class II PI3Ks (alternative names)
)

# Filter using TargetFeature column
filtered = data3[data3['TargetFeature'].str.contains('|'.join(prots), case=False, na=False)]

# Print the filtered dataframe to ensure it worked
print(f"Filtered data contains {len(filtered)} entries")
print(filtered[['TargetFeature', 'FisherScore', 'LogFisherScore']].head())

# Calculate the top 10% cutoff for LogFisherScore
top_10_percent_cutoff = np.percentile(data3['LogFisherScore'], 90)
print(f"Top 10% Log Fisher Score cutoff: {top_10_percent_cutoff}")

# Create the histograms with separate plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: All Fisher Scores (Log-transformed) in the left subplot
sns.histplot(data3['LogFisherScore'], bins=75, color='teal', label='All Fisher Scores (Log)', ax=axes[0], alpha=0.5)
axes[0].set_title('Distribution of Log-transformed Fisher Scores (All)', fontsize=14)
axes[0].set_xlabel('Log10(Fisher Score)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend()

# Add the vertical line for the top 10% cutoff
axes[0].axvline(top_10_percent_cutoff, color='black', linestyle='--', label=f'Top 10% Cutoff')
axes[0].legend()

# Plot 2: Highlighted Proteins in the right subplot
sns.histplot(filtered['LogFisherScore'], bins=75, color='tomato', label='Selected PIK3 Proteins (Log)', ax=axes[1], alpha=1.0)
axes[1].set_title('Log-transformed Fisher Scores for Selected PIK3 Proteins', fontsize=14)
axes[1].set_xlabel('Log10(Fisher Score)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend()

# Add the vertical line for the top 10% cutoff on the highlighted proteins plot
axes[1].axvline(top_10_percent_cutoff, color='black', linestyle='--', label=f'Top 10% Cutoff')
axes[1].legend()

# Adjust layout
plt.tight_layout()

# Display the plot
# plt.show()

# Save the plot as a PNG file
plt.savefig("C:/Users/tnaom/OneDrive/Desktop/PPA/09_thesis_figures/PIK3_missing/fisher_scores_separate_plots_with_cutoff.png", dpi=300, bbox_inches='tight')

# Optionally print the sorted dataframe for reference
ordered_data = data3.sort_values(by='FisherScore', ascending=False)
print(ordered_data[['TargetFeature', 'FisherScore', 'LogFisherScore']].head(20))  # Show top 20 Fisher Scores




