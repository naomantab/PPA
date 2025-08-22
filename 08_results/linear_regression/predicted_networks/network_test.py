import pandas as pd

# Load files
reg_df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/08_results/linear_regression/coefficients/linear_regression_nested_cv_PI3KAKT_coefficients_protein_level_min50vals.csv")
# reg_df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/08_results/linear_regression/coefficients/linear_regression_nested_cv_PI3KAKT_coefficients_protein_level_min100vals.csv")
# reg_df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/08_results/linear_regression/coefficients/linear_regression_nested_cv_PI3KAKT_coefficients_protein_level_min150vals.csv")
# reg_df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/08_results/linear_regression/coefficients/linear_regression_nested_cv_PI3KAKT_coefficients_protein_level_min200vals.csv")
biogrid_df = pd.read_csv("C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/BIOGRID-ORGANISM-Mus_musculus-4.4.246.tab3.txt", sep="\t")

# Select columns
reg_edges = set(zip(reg_df['Feature'].str.upper(), reg_df['TargetFeature'].str.upper()))
biogrid_edges = set(zip(biogrid_df['Official Symbol Interactor A'].str.upper(), biogrid_df['Official Symbol Interactor B'].str.upper()))
biogrid_edges_rev = set((b, a) for a, b in biogrid_edges)

# Combine BioGRID as undirected
biogrid_all = biogrid_edges | biogrid_edges_rev

# Find overlaps
matches = reg_edges & biogrid_all
print("Overlapping interactions:", matches)
print("Match count:", len(matches))
