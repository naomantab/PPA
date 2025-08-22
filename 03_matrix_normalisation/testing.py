import pandas as pd

# matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', header=0)
matrix = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/RawMatrix.csv', header=0)


# # # Find columns that are not fully capitalized
not_capitalized = [col for col in matrix.columns if col != col.upper()]
capitalized = [col for col in matrix.columns if col == col.upper()]
columns_with_parentheses = [col for col in matrix.columns if '(' in col or ')' in col]
print(f"Number of columns containing parentheses: {len(columns_with_parentheses)}")

print(f"Number of columns fully capitalized: {len(capitalized)}")
print(f"Number of columns not fully capitalized: {len(not_capitalized)}")

# if capitalized:
#     print("Fully capitalized columns:", capitalized)
# if not_capitalized:
#     print("Not fully capitalized columns:", not_capitalized)