import os
import pandas as pd

# Folder containing your CSV files
folder_path = 'C:/Users/tnaom/OneDrive/Desktop/PPA/01_input_data/processed_datasets'

# Loop through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            # Read CSV with first column as index (row names)
            df = pd.read_csv(file_path, index_col=0)

            # Check all values in the index (row names)
            for index_val in df.index:
                val_str = str(index_val)
                if val_str != val_str.upper() or '.' in val_str:
                    print(f'{filename}: ❌ Found row name not fully capitalized or with dot — "{val_str}"')
                    break
            else:
                print(f'{filename}: ✅ All row names are capitalized and clean')

        except Exception as e:
            print(f'{filename}: ⚠️ Failed to read file — {e}')
