import pandas as pd
import numpy as np

def count_effective_zeros(filepath, column_candidates, threshold=1e-6):
    """
    Count values in column that are effectively zero, including NaNs.
    
    Args:
        filepath (str): CSV file path.
        column_candidates (list): Possible column names to check.
        threshold (float): Values with abs(value) < threshold count as zero.
        
    Returns:
        int: Number of effective zeros.
    """
    df = pd.read_csv(filepath)
    
    # Find the correct column
    col = None
    for c in column_candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"No valid column found in {filepath}")
    
    # Convert to numeric, coerce errors to NaN
    numeric = pd.to_numeric(df[col], errors='coerce')
    
    # Count zeros or near-zeros
    zero_count = ((numeric.abs() < threshold) | numeric.isna()).sum()
    return zero_count

# Example usage
lr_file = "C:/Users/tnaom/OneDrive/Desktop/PPA/08_results/linear_regression/coefficients/linear_regression_cv_coefficients_min50vals.csv"
xgb_file = "C:/Users/tnaom/OneDrive/Desktop/PPA/08_results/xgboost/results_files/xgboost_nested_cv_master_results_file_min50vals.csv"

print("LR effective zeros:", count_effective_zeros(lr_file, ['MeanR2', 'mean_r2']))
print("XGB effective zeros:", count_effective_zeros(xgb_file, ['mean_r2', 'MeanR2']))

