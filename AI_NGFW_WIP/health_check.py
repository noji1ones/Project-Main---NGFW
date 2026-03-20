import pandas as pd
import numpy as np
import glob
import os

csv_folder_path = './datasets/MachineLearningCSV/MachineLearningCVE/*.csv' 

print("check data")
files = glob.glob(csv_folder_path)

if not files:
    print(f"No files found at '{csv_folder_path}'. check path.")
else:
    print(f" {len(files)} CSV files found\n")
    


    for file in files[:3]: 
        filename = os.path.basename(file)
        print(f"Analyzing: {filename} ---")
        
        try:
            
            df = pd.read_csv(file)
            
            
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            
            
            df.columns = df.columns.str.strip()
            
            
            if 'Label' in df.columns:
                unique_labels = df['Label'].unique()
                print(f"'Label' column found. values: {unique_labels}")
            else:
                print("'Label' column is missing")
                
            
            null_count = df.isnull().sum().sum()
            
            # Only check numeric columns for infinity to prevent string errors
            numeric_cols = df.select_dtypes(include=[np.number])
            inf_count = np.isinf(numeric_cols).values.sum()
            
            if null_count > 0 or inf_count > 0:
                print(f" Warning: Found {null_count:,} Blank (NaN) values and {inf_count:,} Infinity values.")
                print("  (For CIC-IDS-2017, Colab script MUST clean them).")
            else:
                print(" Data is clean.")
                
        except Exception as e:
            print(f" Error reading fil: {e}")
        print("\n")

print("Health Check complete.")