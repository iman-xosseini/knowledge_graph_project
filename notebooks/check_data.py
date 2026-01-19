
import pandas as pd
import os
import numpy as np

# Set working directory to notebooks loc
os.chdir(r"c:\Users\Iman\OneDrive\Desktop\knowledge_graph_project\notebooks")

files = [
    "../data/SupplyGraph/Temporal Data/Unit/Delivery To distributor.csv",
    "../data/SupplyGraph/Temporal Data/Unit/Factory Issue.csv",
    "../data/SupplyGraph/Temporal Data/Unit/Production.csv",
    "../data/SupplyGraph/Temporal Data/Unit/Sales Order.csv"
]

for f in files:
    print(f"Checking {f}...")
    try:
        df = pd.read_csv(f)
        # Check all columns except Date (if exists) or the first column if it's product name
        # Based on melt: Date is id_var. Rest are products.
        # So columns other than 'Date' should be numeric values.
        
        cols_to_check = [c for c in df.columns if c != 'Date']
        
        found_issues = False
        for col in cols_to_check:
            # Try converting to numeric
            try:
                pd.to_numeric(df[col])
            except ValueError:
                print(f"  Column '{col}' has non-numeric values.")
                # Show unique non-numeric values
                non_num = df[col][pd.to_numeric(df[col], errors='coerce').isna()]
                print(f"    Values: {non_num.unique()[:10]}")
                found_issues = True
        
        if not found_issues:
            print("  All value columns contain valid numbers.")
            
    except Exception as e:
        print(f"  Error reading file: {e}")
