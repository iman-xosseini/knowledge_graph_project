
import pandas as pd
import os

os.chdir(r"c:\Users\Iman\OneDrive\Desktop\knowledge_graph_project\notebooks")

files = [
    "../data/SupplyGraph/Edges/Edges (Plant).csv",
    "../data/SupplyGraph/Nodes/Nodes.csv"
]

for f in files:
    print(f"Checking {f}...")
    try:
        df = pd.read_csv(f)
        print(df.head())
        print(df.dtypes)
    except Exception as e:
        print(f"Error: {e}")
