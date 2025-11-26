"""
Data Quality Check: Identify Duplicate Product Names
"""
import pandas as pd
import re

def check_duplicate_products(csv_path):
    """
    Check for duplicate product names in the dataset
    """
    df = pd.read_csv(csv_path, nrows=1)
    columns = df.columns.tolist()[1:]  # Skip 'Date' column

    # Find products with .1, .2, etc. suffix (pandas auto-generated)
    duplicates = {}

    for col in columns:
        # Remove .1, .2, etc. suffix
        base_name = re.sub(r'\.\d+$', '', col)

        if base_name != col:  # Has a numeric suffix
            if base_name not in duplicates:
                duplicates[base_name] = []
            duplicates[base_name].append(col)

    # Check for exact duplicates in original data
    original_names = [re.sub(r'\.\d+$', '', col) for col in columns]
    name_counts = pd.Series(original_names).value_counts()

    print("="*80)
    print("DATA QUALITY REPORT: Duplicate Product Detection")
    print("="*80)

    if len(duplicates) == 0:
        print("\nNo duplicate products found!")
    else:
        print(f"\nWARNING: Found {len(duplicates)} product(s) with duplicates:\n")

        for base_name, variants in duplicates.items():
            count = name_counts[base_name]
            print(f"Product: {base_name}")
            print(f"  Appears {count} times in the data")
            print(f"  Pandas renamed to: {base_name}, {', '.join(variants)}")

            # Compare sales volumes
            df_full = pd.read_csv(csv_path)
            print(f"\n  Sales Volume Comparison:")
            all_versions = [base_name] + variants
            for version in all_versions:
                if version in df_full.columns:
                    total = df_full[version].sum()
                    mean = df_full[version].mean()
                    print(f"    {version:20s}: Total = {total:>12,.0f}, Mean = {mean:>8,.2f}")

            print(f"\n  Recommendation:")
            print(f"    1. Verify with business if these are truly different products")
            print(f"    2. If duplicates, merge: df['{base_name}_merged'] = df['{base_name}'] + df['{variants[0]}']")
            print(f"    3. If different, rename for clarity (e.g., '{base_name}_variant1', '{base_name}_variant2')")
            print()

    print("="*80)

    return duplicates


if __name__ == "__main__":
    sales_path = "../data/SupplyGraph/Temporal Data/Unit/Sales Order.csv"
    print("\nChecking Sales Order data...\n")
    check_duplicate_products(sales_path)
