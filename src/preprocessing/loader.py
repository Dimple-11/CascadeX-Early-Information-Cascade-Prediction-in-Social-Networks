import os
import pandas as pd

def load_and_preprocess_dataset(
    raw_filepath="data/raw/InfoNet5000Q1000NEXP.txt",
    output_filepath="data/processed/network_edges.csv"
):
    """
    Loads raw NETINF dataset, performs quality audit and preprocessing,
    and saves clean edge list to data/processed/network_edges.csv.
    """
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Raw dataset not found at {raw_filepath}")
    
    # Read space-delimited text file
    df_raw = pd.read_csv(raw_filepath, sep=r'\s+')
    initial_count = len(df_raw)
    
    # 1. Audit missing values
    df_clean = df_raw.dropna().copy()
    after_na_count = len(df_clean)
    missing_removed = initial_count - after_na_count
    
    # 2. Audit duplicates across (src, dst)
    df_clean = df_clean.drop_duplicates(subset=['src', 'dst']).copy()
    after_dup_count = len(df_clean)
    duplicates_removed = after_na_count - after_dup_count
    
    # 3. Strip string whitespace
    df_clean['src'] = df_clean['src'].astype(str).str.strip()
    df_clean['dst'] = df_clean['dst'].astype(str).str.strip()
    
    # Ensure numerical types
    df_clean['number_trees'] = df_clean['number_trees'].astype(int)
    df_clean['marginal_gain'] = df_clean['marginal_gain'].astype(float)
    df_clean['median_timediff'] = df_clean['median_timediff'].astype(float)
    df_clean['average_timediff'] = df_clean['average_timediff'].astype(float)
    
    # Save to processed folder
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_clean.to_csv(output_filepath, index=False)
    
    audit_report = {
        "initial_records": initial_count,
        "missing_removed": missing_removed,
        "duplicates_removed": duplicates_removed,
        "final_records": len(df_clean),
        "unique_src_nodes": df_clean['src'].nunique(),
        "unique_dst_nodes": df_clean['dst'].nunique(),
        "total_unique_nodes": len(set(df_clean['src']).union(set(df_clean['dst'])))
    }
    return df_clean, audit_report

if __name__ == "__main__":
    df, report = load_and_preprocess_dataset()
    print("Preprocessing Audit Report:", report)
