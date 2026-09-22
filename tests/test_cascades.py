import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

def test_cascade_extraction_and_sampling():
    print("=== TEST 3: RANDOM CASCADE SAMPLING & VALIDATION ===")
    cascade_path = "data/processed/cascade_dataset.csv"
    feat_path = "data/processed/final_cascade_feature_matrix.csv"
    
    assert os.path.exists(cascade_path), "cascade_dataset.csv missing"
    assert os.path.exists(feat_path), "final_cascade_feature_matrix.csv missing"
    
    df_cascades = pd.read_csv(cascade_path)
    df_feat = pd.read_csv(feat_path)
    
    assert len(df_cascades) == 1000, f"Expected 1000 cascades, got {len(df_cascades)}"
    
    # Sample 10 random cascades
    np.random.seed(42)
    sample_ids = np.random.choice(df_cascades['cascade_id'], size=10, replace=False)
    
    inconsistencies = []
    
    for cid in sample_ids:
        c_row = df_cascades[df_cascades['cascade_id'] == cid].iloc[0]
        final_size_dataset = c_row['cascade_size']
        
        # Check corresponding rows in feature matrix across observation levels
        f_rows = df_feat[df_feat['cascade_id'] == cid]
        assert len(f_rows) == 4, f"Cascade {cid} expected 4 observation rows, got {len(f_rows)}"
        
        for _, f_row in f_rows.iterrows():
            p = f_row['observation_level']
            early_size = f_row['early_size']
            target_size = f_row['target_final_size']
            
            if target_size != final_size_dataset:
                inconsistencies.append(f"Cascade {cid} target size mismatch: {target_size} vs {final_size_dataset}")
            if early_size > target_size:
                inconsistencies.append(f"Cascade {cid} at p={p} has early_size ({early_size}) > target_size ({target_size})")
                
    assert len(inconsistencies) == 0, f"Cascade inconsistencies found: {inconsistencies}"
    
    print(f"  [PASS] Random 10 Cascades Sampled & Verified: 0 inconsistencies found across 40 observation rows.")
    return True

if __name__ == "__main__":
    test_cascade_extraction_and_sampling()
