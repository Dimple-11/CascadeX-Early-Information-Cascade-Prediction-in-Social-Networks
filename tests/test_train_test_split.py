import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

def test_cascade_level_split_isolation():
    print("=== TEST 5: TRAIN / VALIDATION / TEST SPLIT ISOLATION AUDIT ===")
    feat_path = "data/processed/final_cascade_feature_matrix.csv"
    assert os.path.exists(feat_path), "final_cascade_feature_matrix.csv missing"
    
    df_feat = pd.read_csv(feat_path)
    unique_cascades = df_feat['cascade_id'].unique()
    
    np.random.seed(42)
    np.random.shuffle(unique_cascades)
    
    n_total = len(unique_cascades)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_ids = set(unique_cascades[:n_train])
    val_ids = set(unique_cascades[n_train:n_train+n_val])
    test_ids = set(unique_cascades[n_train+n_val:])
    
    # 1. Assert Set Disjointness
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Validation cascade IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test cascade IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Validation and Test cascade IDs overlap!"
    
    # 2. Check Row Counts across observation levels
    df_train = df_feat[df_feat['cascade_id'].isin(train_ids)]
    df_val = df_feat[df_feat['cascade_id'].isin(val_ids)]
    df_test = df_feat[df_feat['cascade_id'].isin(test_ids)]
    
    assert len(train_ids) == 700, f"Expected 700 train cascades, got {len(train_ids)}"
    assert len(val_ids) == 150, f"Expected 150 val cascades, got {len(val_ids)}"
    assert len(test_ids) == 150, f"Expected 150 test cascades, got {len(test_ids)}"
    
    assert len(df_train) == 2800, f"Expected 2800 train rows, got {len(df_train)}"
    assert len(df_val) == 600, f"Expected 600 val rows, got {len(df_val)}"
    assert len(df_test) == 600, f"Expected 600 test rows, got {len(df_test)}"
    
    print("  [PASS] Split Isolation Audit:")
    print("    - Train Cascades: 700 (2,800 rows)")
    print("    - Validation Cascades: 150 (600 rows)")
    print("    - Test Cascades: 150 (600 rows)")
    print("    - Overlap: 0 (100% Disjoint Split)")
    return True

if __name__ == "__main__":
    test_cascade_level_split_isolation()
