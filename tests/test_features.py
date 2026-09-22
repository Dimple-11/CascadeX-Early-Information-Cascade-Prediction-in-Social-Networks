import os
import sys
import pandas as pd

sys.path.insert(0, '.')

def test_feature_matrix_integrity():
    print("=== TEST FEATURE MATRIX INTEGRITY ===")
    feat_path = "data/processed/final_cascade_feature_matrix.csv"
    assert os.path.exists(feat_path), "final_cascade_feature_matrix.csv missing"
    
    df_feat = pd.read_csv(feat_path)
    assert len(df_feat) == 4000, f"Expected 4000 rows (1000 cascades x 4 observation levels), got {len(df_feat)}"
    assert df_feat.isnull().sum().sum() == 0, "Feature matrix contains unexpected null values!"
    
    expected_features = [
        'cascade_id', 'observation_level', 'early_event_count', 'early_size',
        'early_comm_count', 'early_duration', 'early_growth_rate', 'early_max_out_degree',
        'early_mean_out_degree', 'early_max_betweenness', 'early_mean_betweenness',
        'early_max_pagerank', 'early_mean_pagerank', 'target_final_size',
        'early_mean_link_score', 'early_max_link_score', 'early_preferential_attachment_max'
    ]
    assert set(df_feat.columns) == set(expected_features), f"Feature columns mismatch: {df_feat.columns.tolist()}"
    print("  [PASS] Feature Matrix Integrity Verified (4000 rows, 17 columns, 0 nulls).")
    return True

if __name__ == "__main__":
    test_feature_matrix_integrity()
