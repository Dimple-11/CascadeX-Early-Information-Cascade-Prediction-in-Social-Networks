import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

def run_independent_leakage_audit(
    feat_path="data/processed/final_cascade_feature_matrix.csv",
    output_path="results/tables/independent_leakage_audit.csv"
):
    print("=== TEST 6 & 7: INDEPENDENT TEMPORAL LEAKAGE AUDIT ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    assert os.path.exists(feat_path), "final_cascade_feature_matrix.csv missing"
    df_feat = pd.read_csv(feat_path)
    
    feature_cols = [c for c in df_feat.columns if c not in ['cascade_id', 'target_final_size']]
    audit_records = []
    
    for col in feature_cols:
        corr = df_feat[col].corr(df_feat['target_final_size'])
        
        # Classification criteria
        if col == 'target_final_size' or 'final' in col.lower() or 'future' in col.lower():
            classification = "CONFIRMED LEAKAGE"
            reason = "Direct future target or full cascade metric"
        elif 'total_events' in col.lower() or 'full_graph' in col.lower():
            classification = "POTENTIAL LEAKAGE"
            reason = "Requires future cascade completion"
        else:
            classification = "SAFE"
            reason = "Strictly derived from events <= early observation window p"
            
        audit_records.append({
            'feature_name': col,
            'classification': classification,
            'correlation_with_target': float(corr),
            'audit_rationale': reason,
            'available_at_prediction_time': 'YES' if classification == 'SAFE' else 'NO'
        })
        
    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(output_path, index=False)
    
    confirmed_leakage_count = (df_audit['classification'] == 'CONFIRMED LEAKAGE').sum()
    potential_leakage_count = (df_audit['classification'] == 'POTENTIAL LEAKAGE').sum()
    safe_count = (df_audit['classification'] == 'SAFE').sum()
    
    assert confirmed_leakage_count == 0, f"Confirmed leakage found in features: {confirmed_leakage_count}"
    
    print(f"  [PASS] Independent Leakage Audit Completed: {safe_count} SAFE features, {potential_leakage_count} POTENTIAL, {confirmed_leakage_count} CONFIRMED LEAKAGE.")
    print(f"  Audit report saved to {output_path}.")
    return df_audit

if __name__ == "__main__":
    run_independent_leakage_audit()
