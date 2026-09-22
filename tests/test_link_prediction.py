import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, '.')
from src.link_prediction.evaluation import evaluate_link_prediction

def test_link_prediction_reproducibility():
    print("=== TEST 8: LINK PREDICTION VALIDATION & CONFUSION MATRIX ===")
    cand_path = "results/link_prediction/link_prediction_candidates.csv"
    metrics_path = "results/link_prediction/link_prediction_metrics.csv"
    
    assert os.path.exists(cand_path), "link_prediction_candidates.csv missing"
    assert os.path.exists(metrics_path), "link_prediction_metrics.csv missing"
    
    df_cand = pd.read_csv(cand_path)
    df_metrics = pd.read_csv(metrics_path)
    
    pos_count = (df_cand['label'] == 1).sum()
    neg_count = (df_cand['label'] == 0).sum()
    
    assert pos_count == 1500, f"Expected 1500 positive test edges, got {pos_count}"
    assert neg_count == 1500, f"Expected 1500 negative test edges, got {neg_count}"
    
    # Recalculate metrics independently
    df_recalc = evaluate_link_prediction(candidates_csv=cand_path, output_dir="results/link_prediction")
    
    rf_row = df_recalc[df_recalc['model_name'] == 'Random Forest'].iloc[0]
    rf_reported = df_metrics[df_metrics['model_name'] == 'Random Forest'].iloc[0]
    
    assert abs(rf_row['roc_auc'] - rf_reported['roc_auc']) < 1e-4, f"ROC-AUC mismatch: {rf_row['roc_auc']} vs {rf_reported['roc_auc']}"
    assert abs(rf_row['pr_auc'] - rf_reported['pr_auc']) < 1e-4, f"PR-AUC mismatch: {rf_row['pr_auc']} vs {rf_reported['pr_auc']}"
    
    print(f"  [PASS] Link Prediction Validated:")
    print(f"    - RF ROC-AUC: {rf_row['roc_auc']:.4f} (Reported: {rf_reported['roc_auc']:.4f})")
    print(f"    - RF PR-AUC:  {rf_row['pr_auc']:.4f} (Reported: {rf_reported['pr_auc']:.4f})")
    print(f"    - RF F1-Score:{rf_row['f1_score']:.4f} (Reported: {rf_reported['f1_score']:.4f})")
    return True

if __name__ == "__main__":
    test_link_prediction_reproducibility()
