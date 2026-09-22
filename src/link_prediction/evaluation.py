import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

sys.path.insert(0, '.')

def evaluate_link_prediction(
    candidates_csv="results/link_prediction/link_prediction_candidates.csv",
    output_dir="results/link_prediction",
    fig_dir="results/figures"
):
    """
    Trains ML models on link prediction structural features and evaluates ROC-AUC, PR-AUC, F1 score.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    df_link = pd.read_csv(candidates_csv)
    
    feature_cols = ['common_neighbors', 'jaccard_coeff', 'adamic_adar', 'preferential_attachment', 'resource_allocation', 'out_deg_u', 'in_deg_v']
    X = df_link[feature_cols]
    y = df_link['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    
    results = []
    plot_data = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        results.append({
            'model_name': name,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        plot_data[name] = {
            'fpr': fpr, 'tpr': tpr,
            'precision_curve': p_curve, 'recall_curve': r_curve,
            'roc_auc': roc_auc, 'pr_auc': pr_auc
        }
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "link_prediction_metrics.csv"), index=False)
    
    # Save ROC & PR Curves Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    for name, d in plot_data.items():
        axes[0].plot(d['fpr'], d['tpr'], label=f"{name} (AUC = {d['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title("Link Prediction ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    
    # PR Curve
    for name, d in plot_data.items():
        axes[1].plot(d['recall_curve'], d['precision_curve'], label=f"{name} (PR-AUC = {d['pr_auc']:.3f})")
    axes[1].set_title("Link Prediction Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "link_prediction_curves.png"), dpi=300)
    plt.close()
    
    return df_results

if __name__ == "__main__":
    from src.link_prediction.candidate_generation import generate_temporal_link_prediction_dataset
    generate_temporal_link_prediction_dataset()
    df_res = evaluate_link_prediction()
    print("Link Prediction Evaluation Metrics:")
    print(df_res)
