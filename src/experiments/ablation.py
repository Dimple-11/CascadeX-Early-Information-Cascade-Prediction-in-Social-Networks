import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

sys.path.insert(0, '.')

def run_ablation_and_importance_study(
    feature_matrix_path="data/processed/final_cascade_feature_matrix.csv",
    output_dir="results/experiments",
    fig_dir="results/figures"
):
    """
    Executes Feature Group Ablation Study and Permutation Feature Importance Analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    df_feat = pd.read_csv(feature_matrix_path)
    
    unique_cascades = df_feat['cascade_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_cascades)
    n_train = int(len(unique_cascades) * 0.70)
    train_ids = set(unique_cascades[:n_train])
    test_ids = set(unique_cascades[n_train:])
    
    df_train = df_feat[df_feat['cascade_id'].isin(train_ids)].copy()
    df_test = df_feat[df_feat['cascade_id'].isin(test_ids)].copy()
    
    y_train = np.log1p(df_train['target_final_size'])
    y_test = df_test['target_final_size']
    
    temporal_cols = ['observation_level', 'early_event_count', 'early_size', 'early_duration', 'early_growth_rate']
    centrality_cols = ['early_max_out_degree', 'early_mean_out_degree', 'early_max_betweenness', 'early_mean_betweenness', 'early_max_pagerank', 'early_mean_pagerank']
    community_cols = ['early_comm_count']
    link_cols = ['early_mean_link_score', 'early_max_link_score', 'early_preferential_attachment_max']
    
    all_cols = temporal_cols + centrality_cols + community_cols + link_cols
    
    ablation_configs = {
        'Full Model': all_cols,
        'Full - Temporal': [c for c in all_cols if c not in temporal_cols],
        'Full - Centrality': [c for c in all_cols if c not in centrality_cols],
        'Full - Community': [c for c in all_cols if c not in community_cols],
        'Full - Link Prediction': [c for c in all_cols if c not in link_cols]
    }
    
    results = []
    
    for config_name, cols in ablation_configs.items():
        X_tr = df_train[cols]
        X_te = df_test[cols]
        
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        gbr.fit(X_tr, y_train)
        
        y_pred = np.expm1(gbr.predict(X_te))
        
        results.append({
            'ablation_config': config_name,
            'features_removed': 'None_All_Features' if config_name == 'Full Model' else config_name.replace('Full - ', ''),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': root_mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        })
        
    df_ablation = pd.DataFrame(results)
    df_ablation.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)
    
    # 2. Permutation Feature Importance
    gbr_full = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    gbr_full.fit(df_train[all_cols], y_train)
    
    perm_imp = permutation_importance(gbr_full, df_test[all_cols], np.log1p(y_test), n_repeats=10, random_state=42)
    
    df_imp = pd.DataFrame({
        'feature_name': all_cols,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values(by='importance_mean', ascending=False).reset_index(drop=True)
    
    df_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    # Visualizations
    # 1. Ablation Study Plot
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_ablation, x='ablation_config', y='r2', palette='Spectral')
    plt.title("Feature Group Ablation Study: Impact on R² Performance")
    plt.xlabel("Ablation Configuration")
    plt.ylabel("R² Variance Explained")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_study.png"), dpi=300)
    plt.close()
    
    # 2. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='importance_mean', y='feature_name', palette='crest')
    plt.title("Model-Agnostic Permutation Feature Importance")
    plt.xlabel("Mean Permutation Importance (Decrease in Model Score)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=300)
    plt.close()
    
    return df_ablation, df_imp

if __name__ == "__main__":
    df_abl, df_imp = run_ablation_and_importance_study()
    print("Ablation Study Results:")
    print(df_abl)
    print("\nTop 5 Most Important Features:")
    print(df_imp.head(5))
