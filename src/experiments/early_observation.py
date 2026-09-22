import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

sys.path.insert(0, '.')

def run_early_observation_experiment(
    feature_matrix_path="data/processed/final_cascade_feature_matrix.csv",
    output_dir="results/experiments",
    fig_dir="results/figures"
):
    """
    Evaluates early prediction performance across observation windows (10%, 20%, 30%, 50%).
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
    
    feature_cols = [
        'observation_level', 'early_event_count', 'early_size', 'early_comm_count',
        'early_duration', 'early_growth_rate', 'early_max_out_degree', 'early_mean_out_degree',
        'early_max_betweenness', 'early_mean_betweenness', 'early_max_pagerank', 'early_mean_pagerank',
        'early_mean_link_score', 'early_max_link_score', 'early_preferential_attachment_max'
    ]
    
    df_train = df_feat[df_feat['cascade_id'].isin(train_ids)].copy()
    df_test_full = df_feat[df_feat['cascade_id'].isin(test_ids)].copy()
    
    X_train = df_train[feature_cols]
    y_train = np.log1p(df_train['target_final_size'])
    
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    gbr.fit(X_train, y_train)
    
    observation_levels = [0.10, 0.20, 0.30, 0.50]
    results = []
    
    for p in observation_levels:
        df_sub = df_test_full[df_test_full['observation_level'] == p]
        X_sub = df_sub[feature_cols]
        y_sub = df_sub['target_final_size']
        
        y_pred = np.expm1(gbr.predict(X_sub))
        
        results.append({
            'observation_level': p,
            'observation_pct': f"{int(p * 100)}%",
            'num_test_cascades': len(df_sub),
            'mean_early_size': float(df_sub['early_size'].mean()),
            'mean_final_size': float(y_sub.mean()),
            'mae': mean_absolute_error(y_sub, y_pred),
            'rmse': root_mean_squared_error(y_sub, y_pred),
            'r2': r2_score(y_sub, y_pred)
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "early_observation_experiment.csv"), index=False)
    
    # Plot Early Observation Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.lineplot(data=df_results, x='observation_pct', y='mae', marker='o', ax=axes[0], color='crimson', linewidth=2.5)
    axes[0].set_title("Prediction MAE vs. Observation Window")
    axes[0].set_xlabel("Early Observation Window")
    axes[0].set_ylabel("MAE (Unique Websites Error)")
    
    sns.lineplot(data=df_results, x='observation_pct', y='r2', marker='s', ax=axes[1], color='darkcyan', linewidth=2.5)
    axes[1].set_title("Prediction R² vs. Observation Window")
    axes[1].set_xlabel("Early Observation Window")
    axes[1].set_ylabel("R² Score")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "early_observation_experiment.png"), dpi=300)
    plt.close()
    
    return df_results

if __name__ == "__main__":
    df_res = run_early_observation_experiment()
    print("Early Observation Stage Experiment Results:")
    print(df_res)
