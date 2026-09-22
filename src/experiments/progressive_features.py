import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

sys.path.insert(0, '.')

def run_progressive_feature_experiment(
    feature_matrix_path="data/processed/final_cascade_feature_matrix.csv",
    output_dir="results/experiments",
    fig_dir="results/figures"
):
    """
    Executes Progressive Feature Addition Experiment:
    Model A: Early + Temporal features
    Model B: Model A + Centrality features
    Model C: Model B + Community features
    Model D: Model C + Link Prediction features (Full Model)
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
    
    feature_groups = {
        'Model A: Temporal Only': [
            'observation_level', 'early_event_count', 'early_size', 'early_duration', 'early_growth_rate'
        ],
        'Model B: Model A + Centrality': [
            'observation_level', 'early_event_count', 'early_size', 'early_duration', 'early_growth_rate',
            'early_max_out_degree', 'early_mean_out_degree', 'early_max_betweenness',
            'early_mean_betweenness', 'early_max_pagerank', 'early_mean_pagerank'
        ],
        'Model C: Model B + Community': [
            'observation_level', 'early_event_count', 'early_size', 'early_duration', 'early_growth_rate',
            'early_max_out_degree', 'early_mean_out_degree', 'early_max_betweenness',
            'early_mean_betweenness', 'early_max_pagerank', 'early_mean_pagerank', 'early_comm_count'
        ],
        'Model D: Model C + Link Prediction (Full)': [
            'observation_level', 'early_event_count', 'early_size', 'early_duration', 'early_growth_rate',
            'early_max_out_degree', 'early_mean_out_degree', 'early_max_betweenness',
            'early_mean_betweenness', 'early_max_pagerank', 'early_mean_pagerank', 'early_comm_count',
            'early_mean_link_score', 'early_max_link_score', 'early_preferential_attachment_max'
        ]
    }
    
    results = []
    
    for model_name, cols in feature_groups.items():
        X_tr = df_train[cols]
        X_te = df_test[cols]
        
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        gbr.fit(X_tr, y_train)
        
        y_pred = np.expm1(gbr.predict(X_te))
        
        results.append({
            'experiment_stage': model_name,
            'num_features': len(cols),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': root_mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "progressive_feature_experiment.csv"), index=False)
    
    # Plot Progressive Feature Addition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df_results, x='experiment_stage', y='mae', ax=axes[0], color='royalblue')
    axes[0].set_title("MAE Across Progressive Feature Configurations")
    axes[0].set_ylabel("Mean Absolute Error (Lower is better)")
    axes[0].tick_params(axis='x', rotation=20)
    
    sns.barplot(data=df_results, x='experiment_stage', y='r2', ax=axes[1], color='forestgreen')
    axes[1].set_title("R² Score Across Progressive Feature Configurations")
    axes[1].set_ylabel("R² Variance Explained (Higher is better)")
    axes[1].tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "progressive_feature_experiment.png"), dpi=300)
    plt.close()
    
    return df_results

if __name__ == "__main__":
    df_res = run_progressive_feature_experiment()
    print("Progressive Feature Experiment Results:")
    print(df_res)
