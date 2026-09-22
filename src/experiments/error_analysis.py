import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, '.')

def run_error_analysis(
    feature_matrix_path="data/processed/final_cascade_feature_matrix.csv",
    output_dir="results/experiments",
    fig_dir="results/figures"
):
    """
    Executes systematic error analysis on cascade size prediction errors.
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
    df_test = df_feat[df_feat['cascade_id'].isin(test_ids)].copy()
    
    X_train = df_train[feature_cols]
    y_train = np.log1p(df_train['target_final_size'])
    
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    gbr.fit(X_train, y_train)
    
    df_test['predicted_final_size'] = np.expm1(gbr.predict(df_test[feature_cols]))
    df_test['absolute_error'] = np.abs(df_test['target_final_size'] - df_test['predicted_final_size'])
    df_test['relative_error'] = df_test['absolute_error'] / df_test['target_final_size']
    df_test['residual'] = df_test['predicted_final_size'] - df_test['target_final_size']
    
    df_test.to_csv(os.path.join(output_dir, "error_analysis_dataset.csv"), index=False)
    
    # Save Error Analysis Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Actual vs Predicted Cascade Size
    sns.scatterplot(data=df_test, x='target_final_size', y='predicted_final_size', hue='observation_level', ax=axes[0], palette='viridis', alpha=0.7)
    axes[0].plot([df_test['target_final_size'].min(), df_test['target_final_size'].max()],
                 [df_test['target_final_size'].min(), df_test['target_final_size'].max()], 'r--', label='Ideal 1:1 Line')
    axes[0].set_title("Actual vs. Predicted Final Cascade Size")
    axes[0].set_xlabel("Actual Final Cascade Size")
    axes[0].set_ylabel("Predicted Final Cascade Size")
    axes[0].legend()
    
    # 2. Residual Distribution
    sns.histplot(df_test['residual'], ax=axes[1], kde=True, color='purple', bins=30)
    axes[1].set_title("Prediction Residual Distribution (Predicted - Actual)")
    axes[1].set_xlabel("Residual Error (Unique Websites)")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "error_analysis_plots.png"), dpi=300)
    plt.close()
    
    return df_test

if __name__ == "__main__":
    df_err = run_error_analysis()
    print("Error Analysis Complete:")
    print(f"Mean Absolute Error across test set: {df_err['absolute_error'].mean():.2f} websites")
    print(f"Mean Relative Error across test set: {df_err['relative_error'].mean():.2%}")
