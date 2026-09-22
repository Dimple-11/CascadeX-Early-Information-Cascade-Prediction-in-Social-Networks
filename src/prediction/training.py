import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

sys.path.insert(0, '.')

def train_and_evaluate_prediction_models(
    feature_matrix_path="data/processed/final_cascade_feature_matrix.csv",
    output_dir="results/experiments",
    models_dir="models",
    use_log1p=True
):
    """
    Executes primary Early Cascade Size Prediction ML experiment:
    - Cascade-level Train / Validation / Test split (70% / 15% / 15%).
    - Evaluates Mean, Median, Naive Early Size baselines alongside Ridge, RF, GBR, and XGBoost.
    - Computes MAE, RMSE, R2 on the original cascade size scale.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    df_feat = pd.read_csv(feature_matrix_path)
    
    # Cascade-level Train/Val/Test Split to prevent observation leakage
    unique_cascades = df_feat['cascade_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_cascades)
    
    n_total = len(unique_cascades)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_ids = set(unique_cascades[:n_train])
    val_ids = set(unique_cascades[n_train:n_train+n_val])
    test_ids = set(unique_cascades[n_train+n_val:])
    
    df_train = df_feat[df_feat['cascade_id'].isin(train_ids)].copy()
    df_val = df_feat[df_feat['cascade_id'].isin(val_ids)].copy()
    df_test = df_feat[df_feat['cascade_id'].isin(test_ids)].copy()
    
    feature_cols = [
        'observation_level', 'early_event_count', 'early_size', 'early_comm_count',
        'early_duration', 'early_growth_rate', 'early_max_out_degree', 'early_mean_out_degree',
        'early_max_betweenness', 'early_mean_betweenness', 'early_max_pagerank', 'early_mean_pagerank',
        'early_mean_link_score', 'early_max_link_score', 'early_preferential_attachment_max'
    ]
    
    X_train, y_train = df_train[feature_cols], df_train['target_final_size']
    X_val, y_val = df_val[feature_cols], df_val['target_final_size']
    X_test, y_test = df_test[feature_cols], df_test['target_final_size']
    
    y_train_target = np.log1p(y_train) if use_log1p else y_train
    
    # 1. Baselines
    mean_val = np.mean(y_train)
    median_val = np.median(y_train)
    
    baseline_metrics = []
    
    # Mean baseline
    pred_mean = np.full_like(y_test, mean_val)
    baseline_metrics.append({
        'model_name': 'Baseline: Mean Prediction',
        'mae': mean_absolute_error(y_test, pred_mean),
        'rmse': root_mean_squared_error(y_test, pred_mean),
        'r2': r2_score(y_test, pred_mean)
    })
    
    # Median baseline
    pred_med = np.full_like(y_test, median_val)
    baseline_metrics.append({
        'model_name': 'Baseline: Median Prediction',
        'mae': mean_absolute_error(y_test, pred_med),
        'rmse': root_mean_squared_error(y_test, pred_med),
        'r2': r2_score(y_test, pred_med)
    })
    
    # Naive Early Size baseline
    pred_naive = df_test['early_size']
    baseline_metrics.append({
        'model_name': 'Baseline: Naive Early Size',
        'mae': mean_absolute_error(y_test, pred_naive),
        'rmse': root_mean_squared_error(y_test, pred_naive),
        'r2': r2_score(y_test, pred_naive)
    })
    
    # 2. ML Models
    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        'XGBoost Regressor': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    }
    
    ml_results = []
    trained_models = {}
    test_predictions = df_test[['cascade_id', 'observation_level', 'target_final_size', 'early_size']].copy()
    
    for name, model in models.items():
        model.fit(X_train, y_train_target)
        raw_pred = model.predict(X_test)
        y_pred = np.expm1(raw_pred) if use_log1p else raw_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        ml_results.append({
            'model_name': name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        
        trained_models[name] = model
        test_predictions[f"pred_{name}"] = y_pred
        
    all_results = baseline_metrics + ml_results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(output_dir, "primary_model_evaluations.csv"), index=False)
    test_predictions.to_csv(os.path.join(output_dir, "test_cascade_predictions.csv"), index=False)
    
    return df_results, trained_models, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    df_res, trained_models, splits = train_and_evaluate_prediction_models()
    print("Primary Prediction Model Evaluation Results:")
    print(df_res)
