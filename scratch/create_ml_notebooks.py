import nbformat as nbf
import os

def create_notebook(filename, title, description, code_cell_content):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{description}"),
        nbf.v4.new_code_cell(code_cell_content)
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {filename}")

os.makedirs('notebooks', exist_ok=True)

# 08 Link Prediction
create_notebook(
    'notebooks/08_link_prediction.ipynb',
    'Phase 7: Temporal Link Prediction Experiment',
    'Evaluates structural link prediction heuristics (Common Neighbors, Jaccard Coefficient, Adamic-Adar, Preferential Attachment) on a 70/30 temporal graph split.',
    """import sys
sys.path.insert(0, '..')
from src.link_prediction.candidate_generation import generate_temporal_link_prediction_dataset
from src.link_prediction.evaluation import evaluate_link_prediction
from IPython.display import Image, display

df_candidates, G_train = generate_temporal_link_prediction_dataset(edges_csv_path='../data/processed/network_edges.csv', output_dir='../results/link_prediction')
df_metrics = evaluate_link_prediction(candidates_csv='../results/link_prediction/link_prediction_candidates.csv', output_dir='../results/link_prediction', fig_dir='../results/figures')

print("--- LINK PREDICTION METRICS ---")
display(df_metrics)
display(Image(filename='../results/figures/link_prediction_curves.png'))"""
)

# 09 Cascade Feature Engineering
create_notebook(
    'notebooks/09_cascade_feature_engineering.ipynb',
    'Phase 8: Cascade Feature Engineering & Leakage Audit',
    'Constructs the unified ML feature dataset combining temporal, centrality, community, and structural link features without temporal leakage.',
    """import sys
sys.path.insert(0, '..')
from src.features.cascade_features import build_final_cascade_feature_matrix
import pandas as pd
from IPython.display import display

df_feat = build_final_cascade_feature_matrix(
    temporal_features_path='../data/processed/temporal_cascade_features.csv',
    edges_csv_path='../data/processed/network_edges.csv',
    output_filepath='../data/processed/final_cascade_feature_matrix.csv'
)

print("--- FEATURE MATRIX SAMPLE ---")
display(df_feat.head())"""
)

# 10 Early Cascade Prediction
create_notebook(
    'notebooks/10_early_cascade_prediction.ipynb',
    'Phase 9: Primary Early Cascade Size Prediction',
    'Trains baselines (Mean, Median, Naive) alongside Ridge, Random Forest, Gradient Boosting, and XGBoost models using cascade-level temporal splitting.',
    """import sys
sys.path.insert(0, '..')
from src.prediction.training import train_and_evaluate_prediction_models
from IPython.display import display

df_res, trained_models, splits = train_and_evaluate_prediction_models(
    feature_matrix_path='../data/processed/final_cascade_feature_matrix.csv',
    output_dir='../results/experiments',
    models_dir='../models'
)

print("--- PRIMARY MODEL EVALUATIONS ---")
display(df_res)"""
)

# 11 Progressive Feature Experiment
create_notebook(
    'notebooks/11_progressive_feature_experiment.ipynb',
    'Phase 10: Progressive Feature Addition Experiment',
    'Evaluates incremental predictive contributions of Temporal (A), Centrality (B), Community (C), and Link Prediction (D) feature groups.',
    """import sys
sys.path.insert(0, '..')
from src.experiments.progressive_features import run_progressive_feature_experiment
from IPython.display import Image, display

df_prog = run_progressive_feature_experiment(
    feature_matrix_path='../data/processed/final_cascade_feature_matrix.csv',
    output_dir='../results/experiments',
    fig_dir='../results/figures'
)

print("--- PROGRESSIVE FEATURE EXPERIMENT ---")
display(df_prog)
display(Image(filename='../results/figures/progressive_feature_experiment.png'))"""
)

# 12 Early Observation Experiment
create_notebook(
    'notebooks/12_early_observation_experiment.ipynb',
    'Phase 11: Early Observation Window Trajectory Experiment',
    'Measures predictive accuracy across 10%, 20%, 30%, and 50% early observation windows to answer how early cascade size can be accurately predicted.',
    """import sys
sys.path.insert(0, '..')
from src.experiments.early_observation import run_early_observation_experiment
from IPython.display import Image, display

df_obs = run_early_observation_experiment(
    feature_matrix_path='../data/processed/final_cascade_feature_matrix.csv',
    output_dir='../results/experiments',
    fig_dir='../results/figures'
)

print("--- EARLY OBSERVATION EXPERIMENT ---")
display(df_obs)
display(Image(filename='../results/figures/early_observation_experiment.png'))"""
)

# 13 Ablation Study
create_notebook(
    'notebooks/13_ablation_study.ipynb',
    'Phase 12 & 13: Feature Group Ablation & Permutation Feature Importance',
    'Conducts single-group feature ablation and model-agnostic permutation feature importance analysis to evaluate feature group contributions.',
    """import sys
sys.path.insert(0, '..')
from src.experiments.ablation import run_ablation_and_importance_study
from IPython.display import Image, display

df_abl, df_imp = run_ablation_and_importance_study(
    feature_matrix_path='../data/processed/final_cascade_feature_matrix.csv',
    output_dir='../results/experiments',
    fig_dir='../results/figures'
)

print("--- ABLATION RESULTS ---")
display(df_abl)
print("--- TOP 10 PERMUTATION FEATURE IMPORTANCES ---")
display(df_imp.head(10))

display(Image(filename='../results/figures/ablation_study.png'))
display(Image(filename='../results/figures/feature_importance.png'))"""
)

# 14 Error Analysis
create_notebook(
    'notebooks/14_error_analysis.ipynb',
    'Phase 14: Systematic Error Analysis',
    'Analyzes residual errors, over-predictions, under-predictions, and prediction accuracy across cascade characteristics.',
    """import sys
sys.path.insert(0, '..')
from src.experiments.error_analysis import run_error_analysis
from IPython.display import Image, display

df_err = run_error_analysis(
    feature_matrix_path='../data/processed/final_cascade_feature_matrix.csv',
    output_dir='../results/experiments',
    fig_dir='../results/figures'
)

print(f"Mean Absolute Error across test set: {df_err['absolute_error'].mean():.2f} websites")
print(f"Mean Relative Error across test set: {df_err['relative_error'].mean():.2%}")

display(Image(filename='../results/figures/error_analysis_plots.png'))"""
)
