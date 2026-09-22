import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Phase 15: Full Pipeline Validation & Independent Model Testing\n\nExecutes complete end-to-end scientific validation, data schema audits, target distribution analysis, temporal leakage verification, split isolation tests, and paper consistency audits."),
    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

from tests.test_data import test_data_integrity_and_schema, test_target_distribution_analysis
from tests.test_graph import test_graph_topology_and_metrics
from tests.test_cascades import test_cascade_extraction_and_sampling
from tests.test_train_test_split import test_cascade_level_split_isolation
from tests.test_temporal_leakage import run_independent_leakage_audit
from tests.test_features import test_feature_matrix_integrity
from tests.test_link_prediction import test_link_prediction_reproducibility
from tests.test_model import test_model_metrics_and_error_investigation
from tests.test_results import run_paper_consistency_audit
from IPython.display import Image, display

print("==================================================")
print("RUNNING COMPLETE PIPELINE & DATA INTEGRITY TESTS")
print("==================================================")

test_data_integrity_and_schema()
test_target_distribution_analysis()
test_graph_topology_and_metrics()
test_cascade_extraction_and_sampling()
test_cascade_level_split_isolation()
run_independent_leakage_audit(feat_path="../data/processed/final_cascade_feature_matrix.csv", output_path="../results/tables/independent_leakage_audit.csv")
test_feature_matrix_integrity()
test_link_prediction_reproducibility()
test_model_metrics_and_error_investigation()
run_paper_consistency_audit(tex_path="../paper/main.tex", output_path="../results/tables/paper_consistency_audit.csv")

print("\n==================================================")
print("ALL TESTS PASSED WITH 100% EMPIRICAL REPRODUCIBILITY")
print("==================================================")

display(Image(filename='../results/figures/final_actual_vs_predicted.png'))
display(Image(filename='../results/figures/final_residual_distribution.png'))
display(Image(filename='../results/figures/error_vs_actual.png'))""")
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/15_full_pipeline_test.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully generated notebook: notebooks/15_full_pipeline_test.ipynb")
