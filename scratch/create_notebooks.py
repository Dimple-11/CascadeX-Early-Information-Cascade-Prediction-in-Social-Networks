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

# 02 Data Preprocessing
create_notebook(
    'notebooks/02_data_preprocessing.ipynb',
    'Phase 1: Data Preprocessing & Data Quality Audit',
    'Loads the raw MemeTracker / NETINF dataset, executes missing value and duplicate record audits, and serializes clean network edge list to `data/processed/network_edges.csv`.',
    """import sys
sys.path.insert(0, '..')
from src.preprocessing.loader import load_and_preprocess_dataset
import pandas as pd

df_edges, audit_report = load_and_preprocess_dataset('../data/raw/InfoNet5000Q1000NEXP.txt', '../data/processed/network_edges.csv')
print("--- DATA PREPROCESSING AUDIT REPORT ---")
for k, v in audit_report.items():
    print(f"{k}: {v}")
df_edges.head()"""
)

# 03 Basic Network Analysis
create_notebook(
    'notebooks/03_basic_network_analysis.ipynb',
    'Phase 2 & 4: Network Construction & Basic SNA Analysis',
    'Constructs the directed weighted information diffusion network $G=(V,E,W)$ and calculates fundamental Social Network Analysis (SNA) metrics.',
    """import sys
sys.path.insert(0, '..')
from src.network.graph_builder import build_network_from_edges, compute_basic_network_analysis
from IPython.display import Image, display

G, df_edges = build_network_from_edges('../data/processed/network_edges.csv')
metrics = compute_basic_network_analysis(G, df_edges, output_dir='../results')

print("--- BASIC NETWORK ANALYSIS METRICS ---")
for k, v in metrics.items():
    print(f"{k}: {v}")

display(Image(filename='../results/figures/degree_distribution.png'))
display(Image(filename='../results/figures/network_subgraph_sample.png'))"""
)

# 04 Centrality Analysis
create_notebook(
    'notebooks/04_centrality_analysis.ipynb',
    'Phase 5: Network Centrality Analysis',
    'Calculates node degree centrality, in-degree, out-degree, betweenness centrality, closeness centrality, eigenvector centrality, and PageRank across all 960 website nodes.',
    """import sys
sys.path.insert(0, '..')
from src.network.graph_builder import build_network_from_edges
from src.centrality.centrality_calculator import compute_all_centralities
from IPython.display import Image, display
import pandas as pd

G, _ = build_network_from_edges('../data/processed/network_edges.csv')
df_cent, df_top10 = compute_all_centralities(G, output_dir='../results', processed_dir='../data/processed')

print("--- TOP 10 NODES BY PAGERANK & OUT-DEGREE ---")
display(df_top10.head(10))

display(Image(filename='../results/figures/centrality_distributions.png'))
display(Image(filename='../results/figures/centrality_correlations.png'))"""
)

# 05 Community Detection
create_notebook(
    'notebooks/05_community_detection.ipynb',
    'Phase 6: Community Detection & Modularity Analysis',
    'Applies the Louvain community detection algorithm to partition the 960 websites into detected network communities, computing modularity Q and inter/intra community connectivity.',
    """import sys
sys.path.insert(0, '..')
from src.network.graph_builder import build_network_from_edges
from src.community.community_detector import detect_communities
from IPython.display import Image, display

G, _ = build_network_from_edges('../data/processed/network_edges.csv')
df_comm, comm_metrics = detect_communities(G, output_dir='../results', processed_dir='../data/processed')

print("--- COMMUNITY DETECTION METRICS ---")
for k, v in comm_metrics.items():
    print(f"{k}: {v}")

display(Image(filename='../results/figures/community_size_distribution.png'))
display(Image(filename='../results/figures/network_communities_sample.png'))"""
)

# 06 Cascade Extraction
create_notebook(
    'notebooks/06_cascade_extraction.ipynb',
    'Phase 7: Cascade Extraction & Distribution Analysis',
    'Extracts individual information diffusion cascade profiles, defines the primary target variable (Final Cascade Size = Total Unique Websites Reached), and analyzes cascade distributions.',
    """import sys
sys.path.insert(0, '..')
from src.cascades.cascade_extractor import extract_cascade_data
from IPython.display import Image, display

df_cascades, summary = extract_cascade_data(output_dir='../results', processed_dir='../data/processed')

print("--- CASCADE EXTRACTION SUMMARY ---")
for k, v in summary.items():
    print(f"{k}: {v}")

display(Image(filename='../results/figures/cascade_size_distribution.png'))
display(Image(filename='../results/figures/cascade_size_vs_communities.png'))"""
)

# 07 Temporal Cascade Analysis
create_notebook(
    'notebooks/07_temporal_cascade_analysis.ipynb',
    'Phase 8: Temporal Cascade Representation & Leakage Audit',
    'Simulates early cascade observation windows (10%, 20%, 30%, 50%), extracts clean early-stage predictors, and performs a strict Temporal Leakage Audit.',
    """import sys
sys.path.insert(0, '..')
from src.temporal.temporal_analyzer import build_temporal_early_observation_dataset
from IPython.display import Image, display

df_temporal, df_audit = build_temporal_early_observation_dataset(output_dir='../results', processed_dir='../data/processed')

print("--- MANDATORY TEMPORAL LEAKAGE AUDIT REPORT ---")
display(df_audit[['feature_name', 'correlation_with_target', 'audit_status']])

display(Image(filename='../results/figures/early_observation_growth_curves.png'))"""
)
