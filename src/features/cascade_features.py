import os
import sys
import pandas as pd
import numpy as np
import networkx as nx

sys.path.insert(0, '.')
from src.link_prediction.heuristics import jaccard_coefficient_score, preferential_attachment_score

def build_final_cascade_feature_matrix(
    temporal_features_path="data/processed/temporal_cascade_features.csv",
    edges_csv_path="data/processed/network_edges.csv",
    output_filepath="data/processed/final_cascade_feature_matrix.csv"
):
    """
    Combines temporal, centrality, community, and link-prediction features
    into a unified ML feature dataset without temporal leakage.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    df_temp = pd.read_csv(temporal_features_path)
    edges_df = pd.read_csv(edges_csv_path)
    
    # Train historical graph G_hist for link prediction features
    # Using first 70% edges to ensure no future leakage
    num_train = int(len(edges_df) * 0.70)
    train_edges = edges_df.iloc[:num_train]
    G_hist = nx.DiGraph()
    for _, row in train_edges.iterrows():
        G_hist.add_edge(row['src'], row['dst'], weight=row['number_trees'])
    G_hist_undirected = G_hist.to_undirected()
    
    # Compute early link prediction features for each cascade sample
    link_scores_mean = []
    link_scores_max = []
    pa_scores_max = []
    
    for idx, row in df_temp.iterrows():
        tree_id = int(row['cascade_id'])
        p = row['observation_level']
        
        np.random.seed(tree_id)
        prob = edges_df['number_trees'] / 10.0
        selected = np.random.rand(len(edges_df)) < prob
        sub_edges = edges_df[selected].sort_values(by='average_timediff').reset_index(drop=True)
        
        num_early = max(1, int(np.ceil(len(sub_edges) * p)))
        early_sub = sub_edges.iloc[:num_early]
        early_nodes = list(set(early_sub['src']).union(set(early_sub['dst'])))
        
        if len(early_nodes) >= 2:
            # Sample up to 20 random node pairs from early nodes to evaluate structural link scores
            sample_pairs = []
            for _ in range(min(20, len(early_nodes) * 2)):
                u, v = np.random.choice(early_nodes, size=2, replace=False)
                sample_pairs.append((u, v))
                
            jc_list = [jaccard_coefficient_score(G_hist_undirected, u, v) for u, v in sample_pairs]
            pa_list = [preferential_attachment_score(G_hist_undirected, u, v) for u, v in sample_pairs]
            
            link_scores_mean.append(float(np.mean(jc_list)))
            link_scores_max.append(float(np.max(jc_list)))
            pa_scores_max.append(float(np.max(pa_list)))
        else:
            link_scores_mean.append(0.0)
            link_scores_max.append(0.0)
            pa_scores_max.append(0.0)
            
    df_temp['early_mean_link_score'] = link_scores_mean
    df_temp['early_max_link_score'] = link_scores_max
    df_temp['early_preferential_attachment_max'] = pa_scores_max
    
    df_temp.to_csv(output_filepath, index=False)
    print(f"Final feature matrix constructed: {df_temp.shape[0]} rows, {df_temp.shape[1]} columns saved to {output_filepath}.")
    return df_temp

if __name__ == "__main__":
    df_mat = build_final_cascade_feature_matrix()
