import os
import sys
import pandas as pd
import numpy as np
import networkx as nx

sys.path.insert(0, '.')
from src.link_prediction.heuristics import (
    common_neighbors_score, jaccard_coefficient_score,
    adamic_adar_score, preferential_attachment_score, resource_allocation_score
)

def generate_temporal_link_prediction_dataset(
    edges_csv_path="data/processed/network_edges.csv",
    output_dir="results/link_prediction",
    train_ratio=0.70,
    random_state=42
):
    """
    Constructs a temporal link prediction experiment dataset:
    - Train graph G_train built from earlier edges (70% = 3,500 edges).
    - Future positive test links from remaining 30% (1,500 edges).
    - Equal number of negative non-existent edge pairs sampled from non-edges.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_edges = pd.read_csv(edges_csv_path)
    
    # Sort chronologically by average_timediff or index
    df_sorted = df_edges.sort_values(by=['average_timediff', 'index']).reset_index(drop=True)
    
    num_train = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:num_train]
    test_df = df_sorted.iloc[num_train:]
    
    # Construct historical graph G_train
    G_train = nx.DiGraph()
    for _, row in train_df.iterrows():
        G_train.add_edge(row['src'], row['dst'], weight=row['number_trees'])
        
    G_train_undirected = G_train.to_undirected()
    
    # Target positive future links
    positive_pairs = [(row['src'], row['dst']) for _, row in test_df.iterrows()]
    
    # Generate negative pairs (non-edges not in G_train and not in test_df)
    existing_edges = set(zip(df_sorted['src'], df_sorted['dst'])).union(set(zip(df_sorted['dst'], df_sorted['src'])))
    all_nodes = list(set(df_sorted['src']).union(set(df_sorted['dst'])))
    
    np.random.seed(random_state)
    negative_pairs = []
    needed_negatives = len(positive_pairs)
    
    attempts = 0
    max_attempts = needed_negatives * 20
    while len(negative_pairs) < needed_negatives and attempts < max_attempts:
        u, v = np.random.choice(all_nodes, size=2, replace=False)
        if (u, v) not in existing_edges and (v, u) not in existing_edges:
            negative_pairs.append((u, v))
            existing_edges.add((u, v))
        attempts += 1
        
    # Construct dataset records
    records = []
    
    for label, pair_list in [(1, positive_pairs), (0, negative_pairs)]:
        for u, v in pair_list:
            cn = common_neighbors_score(G_train_undirected, u, v)
            jc = jaccard_coefficient_score(G_train_undirected, u, v)
            aa = adamic_adar_score(G_train_undirected, u, v)
            pa = preferential_attachment_score(G_train_undirected, u, v)
            ra = resource_allocation_score(G_train_undirected, u, v)
            
            in_deg_u = G_train.in_degree(u) if G_train.has_node(u) else 0
            out_deg_u = G_train.out_degree(u) if G_train.has_node(u) else 0
            in_deg_v = G_train.in_degree(v) if G_train.has_node(v) else 0
            out_deg_v = G_train.out_degree(v) if G_train.has_node(v) else 0
            
            records.append({
                'src': u,
                'dst': v,
                'label': label,
                'common_neighbors': cn,
                'jaccard_coeff': jc,
                'adamic_adar': aa,
                'preferential_attachment': pa,
                'resource_allocation': ra,
                'out_deg_u': out_deg_u,
                'in_deg_v': in_deg_v
            })
            
    df_link = pd.DataFrame(records)
    output_path = os.path.join(output_dir, "link_prediction_candidates.csv")
    df_link.to_csv(output_path, index=False)
    print(f"Generated temporal link prediction dataset with {len(df_link)} samples (Pos: {len(positive_pairs)}, Neg: {len(negative_pairs)}).")
    return df_link, G_train

if __name__ == "__main__":
    df_link, G_train = generate_temporal_link_prediction_dataset()
