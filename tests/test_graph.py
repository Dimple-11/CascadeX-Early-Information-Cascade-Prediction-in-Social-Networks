import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain

sys.path.insert(0, '.')
from src.network.graph_builder import build_network_from_edges

def test_graph_topology_and_metrics():
    print("=== TEST 4: GRAPH TOPOLOGY & REPRODUCIBILITY AUDIT ===")
    edges_path = "data/processed/network_edges.csv"
    cent_path = "data/processed/centrality_scores.csv"
    comm_path = "data/processed/community_assignments.csv"
    
    assert os.path.exists(edges_path), "network_edges.csv missing"
    assert os.path.exists(cent_path), "centrality_scores.csv missing"
    assert os.path.exists(comm_path), "community_assignments.csv missing"
    
    G, df_edges = build_network_from_edges(edges_path)
    
    assert G.number_of_nodes() == 960, f"Expected 960 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() == 5000, f"Expected 5000 edges, got {G.number_of_edges()}"
    assert G.is_directed() == True, "Graph must be directed"
    
    # Check self loops
    self_loops = list(nx.nodes_with_selfloops(G))
    assert len(self_loops) == 0, f"Graph contains self loops: {self_loops}"
    
    # Check Louvain communities
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected, weight='number_trees', random_state=42)
    num_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G_undirected, weight='number_trees')
    
    assert num_communities == 13, f"Expected 13 communities, got {num_communities}"
    assert abs(modularity - 0.42638) < 1e-3, f"Modularity mismatch: {modularity}"
    
    df_cent = pd.read_csv(cent_path)
    assert len(df_cent) == 960, f"Centrality table row count mismatch: {len(df_cent)}"
    
    print(f"  [PASS] Graph Reproducibility: 960 nodes, 5000 edges, 13 communities, Modularity Q = {modularity:.4f}.")
    return True

if __name__ == "__main__":
    test_graph_topology_and_metrics()
