import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '.')

def detect_communities(G, output_dir="results", processed_dir="data/processed"):
    """
    Performs Louvain community detection on graph G, calculates modularity and edge ratios,
    and outputs node assignments and visualization figures.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Louvain runs on undirected graph with weights
    G_undirected = G.to_undirected()
    
    # Run Louvain algorithm
    partition = community_louvain.best_partition(G_undirected, weight='number_trees', random_state=42)
    modularity = community_louvain.modularity(partition, G_undirected, weight='number_trees')
    
    # Construct DataFrame
    nodes = list(G.nodes())
    df_comm = pd.DataFrame({
        'node': nodes,
        'community_id': [partition[n] for n in nodes]
    })
    
    # Save assignments
    df_comm.to_csv(os.path.join(processed_dir, "community_assignments.csv"), index=False)
    
    # Calculate Community Statistics
    comm_sizes = df_comm['community_id'].value_counts()
    num_communities = len(comm_sizes)
    
    # Intra-community vs Inter-community edges
    intra_edges = 0
    inter_edges = 0
    for u, v in G.edges():
        if partition[u] == partition[v]:
            intra_edges += 1
        else:
            inter_edges += 1
            
    total_edges = G.number_of_edges()
    intra_ratio = intra_edges / total_edges if total_edges > 0 else 0
    inter_ratio = inter_edges / total_edges if total_edges > 0 else 0
    
    comm_metrics = {
        "algorithm": "Louvain",
        "num_communities": num_communities,
        "modularity": modularity,
        "largest_community_size": int(comm_sizes.max()),
        "smallest_community_size": int(comm_sizes.min()),
        "mean_community_size": float(comm_sizes.mean()),
        "median_community_size": float(comm_sizes.median()),
        "intra_community_edges": intra_edges,
        "inter_community_edges": inter_edges,
        "intra_edge_ratio": intra_ratio,
        "inter_edge_ratio": inter_ratio
    }
    
    # Save table
    df_comm_metrics = pd.DataFrame([comm_metrics])
    df_comm_metrics.to_csv(os.path.join(output_dir, "tables", "community_detection_metrics.csv"), index=False)
    
    # Save Top Communities Summary Table
    top_comms = comm_sizes.head(10).reset_index()
    top_comms.columns = ['community_id', 'node_count']
    top_comms.to_csv(os.path.join(output_dir, "tables", "top_communities_summary.csv"), index=False)
    
    # Visualizations
    # 1. Community Size Distribution
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_comms['community_id'].astype(str), y=top_comms['node_count'], color='darkcyan')
    plt.title(f"Top 10 Largest Detected Network Communities (Total: {num_communities}, Modularity Q={modularity:.4f})")
    plt.xlabel("Community ID")
    plt.ylabel("Number of Websites")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "community_size_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Community Network Subgraph Sample (Top 5 communities)
    top5_comm_ids = set(comm_sizes.head(5).index)
    nodes_sample = [n for n in G.nodes() if partition[n] in top5_comm_ids]
    # Further sample to top 100 highest degree nodes for readability
    nodes_sample = sorted(nodes_sample, key=lambda n: G.degree(n), reverse=True)[:100]
    subgraph = G.subgraph(nodes_sample)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    node_colors = [partition[n] for n in subgraph.nodes()]
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=120, node_color=node_colors, cmap=plt.cm.tab10, alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='grey', arrows=False)
    
    plt.title("Sampled Network Subgraph Colored by Detected Communities (Top 5 Communities)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "network_communities_sample.png"), dpi=300)
    plt.close()
    
    return df_comm, comm_metrics

if __name__ == "__main__":
    from src.network.graph_builder import build_network_from_edges
    G, df_edges = build_network_from_edges()
    df_comm, metrics = detect_communities(G)
    print("Community Detection Complete:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
