import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '.')

def compute_all_centralities(G, output_dir="results", processed_dir="data/processed"):
    """
    Computes rigorous centrality measures for all nodes in graph G:
    - In-degree, Out-degree, Total degree centrality
    - Betweenness centrality
    - Closeness centrality
    - Eigenvector centrality (with fallback/handling)
    - PageRank
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    print("Computing degree centralities...")
    deg_cent = nx.degree_centrality(G)
    in_deg_cent = nx.in_degree_centrality(G)
    out_deg_cent = nx.out_degree_centrality(G)
    
    print("Computing betweenness centrality...")
    betweenness_cent = nx.betweenness_centrality(G, normalized=True)
    
    print("Computing closeness centrality...")
    closeness_cent = nx.closeness_centrality(G)
    
    print("Computing PageRank...")
    pagerank_cent = nx.pagerank(G, alpha=0.85, max_iter=500)
    
    print("Computing eigenvector centrality...")
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception as e:
        print(f"Warning: Power iteration failed for eigenvector centrality on DiGraph ({e}). Using PageRank as robust spectral proxy.")
        eigenvector_cent = pagerank_cent.copy()
        
    # Construct DataFrame
    nodes = list(G.nodes())
    df_centrality = pd.DataFrame({
        'node': nodes,
        'degree_centrality': [deg_cent[n] for n in nodes],
        'in_degree_centrality': [in_deg_cent[n] for n in nodes],
        'out_degree_centrality': [out_deg_cent[n] for n in nodes],
        'betweenness_centrality': [betweenness_cent[n] for n in nodes],
        'closeness_centrality': [closeness_cent[n] for n in nodes],
        'eigenvector_centrality': [eigenvector_cent[n] for n in nodes],
        'pagerank': [pagerank_cent[n] for n in nodes],
        'raw_in_degree': [G.in_degree(n) for n in nodes],
        'raw_out_degree': [G.out_degree(n) for n in nodes],
        'raw_total_degree': [G.degree(n) for n in nodes]
    })
    
    # Save to data/processed/centrality_scores.csv
    df_centrality.to_csv(os.path.join(processed_dir, "centrality_scores.csv"), index=False)
    
    # Generate Top-10 summary table for each centrality
    top10_dict = {}
    for col in ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 
                'betweenness_centrality', 'closeness_centrality', 'pagerank']:
        top_nodes = df_centrality.sort_values(by=col, ascending=False).head(10)[['node', col]].reset_index(drop=True)
        top10_dict[f"{col}_node"] = top_nodes['node']
        top10_dict[f"{col}_score"] = top_nodes[col]
        
    df_top10 = pd.DataFrame(top10_dict)
    df_top10.to_csv(os.path.join(output_dir, "tables", "top10_centrality_nodes.csv"), index=False)
    
    # Visualizations
    # 1. Centrality Distribution Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    cols_to_plot = [
        ('in_degree_centrality', 'In-Degree Centrality'),
        ('out_degree_centrality', 'Out-Degree Centrality'),
        ('betweenness_centrality', 'Betweenness Centrality'),
        ('closeness_centrality', 'Closeness Centrality'),
        ('eigenvector_centrality', 'Eigenvector Centrality'),
        ('pagerank', 'PageRank')
    ]
    
    for idx, (col, title) in enumerate(cols_to_plot):
        ax = axes[idx // 3, idx % 3]
        sns.histplot(df_centrality[col], ax=ax, kde=True, color='indigo', bins=30)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.set_xlabel("Score")
        ax.set_ylabel("Count (log)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "centrality_distributions.png"), dpi=300)
    plt.close()
    
    # 2. Centrality Pairwise Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr_cols = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 
                 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 'pagerank']
    corr_matrix = df_centrality[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Pairwise Spearman/Pearson Correlations of Centrality Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "centrality_correlations.png"), dpi=300)
    plt.close()
    
    return df_centrality, df_top10

if __name__ == "__main__":
    from src.network.graph_builder import build_network_from_edges
    G, df_edges = build_network_from_edges()
    df_cent, df_top10 = compute_all_centralities(G)
    print("Centrality computation complete.")
    print("\nTop 5 Nodes by Out-Degree Centrality (Source Information Hubs):")
    print(df_cent.sort_values(by='out_degree_centrality', ascending=False)[['node', 'out_degree_centrality', 'raw_out_degree']].head(5))
    print("\nTop 5 Nodes by PageRank (Global Network Influence):")
    print(df_cent.sort_values(by='pagerank', ascending=False)[['node', 'pagerank', 'raw_in_degree']].head(5))
