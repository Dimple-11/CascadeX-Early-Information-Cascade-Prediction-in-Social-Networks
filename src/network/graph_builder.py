import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def build_network_from_edges(edges_csv_path="data/processed/network_edges.csv"):
    """
    Builds a directed weighted networkX DiGraph from processed edges CSV.
    """
    df_edges = pd.read_csv(edges_csv_path)
    G = nx.DiGraph()
    
    for idx, row in df_edges.iterrows():
        G.add_edge(
            row['src'],
            row['dst'],
            weight=row['marginal_gain'],
            number_trees=row['number_trees'],
            marginal_gain=row['marginal_gain'],
            median_timediff=row['median_timediff'],
            average_timediff=row['average_timediff']
        )
    return G, df_edges

def compute_basic_network_analysis(G, df_edges, output_dir="results"):
    """
    Computes rigorous Social Network Analysis (SNA) statistics on graph G.
    """
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    total_degrees = [d for n, d in G.degree()]
    
    # Connected components
    wcc = list(nx.weakly_connected_components(G))
    scc = list(nx.strongly_connected_components(G))
    largest_wcc = max(wcc, key=len) if wcc else set()
    largest_scc = max(scc, key=len) if scc else set()
    
    # Reciprocity
    reciprocity = nx.reciprocity(G)
    
    # Clustering coefficient (Transitivity)
    clustering_coeff = nx.transitivity(G)
    
    # Degree assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = np.nan
        
    metrics = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "mean_in_degree": float(np.mean(in_degrees)),
        "max_in_degree": int(np.max(in_degrees)),
        "min_in_degree": int(np.min(in_degrees)),
        "mean_out_degree": float(np.mean(out_degrees)),
        "max_out_degree": int(np.max(out_degrees)),
        "min_out_degree": int(np.min(out_degrees)),
        "mean_total_degree": float(np.mean(total_degrees)),
        "max_total_degree": int(np.max(total_degrees)),
        "min_total_degree": int(np.min(total_degrees)),
        "num_weakly_connected_components": len(wcc),
        "largest_wcc_size": len(largest_wcc),
        "wcc_coverage_ratio": len(largest_wcc) / num_nodes if num_nodes > 0 else 0,
        "num_strongly_connected_components": len(scc),
        "largest_scc_size": len(largest_scc),
        "scc_coverage_ratio": len(largest_scc) / num_nodes if num_nodes > 0 else 0,
        "reciprocity": reciprocity,
        "clustering_coefficient_transitivity": clustering_coeff,
        "degree_assortativity": assortativity
    }
    
    # Save table
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(output_dir, "tables", "basic_network_metrics.csv"), index=False)
    
    # Generate Visualizations
    # 1. Degree Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(in_degrees, ax=axes[0], kde=True, color='skyblue', bins=30)
    axes[0].set_title("In-Degree Distribution")
    axes[0].set_xlabel("In-Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale('log')
    
    sns.histplot(out_degrees, ax=axes[1], kde=True, color='salmon', bins=30)
    axes[1].set_title("Out-Degree Distribution")
    axes[1].set_xlabel("Out-Degree")
    axes[1].set_ylabel("Count")
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "degree_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Network Sample Visualization (Top-50 hubs subgraph)
    top_hubs = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:50]]
    subgraph = G.subgraph(top_hubs)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    node_sizes = [G.degree(n) * 15 for n in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='teal', alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray', arrows=True)
    labels = {n: n.split('.')[0] if '.' in n else n for n in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
    plt.title("Sampled Subgraph: Top 50 Hub Websites (Node size proportional to degree)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "network_subgraph_sample.png"), dpi=300)
    plt.close()
    
    return metrics

if __name__ == "__main__":
    G, df_edges = build_network_from_edges()
    metrics = compute_basic_network_analysis(G, df_edges)
    print("Basic Network Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
