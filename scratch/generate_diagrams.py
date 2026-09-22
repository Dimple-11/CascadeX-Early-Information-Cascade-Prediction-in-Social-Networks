import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_pipeline_diagram():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    
    steps = [
        ("Dataset Inspection & Preprocessing", "MemeTracker / NETINF Edge Dataset (960 Nodes, 5,000 Edges)\nData Quality & Missing Value Audit"),
        ("Graph Construction & Topological Analysis", "Directed Weighted Graph G=(V,E,W)\nDegree Statistics, Density, Reciprocity & Assortativity"),
        ("Centrality Analysis & Network Influence", "In/Out-Degree, Betweenness, Closeness, PageRank\nIdentification of Originators vs. Global Flow Hubs"),
        ("Community Detection & Structure Analysis", "Louvain Algorithm (13 Communities, Modularity Q=0.4264)\nIntra-Community (43.4%) vs Inter-Community (56.6%) Edges"),
        ("Cascade Extraction & Target Definition", "1,000 Extracted Cascades\nPrimary Target Variable: Final Cascade Size (|V_cascade|)"),
        ("Temporal Representation & Leakage Audit", "Early Observation Cutoffs (10%, 20%, 30%, 50%)\nStrict Temporal Feature Isolation Audit (100% Pass)"),
        ("Temporal Link Prediction Experiment", "70/30 Temporal Train/Test Edge Split (CN, JC, AA, PA, RA)\nRandom Forest ROC-AUC = 0.7585, PR-AUC = 0.7549"),
        ("ML Early Cascade Size Prediction", "Cascade-Level Split (70/15/15), Ridge, RF, GBR, XGBoost\nRandom Forest R² = 0.3298, MAE = 8.63 websites"),
        ("Progressive Feature & Observation Dynamics", "Model A -> D Feature Additions (+3.25% R² from Centrality)\n50% Observation Window achieves R² = 0.4753"),
        ("Ablation Study & Systematic Error Analysis", "Temporal Features Anchor (R² Drops to 0.0319 Without Them)\nOutlier Diagnostic & Error Distribution Analysis")
    ]
    
    y_pos = 0.92
    box_height = 0.065
    box_width = 0.85
    x_left = 0.075
    
    colors = ['#2b5c8f', '#2b7b8f', '#2b8f74', '#3d8f2b', '#6c8f2b', '#8f882b', '#8f682b', '#8f462b', '#8f2b46', '#722b8f']
    
    for idx, (title, desc) in enumerate(steps):
        rect = patches.FancyBboxPatch(
            (x_left, y_pos - box_height), box_width, box_height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            ec="none", fc=colors[idx % len(colors)], alpha=0.9
        )
        ax.add_patch(rect)
        
        ax.text(x_left + 0.03, y_pos - 0.02, f"Phase {idx+1}: {title}", fontsize=11, fontweight='bold', color='white', va='center')
        ax.text(x_left + 0.03, y_pos - 0.048, desc, fontsize=9, color='white', va='center')
        
        if idx < len(steps) - 1:
            ax.annotate('', xy=(0.5, y_pos - box_height - 0.015), xytext=(0.5, y_pos - box_height),
                        arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5))
            
        y_pos -= (box_height + 0.025)
        
    plt.title("End-to-End Research & Experimental Pipeline Architecture", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("results/figures/methodology_pipeline.png", dpi=300)
    plt.savefig("paper/figures/methodology_pipeline.png", dpi=300)
    plt.close()
    print("Generated methodology pipeline diagram.")

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis('off')
    
    # Layer boxes
    layers = [
        ("Data Layer", "data/raw/InfoNet5000Q1000NEXP.txt\n--> data/processed/network_edges.csv", 0.80, '#1f77b4'),
        ("Network & Topological Layer", "Graph G=(V,E,W) | Centrality (PageRank/Degree) | Louvain Communities (Q=0.426)", 0.60, '#2ca02c'),
        ("Link Prediction Layer", "Temporal 70/30 Split | CN, JC, AA, PA, RA Structural Heuristics | RF ROC-AUC=0.758", 0.40, '#ff7f0e'),
        ("Cascade & Feature Engine", "Cascade Trajectories | 10%-50% Windows | 15 Unified Features | Leakage Audit", 0.20, '#9467bd'),
        ("ML Prediction & Evaluation", "Baselines vs. Ridge / RF / GBR / XGBoost | Progressive & Ablation Experiments", 0.00, '#d62728')
    ]
    
    for title, desc, y_val, col in layers:
        rect = patches.FancyBboxPatch(
            (0.1, y_val), 0.8, 0.12,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            ec="black", fc=col, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(0.5, y_val + 0.08, title, fontsize=12, fontweight='bold', color='white', ha='center', va='center')
        ax.text(0.5, y_val + 0.04, desc, fontsize=9.5, color='white', ha='center', va='center')
        
    plt.title("Software & System Modular Architecture", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("results/figures/architecture_diagram.png", dpi=300)
    plt.savefig("paper/figures/architecture_diagram.png", dpi=300)
    plt.close()
    print("Generated architecture diagram.")

if __name__ == "__main__":
    generate_pipeline_diagram()
    generate_architecture_diagram()
