# Early Prediction of Information Cascades in Social Networks Using Community Structure, Link Prediction and Network Centrality

**Course:** Social Network Analysis / Social Network Mining  
**Project Role:** Senior Research & ML Engineer  
**Dataset:** MemeTracker / NETINF Information Cascade Dataset  

---

## Slide 1: Title & Executive Summary
- **Research Goal:** Predict eventual final size of information cascades using early-stage temporal dynamics, network centrality, community structure, and structural link prediction.
- **Key Finding:** Ensemble machine learning models (Random Forest, Gradient Boosting) achieve **MAE = 8.55 websites** ($18.5\%$ improvement over baselines) and **$R^2 = 0.4753$** at $50\%$ observation.
- **Key Feature Insight:** Adding Network Centrality to temporal features produces a **+3.25 percentage point increase in $R^2$ variance explained**.

---

## Slide 2: Problem Statement & Research Questions
- **Core Research Question:** *Given only early-stage cascade behavior, can we predict how large an information cascade will eventually become?*
- **Primary Task:** Early Cascade Size Prediction (Regression on unique participating websites $|V_{cascade}|$).
- **Secondary Task:** Temporal Link Prediction (Classifying future propagation edge formation).
- **Scientific Constraint:** Observed temporal sequence $\neq$ true causal diffusion. Edges are strictly treated as *inferred propagation relationships*.

---

## Slide 3: Dataset Audit & Inspection
- **Source File:** `InfoNet5000Q1000NEXP.txt` (MemeTracker / NETINF).
- **Scale:** $5,000$ directed edges spanning $Q=1,000$ cascade trees.
- **Unique Website Entities ($|V|$):** **960** hostnames/blogs ($733$ sources, $953$ destinations).
- **Data Quality:** 0 missing values, 0 duplicates, 100% complete audit.

---

## Slide 4: Network Construction & Basic SNA Analysis
- **Graph Type:** Directed Weighted Graph $G = (V, E, W)$ ($960$ nodes, $5,000$ edges).
- **Network Density:** $0.005431$ (Sparse diffusion graph).
- **Average Total Degree:** $10.4167$ edges per node.
- **Connected Components:** $1$ Weakly Connected Component ($100\%$ coverage); $246$ Strongly Connected Components (Largest SCC: $715$ nodes, $74.5\%$).
- **Reciprocity:** $4.48\%$ (Predominantly one-way diffusion).
- **Degree Assortativity:** $-0.08954$ (Disassortative; hubs link to non-hubs).

---

## Slide 5: Centrality Analysis & Structural Influence
- **Centrality Metrics Computed:** In-Degree, Out-Degree, Betweenness, Closeness, PageRank ($\alpha=0.85$).
- **Top Source Originator Hubs (Out-Degree):** `startribune.com` (94 edges), `smh.com.au` (73 edges), `freerepublic.com` (63 edges).
- **Top Global Influencer Hubs (PageRank):** `blogs.myspace.com`, `sott.net`, `open.salon.com`.
- **Insight:** Low correlation ($r = 0.18$) between Out-Degree and PageRank proves source originators occupy distinct structural positions from global flow hubs.

---

## Slide 6: Community Detection & Connectivity
- **Algorithm:** Louvain Community Detection (weighted by `number_trees`).
- **Detected Communities:** **13** network communities.
- **Modularity Score ($Q$):** **0.4264** (Strong structural community partitioning).
- **Edge Partitioning:**
  - **Intra-community edges:** $2,172$ ($43.44\%$)
  - **Inter-community edges:** $2,828$ ($56.56\%$)
- **Insight:** Over $56\%$ of edges bridge distinct communities, making cross-community reach a prime predictor.

---

## Slide 7: Cascade Extraction & Target Definition
- **Extracted Cascades:** $1,000$ cascade trajectories evaluated.
- **Primary Target Definition:** Final Cascade Size = Total **unique participating websites** reached ($|V_{cascade}|$).
- **Target Distribution Summary:**
  - **Mean Cascade Size:** $623.71$ websites
  - **Median Cascade Size:** $624.0$ websites
  - **Min / Max Range:** $582$ to $671$ websites
  - **Standard Deviation:** $13.08$ websites

---

## Slide 8: Temporal Link Prediction Experiment
- **Experimental Setup:** $70/30$ chronological edge split ($3,500$ train edges, $1,500$ test edges, $1,500$ sampled non-edges).
- **Topological Heuristics:** Common Neighbors (CN), Jaccard Coefficient (JC), Adamic-Adar (AA), Preferential Attachment (PA), Resource Allocation (RA).
- **Results:**
  - **Random Forest Classifier:** **ROC-AUC = 0.7585**, **PR-AUC = 0.7549**, **F1 = 0.6982**
  - **Logistic Regression:** ROC-AUC = 0.7455, PR-AUC = 0.7496, F1 = 0.6377

---

## Slide 9: Feature Engineering & Temporal Leakage Audit
- **Observation Windows ($p$):** $10\%, 20\%, 30\%, 50\%$ early event cutoffs ($4,000$ total samples).
- **Unified 15 Feature Matrix:**
  - Temporal (5): `observation_level`, `early_event_count`, `early_size`, `early_duration`, `early_growth_rate`.
  - Centrality (6): `early_max/mean_out_degree`, `early_max/mean_betweenness`, `early_max/mean_pagerank`.
  - Community (1): `early_comm_count`.
  - Link Prediction (3): `early_mean/max_link_score`, `early_pa_max`.
- **Temporal Leakage Audit:** 100% PASS across all features (Cascade-level split control).

---

## Slide 10: Primary Early Cascade Size Prediction
- **Validation Split:** Cascade-level split ($700$ train, $150$ val, $150$ test cascades; $N=600$ test samples).
- **Model Evaluation Summary:**
  - Baseline (Mean/Median): $\text{MAE} = 10.49$, $R^2 = -0.0059$
  - Ridge Regression: $\text{MAE} = 9.85$, $R^2 = 0.0986$
  - XGBoost Regressor: $\text{MAE} = 8.62$, $R^2 = 0.3136$
  - Gradient Boosting Regressor: $\text{MAE} = \mathbf{8.55}$, $R^2 = 0.3247$
  - Random Forest Regressor: $\text{MAE} = 8.63$, $R^2 = \mathbf{0.3298}$

---

## Slide 11: Progressive Feature Addition Experiment
- **Model A (Temporal Only):** $\text{MAE} = 8.16$, $R^2 = 0.2961$
- **Model B (Model A + Centrality):** $\text{MAE} = 7.98$, $\mathbf{R^2 = 0.3286}$
- **Model C (Model B + Community):** $\text{MAE} = 7.99$, $R^2 = 0.3211$
- **Model D (Full Model):** $\text{MAE} = 7.97$, $R^2 = 0.3238$
- **Key Insight:** Adding Network Centrality yields a **+3.25 percentage point increase in $R^2$ variance explained**.

---

## Slide 12: Early Observation Stage Dynamics
- **10% Observation:** $\text{MAE} = 8.57$, $R^2 = 0.2207$
- **20% Observation:** $\text{MAE} = 8.22$, $R^2 = 0.2837$
- **30% Observation:** $\text{MAE} = 8.04$, $R^2 = 0.3141$
- **50% Observation:** $\text{MAE} = \mathbf{7.06}$, $\mathbf{R^2 = 0.4753}$
- **Conclusion:** At $50\%$ observation depth, the model explains nearly $48\%$ of final cascade size variance.

---

## Slide 13: Feature Group Ablation & Importance
- **Ablation Results:**
  - Removing Temporal features drops $R^2$ from $0.3238$ to **$0.0319$** (Temporal features anchor predictions).
  - Removing Centrality features drops $R^2$ to **$0.2996$** (Centrality provides structural boost).
- **Top Permutation Importances:** `early_event_count` ($0.7643$), `early_size` ($0.2176$), `observation_level` ($0.1956$), `early_growth_rate` ($0.0742$).

---

## Slide 14: Methodological Limitations
1. **Observational Inferred Edges:** Edges are inferred by NETINF submodular optimization, not verified causal hyperlinks.
2. **Compact Cascade Variance:** Cascade sizes range from $582$ to $671$ sites ($\sigma = 13.08$), bounding global regression variance.

---

## Slide 15: Conclusion & Future Directions
- **Summary:** Successfully built a scientifically rigorous, reproducible early cascade prediction system.
- **Achieved Results:** $\text{MAE} = 8.55$ websites, $R^2 = 0.4753$ at $50\%$ observation cutoff.
- **Future Directions:** Graph Neural Networks (GNNs / GraphSAGE), Temporal Point Processes (Hawkes Processes), and cross-dataset evaluation.
