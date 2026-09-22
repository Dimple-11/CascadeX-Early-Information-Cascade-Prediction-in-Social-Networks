# Early Prediction of Information Cascades in Social Networks Using Community Structure, Link Prediction and Network Centrality

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains a reproducible Social Network Analysis (SNA) and Machine Learning pipeline for studying and predicting the eventual size of information cascades in a web-media network.

The project combines:

- Social Network Analysis
- Network centrality
- Community detection
- Temporal cascade analysis
- Structural link prediction
- Early cascade feature engineering
- Machine learning-based cascade size prediction
- Progressive feature experiments
- Early-observation experiments
- Ablation studies
- Error analysis

---

## 📌 Research Overview

### Research Question

> **Can network centrality, community structure, link-prediction features, and early temporal information be used to predict the eventual size of an information cascade in a web-media network?**

### Primary Prediction Task

The primary task is **early prediction of final cascade size**.

For each cascade, information available during an early observation period is used to predict the eventual number of unique participating websites.

The primary target variable is:

```text
target_final_size
````

where:

$$
target\_final\_size = |V_{cascade}|
$$

and \(V_{cascade}\) represents the set of unique websites/nodes observed in the complete cascade.

### Secondary Tasks

The project also investigates:

1. Temporal structural link prediction
2. Network centrality and cascade growth
3. Community-level cascade propagation
4. Temporal growth characteristics of cascades
5. The contribution of different feature groups to prediction performance

---

## ⚠️ Scientific Interpretation of Propagation Edges

An important methodological distinction is maintained throughout this project.

The network edges in the NETINF-based dataset represent **inferred propagation relationships produced by the NETINF optimization procedure**.

Therefore, an edge:

```text
A → B
```

should not automatically be interpreted as proof that website A directly caused website B to publish the information.

The project distinguishes between:

* observed cascade events
* temporal ordering of events
* inferred propagation relationships
* structural network relationships

Consequently, the analysis does **not claim causal relationships** from temporal ordering alone.

---

# 📊 Dataset

### Dataset

**MemeTracker / NETINF Information Cascade Dataset**

Primary dataset file:

```text
InfoNet5000Q1000NEXP.txt
```

The dataset contains information-diffusion observations involving web media sources and inferred propagation relationships.

The processed network used by the current implementation contains:

* **960 unique web-media domain nodes**
* **5,000 directed weighted propagation edges**
* **1,000 extracted cascade trajectories**

> These statistics should be regenerated and verified from the preprocessing pipeline before being treated as final experimental statistics.

### Dataset Processing

The raw dataset is preserved without modification.

Processing includes:

1. Dataset parsing
2. Schema and data-quality validation
3. Node and edge extraction
4. Duplicate handling where applicable
5. Graph construction
6. Cascade extraction
7. Temporal ordering
8. Feature generation

All processed datasets are stored separately from the raw dataset.

---

# 🏗 Repository Structure

```text
project/
│
├── data/
│   ├── raw/
│   │   └── InfoNet5000Q1000NEXP.txt
│   │
│   └── processed/
│       ├── network_edges.csv
│       ├── centrality_scores.csv
│       ├── community_assignments.csv
│       ├── node_cascade_participation.csv
│       ├── cascade_dataset.csv
│       ├── temporal_cascade_features.csv
│       └── final_cascade_feature_matrix.csv
│
├── src/
│   ├── preprocessing/
│   │   └── Dataset loading and validation
│   │
│   ├── network/
│   │   └── Graph construction and network analysis
│   │
│   ├── centrality/
│   │   └── Network centrality calculations
│   │
│   ├── community/
│   │   └── Community detection
│   │
│   ├── cascades/
│   │   └── Cascade extraction and target construction
│   │
│   ├── temporal/
│   │   └── Temporal cascade analysis
│   │
│   ├── link_prediction/
│   │   └── Temporal link prediction
│   │
│   ├── features/
│   │   └── Cascade feature engineering and leakage checks
│   │
│   ├── prediction/
│   │   └── ML models and evaluation
│   │
│   └── experiments/
│       └── Progressive, early-observation, ablation and error analysis
│
├── notebooks/
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_basic_network_analysis.ipynb
│   ├── 04_centrality_analysis.ipynb
│   ├── 05_community_detection.ipynb
│   ├── 06_cascade_extraction.ipynb
│   ├── 07_temporal_cascade_analysis.ipynb
│   ├── 08_link_prediction.ipynb
│   ├── 09_cascade_feature_engineering.ipynb
│   ├── 10_early_cascade_prediction.ipynb
│   ├── 11_progressive_feature_experiment.ipynb
│   ├── 12_early_observation_experiment.ipynb
│   ├── 13_ablation_study.ipynb
│   └── 14_error_analysis.ipynb
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── link_prediction/
│   └── experiments/
│
├── paper/
│   ├── main.tex
│   └── references.bib
│
├── presentation/
│   └── presentation_slides.md
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

### Programming

* Python 3.10+
* Jupyter Notebook

### Data Processing

* NumPy
* pandas

### Network Analysis

* NetworkX
* Community detection algorithms such as Louvain, where applicable

### Machine Learning

* scikit-learn
* XGBoost, where applicable

### Visualization

* Matplotlib
* Seaborn
* Plotly, where applicable

---

# ⚡ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Dimple-11/CascadeX-Early-Information-Cascade-Prediction-in-Social-Networks.git
cd CascadeX-Early-Information-Cascade-Prediction-in-Social-Networks
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Pipeline

The research pipeline is executed in the following order:

### 1. Data preprocessing and network construction

```bash
python src/preprocessing/loader.py
python src/network/graph_builder.py
```

### 2. Centrality and community detection

```bash
python src/centrality/centrality_calculator.py
python src/community/community_detector.py
```

### 3. Cascade and temporal analysis

```bash
python src/cascades/cascade_extractor.py
python src/temporal/temporal_analyzer.py
```

### 4. Link prediction

```bash
python src/link_prediction/evaluation.py
```

### 5. Feature engineering and prediction

```bash
python src/features/cascade_features.py
python src/prediction/training.py
```

### 6. Experimental analysis

```bash
python src/experiments/progressive_features.py
python src/experiments/early_observation.py
python src/experiments/ablation.py
python src/experiments/error_analysis.py
```

---

# 🔬 Methodology

The complete methodology consists of the following stages:

```text
Raw Dataset
     ↓
Data Validation & Preprocessing
     ↓
Network Construction
     ↓
Basic Network Analysis
     ↓
Centrality Analysis
     ↓
Community Detection
     ↓
Cascade Extraction
     ↓
Temporal Cascade Analysis
     ↓
Temporal Link Prediction
     ↓
Cascade Feature Engineering
     ↓
Early Cascade Size Prediction
     ↓
Progressive Feature Experiment
     ↓
Early Observation Experiment
     ↓
Ablation Study
     ↓
Error Analysis
```

---

## 1. Network Construction

A directed weighted network is constructed from the processed propagation relationships.

Nodes represent web-media sources, while edges represent inferred propagation relationships.

Network properties such as:

* number of nodes
* number of edges
* in-degree
* out-degree
* density
* connected components
* clustering
* reciprocity
* edge-weight distribution

are analyzed where applicable to the constructed graph.

---

## 2. Network Centrality

The project evaluates structural importance using appropriate centrality measures, including:

* Degree centrality
* Betweenness centrality
* Closeness centrality
* Eigenvector centrality, where applicable
* PageRank

These features are subsequently investigated as potential predictors of cascade growth.

---

## 3. Community Detection

Community structure is identified using a community-detection algorithm such as Louvain.

The analysis considers:

* number of communities
* community sizes
* modularity
* cascade-community participation
* intra-community and inter-community propagation

Community-level characteristics are incorporated into the cascade prediction experiments where appropriate.

---

## 4. Cascade Extraction

For every identified cascade, the pipeline extracts information such as:

* cascade identifier
* start time
* end time
* cascade duration
* number of observed events
* number of unique participating websites
* temporal event sequence
* participating communities

The final number of unique participating websites is used as the primary prediction target.

---

## 5. Temporal Cascade Features

Early-stage information is used to construct features such as:

* early cascade size
* event count
* temporal growth rate
* inter-arrival characteristics
* cascade duration information available at observation time
* centrality summaries
* community summaries
* structural link-prediction features

Only information available up to the selected observation point should be used for prediction.

---

# 🔗 Link Prediction

The project investigates structural link prediction using measures including:

* Common Neighbors
* Jaccard Coefficient
* Adamic-Adar
* Preferential Attachment
* Resource Allocation, where applicable

The link-prediction experiment is performed using a temporal setup in which earlier observations are used to predict future relationships.

Candidate generation is restricted to feasible node pairs rather than evaluating every possible pair when the graph size makes exhaustive enumeration impractical.

Evaluation metrics include:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

---

# 🤖 Early Cascade Size Prediction

The primary machine learning task is a **regression problem**.

The target is:

```text
Final number of unique websites participating in the cascade
```

The prediction models investigated include:

* Mean baseline
* Median baseline
* Naive early-size baseline
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor, where applicable

The primary evaluation metrics are:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* \(R^2\)

Additional error analyses may include:

* Median Absolute Error
* Percentage of predictions within specified error ranges
* Actual vs. predicted plots
* Residual distributions
* Error as a function of actual cascade size

Because this is a regression task, classification accuracy and a confusion matrix are not used as the primary evaluation measures.

---

# 📊 Experimental Design

## Progressive Feature Experiment

The contribution of different feature groups is evaluated progressively.

```text
Model A
Early + Temporal Features

        ↓

Model B
Model A + Network Centrality

        ↓

Model C
Model B + Community Features

        ↓

Model D
Model C + Link Prediction Features
```

This experiment evaluates whether additional structural information provides measurable improvements over early temporal information alone.

---

## Early Observation Experiment

Prediction performance is evaluated at different stages of cascade observation.

The current experimental stages include:

```text
10%
20%
30%
50%
```

The purpose is to investigate the trade-off between:

* amount of cascade information available
* prediction error
* timeliness of prediction

---

# 📈 Current Experimental Results

The following results are from the current implementation and should be regenerated whenever the pipeline or preprocessing changes.

## Primary Prediction Results

| Model / Baseline            |    MAE |   RMSE |      R² |
| --------------------------- | -----: | -----: | ------: |
| Mean Prediction             |  10.49 |  13.32 | -0.0059 |
| Naive Early Size            | 350.78 | 368.42 | -768.62 |
| Ridge Regression            |   9.85 |  12.61 |  0.0986 |
| XGBoost Regressor           |   8.62 |  11.00 |  0.3136 |
| Gradient Boosting Regressor |   8.55 |  10.91 |  0.3247 |
| Random Forest Regressor     |   8.63 |  10.87 |  0.3298 |

The reported experiments show that the evaluated machine-learning models reduce prediction error relative to the mean baseline on the current test split.

The Random Forest model has the highest reported \(R^2\) and lowest reported RMSE among the models in this table.

These values describe the current experimental configuration and should not be interpreted as universal performance estimates for other datasets or splits.

---

# 📊 Progressive Feature Results

| Model   | Feature Groups            |  MAE |     R² |
| ------- | ------------------------- | ---: | -----: |
| Model A | Early + Temporal          | 8.16 | 0.2961 |
| Model B | Model A + Centrality      | 7.98 | 0.3286 |
| Model C | Model B + Community       | 7.99 | 0.3211 |
| Model D | Model C + Link Prediction | 7.97 | 0.3238 |

The progressive experiment allows the contribution of different feature groups to be examined rather than relying only on the final model.

---

# ⏱️ Early Observation Results

| Observation | Mean Early Size |  MAE |     R² |
| ----------: | --------------: | ---: | -----: |
|         10% |           200.7 | 8.57 | 0.2207 |
|         20% |           308.2 | 8.22 | 0.2837 |
|         30% |           390.6 | 8.04 | 0.3141 |
|         50% |           509.3 | 7.06 | 0.4753 |

These results represent the current experimental setup and should be interpreted together with the corresponding test-set construction and observation-window definitions.

---

# 🔒 Leakage Prevention

Temporal leakage is a critical concern in early cascade prediction.

The implementation is designed so that features used for prediction are restricted to information available up to the selected observation point.

The dataset is partitioned at the **cascade level** so that observations belonging to the same cascade are not distributed across training and test sets.

The intended split is:

```text
70% Training
15% Validation
15% Test
```

All preprocessing and feature-generation steps should preserve this temporal and cascade-level separation.

A dedicated leakage audit is included in the experimental pipeline.

---

# 🎲 Reproducibility

Experiments use fixed random seeds where stochastic procedures are involved.

The current configuration uses:

```text
random_state = 42
```

for applicable train/test splits, sampling procedures and machine-learning models.

Exact reproducibility additionally depends on:

* Python version
* package versions
* dataset version
* preprocessing configuration
* model hyperparameters
* hardware/software environment

The dependency versions used by the project are specified in:

```text
requirements.txt
```

---

# 📁 Generated Outputs

The pipeline generates:

### Processed datasets

```text
data/processed/
```

### Figures

```text
results/figures/
```

### Tables

```text
results/tables/
```

### Link prediction results

```text
results/link_prediction/
```

### Experimental results

```text
results/experiments/
```

---

# 🧪 Validation and Testing

The project includes validation procedures for:

* dataset integrity
* graph construction
* cascade extraction
* temporal ordering
* centrality calculations
* community detection
* link prediction
* feature leakage
* train/test separation
* model performance
* early-observation experiments
* ablation experiments
* reproducibility

Before finalizing reported results, all experiments should be rerun from the processed data and their outputs checked for consistency.

---

# ⚠️ Limitations

The current study has several limitations:

1. The dataset represents a web-media propagation environment and may not generalize directly to other social-network platforms.

2. Propagation relationships are inferred rather than necessarily representing verified causal interactions.

3. Observational temporal ordering does not establish causality.

4. The final cascade-size distribution may be highly skewed, which can make prediction performance vary substantially across small and large cascades.

5. Link prediction performance depends on the definition of positive and negative future edges and the candidate-generation strategy.

6. The results depend on the selected observation windows, feature definitions and model configurations.

7. The current experiments evaluate a specific dataset and should not be interpreted as universal estimates of information-cascade predictability.

---

# 🔮 Future Work

Potential extensions include:

* Evaluation on additional information-diffusion datasets
* Alternative community-detection algorithms
* Graph embedding methods such as Node2Vec
* Temporal graph representation learning
* Graph neural networks
* Richer temporal point-process features
* Cross-network validation
* Uncertainty estimation
* Real-time cascade monitoring
* Larger-scale deployment

---

# 👥 Project Structure

This project was developed as an academic Social Network Analysis and Mining project with a focus on reproducible network analysis and machine-learning experiments.

---

# 📜 License

This project is intended for academic and research purposes.

Please refer to the original dataset source for dataset-specific licensing and usage conditions.
