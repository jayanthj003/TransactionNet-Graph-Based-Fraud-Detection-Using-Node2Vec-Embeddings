# TransactionNet-Graph-Based-Fraud-Detection-Using-Node2Vec-Embeddings
Graph-based fraud detection system using Node2Vec embeddings and KMeans clustering to identify behavioral patterns in financial transaction networks.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-2.6+-orange.svg)](https://networkx.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A machine learning project that leverages Node2Vec embeddings to detect fraudulent patterns in financial transaction networks through unsupervised clustering.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)

## 🎯 Overview

TransactionNet represents financial transactions as a directed weighted graph and applies Node2Vec algorithm to generate vector embeddings based on neighborhood and transaction behavior. The embeddings are then clustered to identify hidden communities and behavioral patterns within the network.

**Applications:**
- Fraud Detection
- Customer Segmentation
- Network Analysis
- Behavioral Pattern Recognition

## ✨ Features

- **Graph Construction**: Builds directed weighted graphs from transaction data
- **Node2Vec Embeddings**: Generates 128-dimensional vector representations using biased random walks
- **Unsupervised Clustering**: Applies KMeans to identify behavioral clusters
- **Visualization**: PCA-based 2D visualization of high-dimensional embeddings
- **Evaluation Metrics**: Silhouette score for cluster quality assessment

## 🔬 Algorithm

### Core Pipeline:

1. **Graph Construction**
   - Parse transaction data into directed weighted graph
   - Nodes: Senders/Receivers
   - Edges: Transactions (weighted by amount)

2. **Random Walks (Node2Vec)**
   - Simulate biased random walks to capture network structure
   - Parameters: `walk_length=10`, `num_walks=100`, `p=1`, `q=1`

3. **Embedding Learning**
   - Train Skip-Gram model on random walks
   - Generate 128-dimensional embeddings per node

4. **Clustering**
   - Apply KMeans clustering on embeddings
   - Determine optimal clusters using elbow method/silhouette score

5. **Visualization & Analysis**
   - PCA dimensionality reduction for 2D visualization
   - Cluster analysis and interpretation

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
```

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/TransactionNet.git
cd TransactionNet
```

2. Install required packages
```bash
pip install -r requirements.txt
```

### Required Libraries
```
networkx==2.6.3
numpy==1.21.0
pandas==1.3.0
torch==1.9.0
scikit-learn==0.24.2
matplotlib==3.4.2
```

## 💻 Usage

### Basic Usage

```python
from graph_builder import build_graph_from_csv
from node2vec import get_all_walks, train_skipgram
from clustering import perform_clustering

# Build graph from transaction data
graph = build_graph_from_csv('transactions.csv')

# Generate random walks
walks = get_all_walks(graph, num_walks=100, walk_length=10)

# Train Skip-Gram model and get embeddings
embeddings = train_skipgram(walks, embedding_dim=128, epochs=10)

# Perform clustering
clusters = perform_clustering(embeddings, n_clusters=4)
```

### Running the Complete Pipeline

```bash
python main.py --input transactions.csv --clusters 4 --visualize
```

### Command Line Arguments
- `--input`: Path to transaction CSV file
- `--clusters`: Number of clusters (default: 4)
- `--dimensions`: Embedding dimensions (default: 128)
- `--walks`: Number of walks per node (default: 100)
- `--walk-length`: Length of each walk (default: 10)
- `--visualize`: Generate PCA visualization

## 📁 Project Structure

```
TransactionNet/
│
├── data/
│   └── transactions.csv          # Sample transaction data
│
├── src/
│   ├── graph_builder.py          # Graph construction functions
│   ├── node2vec.py                # Node2Vec implementation
│   ├── skipgram.py                # Skip-Gram model (PyTorch)
│   ├── clustering.py              # KMeans clustering
│   └── visualization.py           # PCA and plotting utilities
│
├── notebooks/
│   └── analysis.ipynb             # Jupyter notebook for exploration
│
├── results/
│   ├── embeddings.npy             # Saved embeddings
│   └── clusters_visualization.png # PCA plot
│
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── report.pdf                     # Detailed project report
```

## 📊 Results

The Node2Vec clustering successfully identified **4 distinct behavioral clusters** in the transaction network:

- **Cluster 0 (Blue)**: Bridge entities connecting multiple communities
- **Cluster 1 (Orange)**: Distinct behavioral group with unique transaction patterns
- **Cluster 2 (Green)**: Extended network with gradual behavioral variations
- **Cluster 3 (Red)**: Dense community with highly similar transaction behaviors

### Performance Metrics
- **Silhouette Score**: [Your score here]
- **Number of Nodes**: [Your number]
- **Embedding Dimensions**: 128
- **Clear cluster separation achieved through PCA visualization**

![Cluster Visualization](results/clusters_visualization.png)

## 🛠️ Technologies Used

- **Python**: Core programming language
- **NetworkX**: Graph construction and manipulation
- **PyTorch**: Neural network implementation for Skip-Gram model
- **Scikit-learn**: KMeans clustering and PCA
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization

## 👥 Contributors

- **Divyanshu Ranjan** (CS24MTECH11013)
- **Jayanth Jatavath** (CS24MTECH11014)
- **Vaibhav Barsaiyan** (CS24MTECH12008)

**Course**: CS 6890 - Fraud Analytics Using Predictive and Social Network Techniques  
**Instructor**: Dr. Sobhan Babu

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course instructors and TAs for guidance
- Node2Vec paper by Grover & Leskovec (2016)
- NetworkX and PyTorch communities

