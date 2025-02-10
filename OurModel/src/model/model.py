import numpy as np
import pandas as pd
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from stock_encoding import StockEncoding
from stock_clustering import DynamicStockClustering

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.loss import get_loss, weighted_kl_divergence
from utils.evaluator import evaluate

market_name = 'NASDAQ'

stock_num = 1026
market_num = 11

valid_index = 756
test_index = 1008
bins_class = 100
hidden_dim = 40
learning_rate = 0.001
bins = np.load(f'../../data/KL-Data/NASDAQ_{bins_class}.npy')

# Dynamic Stock Clustering parameters
n_clusters = 12
n_subclusters = 2

# Stock Encoding parameters
lookback_length = 16
fea_num = 5
steps = 1

# Multi-head Attention parameters
dropout_prob = 0.3
layer_norm_eps = 1e-12
n_heads = 4

similarity_thresholds = [(i+1)/n_clusters for i in range(n_clusters-1)].reverse()

# Training parameters
epochs = 30
patience = 5
min_delta = 0

class model(nn.Module):
    """
    Main model architecture combining three key components:
    1. Stock Encoding
    2. Dynamic Stock Clustering
    3. Gating Mechanism
    """
    def __init__(self, stock_num, lookback_length, fea_num, market_num, n_clusters, n_subclusters, hidden_dim):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_subclusters = n_subclusters

        # Stock Encoding component
        self.stock_encoding = StockEncoding(
            stocks=stock_num,
            time_steps=lookback_length,
            channels=fea_num,
            market=market_num,
        )
        
        # Dynamic Stock Clustering component
        self.stock_clustering = DynamicStockClustering(
            n_clusters=n_clusters,
            n_subclusters=n_subclusters,
            hidden_dim=hidden_dim
        )
        
        # Hidden dimension for representations
        self.hidden_dim = hidden_dim
        
        # Combines representations from different sub-clusters
        self.subcluster_combine = nn.Sequential(
            nn.Linear(hidden_dim * n_subclusters, hidden_dim)
        )

        # MLP for generating local prediction ĉ^l_i from stock encoding
        self.encoding_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # MLP for generating cluster-based prediction ĉ^c_i from clustering
        self.clustering_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Gating mechanism
        # Generates α_i to combine local and cluster-based predictions
        self.gate_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1),
            nn.Sigmoid()
        )

        # Learnable weights for composite loss
        self.reg_loss_weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.rank_loss_weight = nn.Parameter(torch.tensor(10.0, requires_grad=True))
        self.kl_loss_weight = nn.Parameter(torch.tensor(0.005, requires_grad=True))  
        self.ic_weight = nn.Parameter(torch.tensor(0.05, requires_grad=True))

    def forward(self, data_batch):
        """
        Forward pass of the model:
        1. Get representations from stock encoding
        2. Perform dynamic stock clustering
        3. Generate and combine predictions using gate mechanism
        """
        # Get encoded representations
        _, encoding_reps, market_reps, stock_reps = self.stock_encoding(data_batch)
        
        # Get clustering-based representations
        clustering_reps, cluster_indices, market_stock_similarities = self.stock_clustering(stock_reps, market_reps)

        # Generate local and cluster-based predictions
        o_local = self.encoding_mlp(encoding_reps)      # ĉ^l_i in paper
        o_cluster = self.clustering_mlp(clustering_reps) # ĉ^c_i in paper

        # Compute gating parameter α_i
        gate_input = torch.cat([encoding_reps, clustering_reps], dim=-1)
        gate = self.gate_network(gate_input)
        
        # Final prediction: α_i * ĉ^l_i + (1-α_i) * ĉ^c_i
        final_prediction = gate * o_local + (1 - gate) * o_cluster
        
        return final_prediction, cluster_indices, market_stock_similarities, gate