import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention

class DynamicStockClustering(nn.Module):
    """
    Dynamic Stock Clustering Component
    Performs hierarchical clustering of stocks based on market similarities
    """
    def __init__(self, n_clusters, n_subclusters, hidden_dim):
        super().__init__()
        self.n_clusters = n_clusters          # Number of primary clusters K
        self.n_subclusters = n_subclusters    # Number of sub-clusters L
        self.hidden_dim = hidden_dim
        self.similarity_thresholds = None
        
        # Multi-head attention for processing each sub-cluster
        self.interval_attentions = nn.ModuleList([
            MultiHeadAttention(
                n_heads=4,                    # Number of attention heads H
                hidden_size=hidden_dim,
                hidden_dropout_prob=0.3,
                attn_dropout_prob=0.3,
                layer_norm_eps=1e-12
            ) for _ in range(n_subclusters)
        ])

        # Combines representations from sub-clusters
        self.subcluster_combine = nn.Sequential(
            nn.Linear(hidden_dim * n_subclusters, hidden_dim)
        )

    def calculate_similarities(self, stock_emb, market_rep):
        """
        Calculate cosine similarities between stocks and market representations
        """
        stock_emb_normalized = F.normalize(stock_emb, p=2, dim=-1)
        market_rep_normalized = F.normalize(market_rep, p=2, dim=-1)
        similarities = torch.mm(stock_emb_normalized, market_rep_normalized.t())
        return similarities.mean(dim=1)

    def update_thresholds(self, similarities):
        """
        Update similarity thresholds dynamically based on current batch
        Computes percentile-based thresholds for clustering
        """
        with torch.no_grad():
            sims = similarities.detach().cpu().numpy()
            valid_sims = sims[~np.isnan(sims)]
            percentiles = np.linspace(1, 99, self.n_clusters)
            thresholds = np.percentile(valid_sims, percentiles)
            self.similarity_thresholds = thresholds
            return self.similarity_thresholds.tolist()

    def dynamic_clustering(self, similarities):
        """
        Perform primary clustering based on market similarities
        Divides stocks into K clusters based on their similarity scores
        """
        batch_size = similarities.size(0)
        cluster_size = batch_size // self.n_clusters
        sorted_indices = torch.argsort(similarities, descending=False)
        cluster_indices = torch.zeros_like(similarities, dtype=torch.long)
        
        for i in range(self.n_clusters):
            start_idx = i * cluster_size
            end_idx = (i + 1) * cluster_size if i < self.n_clusters - 1 else batch_size
            cluster_indices[sorted_indices[start_idx:end_idx]] = self.n_clusters - 1 - i
        
        return cluster_indices

    def sub_clustering(self, encoded_stocks, cluster_indices):
        """
        Perform secondary clustering within each primary cluster
        Divides each cluster into L sub-clusters based on centroid similarities
        """
        N = encoded_stocks.size(0)
        interval_indices = torch.zeros(N, device=encoded_stocks.device, dtype=torch.long)
        
        for cluster_idx in range(self.n_clusters):
            cluster_mask = (cluster_indices == cluster_idx)
            if not cluster_mask.any():
                continue
                
            # Calculate centroid similarities within cluster
            cluster_stocks = encoded_stocks[cluster_mask]
            centroid = cluster_stocks.mean(dim=0, keepdim=True)
            stocks_normalized = F.normalize(cluster_stocks, p=2, dim=-1)
            centroid_normalized = F.normalize(centroid, p=2, dim=-1)
            similarities = torch.mm(stocks_normalized, centroid_normalized.t()).squeeze()
            
            # Assign sub-cluster indices based on similarity ranking
            sorted_sims, sort_indices = torch.sort(similarities, descending=False)
            interval_size = len(sorted_sims) // self.n_subclusters
            
            for i in range(self.n_subclusters):
                start_idx = i * interval_size
                end_idx = (i + 1) * interval_size if i < self.n_subclusters - 1 else len(sorted_sims)
                indices_in_interval = sort_indices[start_idx:end_idx]
                original_indices = torch.where(cluster_mask)[0][indices_in_interval]
                interval_indices[original_indices] = self.n_subclusters - 1 - i

        return interval_indices

    def create_attention_masks(self, cluster_indices, interval_indices, N, device):
        """
        Create attention masks for multi-head attention in sub-clusters
        Ensures attention is only applied between stocks in the same sub-cluster
        
        Args:
            cluster_indices: Primary cluster assignments
            interval_indices: Sub-cluster assignments
            N: Number of stocks
            device: Device for tensor operations
        """
        big_cluster = cluster_indices.view(N, 1).expand(N, N)
        big_interval = interval_indices.view(N, 1).expand(N, N)
        inf_tensor = torch.tensor(-1e9, device=device)
        
        attention_masks = []
        for k in range(self.n_subclusters):
            # Check if stocks are in same primary cluster
            same_cluster = (big_cluster == big_cluster.transpose(0, 1))
            # Check if stocks are in same sub-cluster k
            same_interval = (big_interval == k) & (big_interval.transpose(0, 1) == k)
            valid_positions = same_cluster & same_interval
            
            # Create mask: 0 for valid attention pairs, -inf for others
            attn_mask = torch.where(
                valid_positions,
                torch.zeros_like(valid_positions, dtype=torch.float32),
                inf_tensor
            )
            attention_masks.append(attn_mask.view(N, 1, 1, N))
            
        return attention_masks

    def forward(self, stock_reps, market_reps):
        """
        Forward pass of dynamic clustering:
        1. Calculate stock-market similarities
        2. Perform hierarchical clustering
        3. Apply attention within sub-clusters
        4. Combine sub-cluster representations
        """
        market_reps = market_reps.squeeze(0).transpose(0, 1)

        # Perform hierarchical clustering
        market_stock_similarities = self.calculate_similarities(stock_reps, market_reps)
        self.update_thresholds(market_stock_similarities)
        cluster_indices = self.dynamic_clustering(market_stock_similarities)
        interval_indices = self.sub_clustering(stock_reps, cluster_indices)
        
        # Create attention masks for sub-clusters
        N = stock_reps.size(0)
        attention_masks = self.create_attention_masks(
            cluster_indices, interval_indices, N, stock_reps.device
        )
        
        # Process each sub-cluster through attention
        interval_outputs = []
        for k in range(self.n_subclusters):
            interval_query = stock_reps.unsqueeze(1)
            interval_key = stock_reps.unsqueeze(0).expand(N, -1, -1)
            
            out_k = self.interval_attentions[k](
                interval_query,
                interval_key,
                attention_masks[k]
            ).squeeze(1)
            
            interval_outputs.append(out_k)
            
        # Combine sub-cluster outputs
        combined_interval = torch.cat(interval_outputs, dim=-1)
        clustering_reps = self.subcluster_combine(combined_interval)
        
        return clustering_reps, cluster_indices, market_stock_similarities