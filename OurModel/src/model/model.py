class model(nn.Module):
    """
    Main model architecture combining three key components:
    1. Stock Encoding
    2. Dynamic Stock Clustering
    3. Adaptive Output Aggregation Using Gating Mechanism
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

        # Adaptive Output Aggregation Using Gating Mechanism
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
       3. Generate and combine predictions using Adaptive Output Aggregation

        data_batch: Tensor[batch_size(1), stock_num, lookback_length, fea_num]
    
       """
        
       # Get encoded representations
       _, encoding_reps, market_reps, stock_reps = self.stock_encoding(data_batch)
       
       # Get clustering-based representations
       # stock_reps: Tensor[stock_num, hidden_dim]
       # market_reps: Tensor[market_num, hidden_dim]
       clustering_reps, cluster_indices, market_stock_similarities = self.stock_clustering(stock_reps, market_reps)
    
       # Generate local and cluster-based predictions
       # Input: encoding_reps: Tensor[stock_num, hidden_dim]
       # Output: o_local: Tensor[stock_num, 1]
       o_local = self.encoding_mlp(encoding_reps)      # ĉ^l_i in paper
    
       # Input: clustering_reps: Tensor[stock_num, hidden_dim]
       # Output: o_cluster: Tensor[stock_num, 1]
       o_cluster = self.clustering_mlp(clustering_reps) # ĉ^c_i in paper
    
       # Compute gating parameter α_i
       # Input: concatenated: Tensor[stock_num, 2*hidden_dim]
       # Output: gate: Tensor[stock_num, 1]
       gate_input = torch.cat([encoding_reps, clustering_reps], dim=-1)
       gate = self.gate_network(gate_input)
       
       # Final prediction: α_i * ĉ^l_i + (1-α_i) * ĉ^c_i
       # All inputs and output: Tensor[stock_num, 1]
       final_prediction = gate * o_local + (1 - gate) * o_cluster
       
       return final_prediction, cluster_indices, market_stock_similarities, gate
