"""
GraphRAG-IM: Retrieval-Augmented Graph Neural Networks for Influence Maximization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """3-layer GraphSAGE encoder for topology branch."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for combining topology and text embeddings."""
    
    def __init__(self, topo_dim=128, text_dim=384, hidden_dim=64, num_projections=6, dropout=0.1):
        super().__init__()
        self.num_projections = num_projections
        self.hidden_dim = hidden_dim
        
        # Query projection from topology
        self.W_q = nn.Linear(topo_dim, hidden_dim)
        
        # Key/Value projections from text (6 learned projections)
        self.W_kv = nn.ModuleList([
            nn.Linear(text_dim, hidden_dim) for _ in range(num_projections)
        ])
        
        # Residual projection
        self.W_r = nn.Linear(topo_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, topo_dim)
        
        self.layer_norm = nn.LayerNorm(topo_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h_topo, h_text):
        """
        Args:
            h_topo: [batch_size, topo_dim] topology embeddings
            h_text: [batch_size, text_dim] text embeddings
        Returns:
            h_fused: [batch_size, topo_dim] fused embeddings
        """
        # Query from topology
        q = self.W_q(h_topo)  # [batch_size, hidden_dim]
        
        # Keys/Values from text (6 projections)
        kv_list = [proj(h_text) for proj in self.W_kv]  # list of [batch_size, hidden_dim]
        H_txt = torch.stack(kv_list, dim=1)  # [batch_size, num_proj, hidden_dim]
        
        # Cross-attention
        attn_scores = torch.bmm(q.unsqueeze(1), H_txt.transpose(1, 2))  # [batch_size, 1, num_proj]
        attn_scores = attn_scores / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum
        context = torch.bmm(attn_weights, H_txt).squeeze(1)  # [batch_size, hidden_dim]
        
        # Residual connection
        residual = self.W_r(h_topo)
        h_fused = context + residual
        
        # Output projection + LayerNorm + Dropout
        h_fused = self.W_o(h_fused)
        h_fused = self.layer_norm(h_fused)
        h_fused = self.dropout(h_fused)
        
        return h_fused


class InfluenceScorer(nn.Module):
    """MLP scorer for influence prediction."""
    
    def __init__(self, input_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h):
        return self.mlp(h).squeeze(-1)


class GraphRAGIM(nn.Module):
    """
    GraphRAG-IM: Full model combining GraphSAGE, Cross-Attention Fusion, and Influence Scorer.
    
    Stage 1 (Retrieval + LLM) is done offline - this model handles Stage 2 & 3.
    """
    
    def __init__(
        self,
        struct_input_dim=4,  # degree, pagerank, clustering, betweenness
        text_dim=384,  # SBERT dimension
        hidden_dim=128,
        num_gnn_layers=3,
        num_projections=6,
        dropout=0.1,
        scorer_dropout=0.3
    ):
        super().__init__()
        
        # Stage 2: Dual-Branch Encoding
        self.topo_encoder = GraphSAGEEncoder(
            input_dim=struct_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        self.fusion = CrossAttentionFusion(
            topo_dim=hidden_dim,
            text_dim=text_dim,
            hidden_dim=64,
            num_projections=num_projections,
            dropout=dropout
        )
        
        # Stage 3: Seed Selection
        self.scorer = InfluenceScorer(
            input_dim=hidden_dim,
            hidden_dim=64,
            dropout=scorer_dropout
        )
        
    def forward(self, x_struct, edge_index, h_text):
        """
        Args:
            x_struct: [num_nodes, struct_input_dim] structural features
            edge_index: [2, num_edges] edge indices
            h_text: [num_nodes, text_dim] SBERT embeddings of LLM summaries
        Returns:
            scores: [num_nodes] influence scores
        """
        # Topology branch
        h_topo = self.topo_encoder(x_struct, edge_index)
        
        # Cross-attention fusion
        h_fused = self.fusion(h_topo, h_text)
        
        # Influence scoring
        scores = self.scorer(h_fused)
        
        return scores
    
    def select_seeds(self, scores, k):
        """Greedy top-k seed selection."""
        _, indices = torch.topk(scores, k)
        return indices


class GraphRAGIMTopologyOnly(nn.Module):
    """Ablation: Topology-only baseline (w/o Text)."""
    
    def __init__(self, struct_input_dim=4, hidden_dim=128, num_gnn_layers=3, dropout=0.1):
        super().__init__()
        self.topo_encoder = GraphSAGEEncoder(struct_input_dim, hidden_dim, num_gnn_layers, dropout)
        self.scorer = InfluenceScorer(hidden_dim, 64, 0.3)
        
    def forward(self, x_struct, edge_index, h_text=None):
        h_topo = self.topo_encoder(x_struct, edge_index)
        return self.scorer(h_topo)


class GraphRAGIMConcatFusion(nn.Module):
    """Ablation: Concatenation fusion instead of cross-attention (w/o Cross-Attention)."""
    
    def __init__(self, struct_input_dim=4, text_dim=384, hidden_dim=128, num_gnn_layers=3, dropout=0.1):
        super().__init__()
        self.topo_encoder = GraphSAGEEncoder(struct_input_dim, hidden_dim, num_gnn_layers, dropout)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.scorer = InfluenceScorer(hidden_dim, 64, 0.3)
        
    def forward(self, x_struct, edge_index, h_text):
        h_topo = self.topo_encoder(x_struct, edge_index)
        h_concat = torch.cat([h_topo, h_text], dim=-1)
        h_fused = self.fusion_mlp(h_concat)
        return self.scorer(h_fused)
