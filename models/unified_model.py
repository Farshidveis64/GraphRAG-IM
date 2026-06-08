import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

    class _Missing:
        pass

    nn = type("nn", (), {"Module": _Missing})


def _require_torch():
    if not _TORCH_OK:
        raise ImportError(
            "The neural path requires torch + torch_geometric. Install the "
            "'neural' extras, or use HeuristicScorer / scoring='heuristic'.")


class GraphSAGEEncoder(nn.Module):
    """Three-layer GraphSAGE structure encoder (mean aggregation)."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        _require_torch()
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim, aggr="mean")])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.dropout(self.act(x))
        return x


class CrossAttentionFusion(nn.Module):
    """Topology-queries-content cross-attention fusion."""

    def __init__(self, top_dim: int = 128, text_dim: int = 384, attn_dim: int = 64,
                 num_slots: int = 6, dropout: float = 0.1):
        _require_torch()
        super().__init__()
        self.attn_dim = attn_dim
        self.w_q = nn.Linear(top_dim, attn_dim, bias=False)
        self.w_kv = nn.ModuleList(
            [nn.Linear(text_dim, attn_dim, bias=False) for _ in range(num_slots)])
        self.w_r = nn.Linear(top_dim, attn_dim, bias=False)
        self.w_o = nn.Linear(attn_dim, top_dim, bias=False)
        self.norm = nn.LayerNorm(top_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_top, h_txt):
        q = self.w_q(h_top).unsqueeze(1)
        kv = torch.stack([proj(h_txt) for proj in self.w_kv], dim=1)
        scores = (q @ kv.transpose(1, 2)) / (self.attn_dim ** 0.5)
        attended = (F.softmax(scores, dim=-1) @ kv).squeeze(1)
        fused = attended + self.w_r(h_top)
        return self.norm(self.dropout(self.w_o(fused)))


class UnifiedGraphRAGIM(nn.Module):
    """Structure encoder + cross-attention fusion + MLP influence head.

    Outputs a per-node influence score s_v in [0, 1]. Requires torch; use
    HeuristicScorer for the dependency-free path.
    """

    def __init__(self, struct_dim: int = 4, text_dim: int = 384,
                 graph_hidden: int = 128, graph_layers: int = 3, attn_dim: int = 64,
                 attn_slots: int = 6, dropout: float = 0.1, mlp_hidden: int = 64,
                 mlp_dropout: float = 0.3):
        _require_torch()
        super().__init__()
        self.encoder = GraphSAGEEncoder(struct_dim, graph_hidden, graph_layers, dropout)
        self.fusion = CrossAttentionFusion(graph_hidden, text_dim, attn_dim,
                                           attn_slots, dropout)
        self.head = nn.Sequential(
            nn.Linear(graph_hidden, mlp_hidden), nn.ReLU(),
            nn.Dropout(mlp_dropout), nn.Linear(mlp_hidden, 1))

    def forward(self, x_struct, edge_index, h_txt):
        h_top = self.encoder(x_struct, edge_index)
        h_fused = self.fusion(h_top, h_txt)
        return torch.sigmoid(self.head(h_fused).squeeze(-1))


class HeuristicScorer:
    """Dependency-free influence scorer (offline stand-in for the GNN).

    Blends a normalized structural prior with content centrality and squashes
    to [0, 1]: s_v = sigmoid(beta * prior_v + (1 - beta) * centrality_v).
    """

    def __init__(self, beta: float = 0.6, structural_prior: str = "pagerank"):
        self.beta = beta
        self.structural_prior = structural_prior

    def score(self, graph: nx.Graph, embeddings: np.ndarray) -> np.ndarray:
        prior = self._prior(graph)
        centroid = embeddings.mean(axis=0, keepdims=True)
        centroid /= np.linalg.norm(centroid) + 1e-12
        emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        centrality = _minmax(emb @ centroid.ravel())
        logits = self.beta * prior + (1.0 - self.beta) * centrality
        return _sigmoid(4.0 * (logits - logits.mean()))

    def _prior(self, graph: nx.Graph) -> np.ndarray:
        n = graph.number_of_nodes()
        if self.structural_prior == "degree":
            vals = np.array([d for _, d in graph.degree()], dtype=np.float64)
        else:
            pr = nx.pagerank(graph)
            vals = np.array([pr[i] for i in range(n)], dtype=np.float64)
        return _minmax(vals)


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
