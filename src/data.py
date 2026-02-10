"""
Data loading and feature extraction for GraphRAG-IM.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


class StructuralFeatureExtractor:
    """Extract normalized structural features for nodes."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, G: nx.Graph) -> np.ndarray:
        """
        Extract structural features: degree, PageRank, clustering coefficient, betweenness.
        
        Args:
            G: NetworkX graph
            
        Returns:
            features: [num_nodes, 4] normalized feature matrix
        """
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Compute features
        degrees = dict(G.degree())
        pagerank = nx.pagerank(G, max_iter=100)
        clustering = nx.clustering(G)
        
        # Betweenness (can be slow for large graphs)
        if num_nodes > 10000:
            # Sample-based approximation for large graphs
            betweenness = nx.betweenness_centrality(G, k=min(500, num_nodes))
        else:
            betweenness = nx.betweenness_centrality(G)
        
        # Build feature matrix
        features = np.zeros((num_nodes, 4))
        for node in nodes:
            idx = node_to_idx[node]
            features[idx, 0] = degrees[node]
            features[idx, 1] = pagerank[node]
            features[idx, 2] = clustering[node]
            features[idx, 3] = betweenness[node]
        
        # Normalize
        features = self.scaler.fit_transform(features)
        
        return features, node_to_idx


class TextEncoder:
    """Encode text summaries using Sentence-BERT."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: SBERT model name (default produces 384-dim embeddings)
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            embeddings: [num_texts, embedding_dim] array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings


class GraphRAGIMDataset:
    """
    Dataset class for GraphRAG-IM.
    Handles temporal splits and prevents leakage.
    """
    
    def __init__(
        self,
        graph_path: str,
        text_path: str,
        summaries_path: str,
        edge_timestamps_path: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1
    ):
        """
        Args:
            graph_path: Path to edge list or graph file
            text_path: Path to node text JSON
            summaries_path: Path to LLM summaries JSON
            edge_timestamps_path: Path to edge timestamps (for temporal split)
            train_ratio: Fraction of edges for training
            val_ratio: Fraction of edges for validation
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Load data
        self.G_full = self._load_graph(graph_path)
        self.node_text = self._load_json(text_path)
        self.summaries = self._load_json(summaries_path)
        
        if edge_timestamps_path:
            self.edge_timestamps = self._load_json(edge_timestamps_path)
        else:
            self.edge_timestamps = None
            
        # Feature extractors
        self.struct_extractor = StructuralFeatureExtractor()
        self.text_encoder = TextEncoder()
        
    def _load_graph(self, path: str) -> nx.Graph:
        """Load graph from various formats."""
        if path.endswith('.gpickle') or path.endswith('.pkl'):
            return nx.read_gpickle(path)
        elif path.endswith('.edgelist'):
            return nx.read_edgelist(path, nodetype=int)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            G = nx.Graph()
            G.add_edges_from(data['edges'])
            return G
        else:
            raise ValueError(f"Unknown graph format: {path}")
    
    def _load_json(self, path: str) -> Dict:
        """Load JSON file with int keys."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle summaries format
        if 'summaries' in data:
            data = data['summaries']
        return {int(k): v for k, v in data.items()}
    
    def temporal_split(self) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
        """
        Split graph temporally based on edge timestamps.
        
        Returns:
            G_train, G_val, G_test graphs
        """
        if self.edge_timestamps is None:
            # Random split if no timestamps
            edges = list(self.G_full.edges())
            np.random.shuffle(edges)
            n = len(edges)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            
            G_train = nx.Graph()
            G_train.add_nodes_from(self.G_full.nodes())
            G_train.add_edges_from(edges[:train_end])
            
            G_val = G_train.copy()
            G_val.add_edges_from(edges[train_end:val_end])
            
            G_test = self.G_full.copy()
            
            return G_train, G_val, G_test
        
        # Temporal split
        edges_with_time = [(e, self.edge_timestamps.get(str(e), 0)) 
                          for e in self.G_full.edges()]
        edges_with_time.sort(key=lambda x: x[1])
        
        n = len(edges_with_time)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        G_train = nx.Graph()
        G_train.add_nodes_from(self.G_full.nodes())
        G_train.add_edges_from([e for e, t in edges_with_time[:train_end]])
        
        G_val = G_train.copy()
        G_val.add_edges_from([e for e, t in edges_with_time[train_end:val_end]])
        
        G_test = self.G_full.copy()
        
        return G_train, G_val, G_test
    
    def prepare_pytorch_data(self, G: nx.Graph, device: str = 'cpu') -> Data:
        """
        Prepare PyTorch Geometric Data object.
        
        IMPORTANT: Use G_train for training to prevent temporal leakage!
        
        Args:
            G: Graph to use (should be G_train for training)
            device: Device to place tensors on
            
        Returns:
            PyG Data object with x_struct, x_text, edge_index
        """
        # Extract structural features (on given graph)
        struct_features, node_to_idx = self.struct_extractor.extract_features(G)
        
        # Get ordered node list
        nodes = list(G.nodes())
        
        # Encode text summaries
        summaries_ordered = [self.summaries.get(n, "") for n in nodes]
        text_embeddings = self.text_encoder.encode(summaries_ordered)
        
        # Build edge index
        edges = list(G.edges())
        edge_index = torch.tensor([
            [node_to_idx[e[0]] for e in edges] + [node_to_idx[e[1]] for e in edges],
            [node_to_idx[e[1]] for e in edges] + [node_to_idx[e[0]] for e in edges]
        ], dtype=torch.long)
        
        # Create Data object
        data = Data(
            x_struct=torch.tensor(struct_features, dtype=torch.float),
            x_text=torch.tensor(text_embeddings, dtype=torch.float),
            edge_index=edge_index,
            num_nodes=len(nodes)
        )
        
        data.node_to_idx = node_to_idx
        data.idx_to_node = {v: k for k, v in node_to_idx.items()}
        
        return data.to(device)


def create_influence_labels(
    G: nx.Graph,
    num_seed_sets: int = 100,
    num_simulations: int = 1000,
    top_percent: float = 0.1,
    threshold: float = 0.5,
    p: float = 0.1
) -> Dict[int, int]:
    """
    Create training labels following DeepIM methodology.
    
    Nodes appearing in ≥threshold of top-percent highest-spread sets are positive.
    
    Args:
        G: Graph for IC simulation
        num_seed_sets: Number of random seed sets to sample
        num_simulations: Number of MC simulations per seed set
        top_percent: Top percentage of spreads to consider
        threshold: Fraction threshold for positive labels
        p: Propagation probability
        
    Returns:
        Dict mapping node IDs to binary labels (0 or 1)
    """
    from .ic_simulation import ic_simulation
    
    nodes = list(G.nodes())
    node_appearances = {n: 0 for n in nodes}
    
    all_spreads = []
    
    # Generate random seed sets and simulate
    for _ in range(num_seed_sets):
        k = np.random.randint(10, 51)  # Random seed budget
        seeds = set(np.random.choice(nodes, size=k, replace=False))
        
        # Monte Carlo simulation
        spread = np.mean([ic_simulation(G, seeds, p) for _ in range(num_simulations)])
        all_spreads.append((seeds, spread))
    
    # Get top spreads
    all_spreads.sort(key=lambda x: x[1], reverse=True)
    top_k = int(len(all_spreads) * top_percent)
    top_sets = [s for s, _ in all_spreads[:top_k]]
    
    # Count appearances
    for seed_set in top_sets:
        for node in seed_set:
            node_appearances[node] += 1
    
    # Create labels
    labels = {}
    for node in nodes:
        labels[node] = 1 if node_appearances[node] >= threshold * top_k else 0
    
    return labels
