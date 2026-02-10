"""
Example: Quick test of GraphRAG-IM on synthetic data.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
import networkx as nx
from model import GraphRAGIM
from ic_simulation import evaluate_seeds


def create_synthetic_data(num_nodes=100, num_edges=300, text_dim=384):
    """Create synthetic graph data for testing."""
    
    # Random graph
    G = nx.gnm_random_graph(num_nodes, num_edges)
    G = G.to_undirected()
    
    # Structural features (4 dims: degree, pagerank, clustering, betweenness)
    degrees = dict(G.degree())
    pagerank = nx.pagerank(G)
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G)
    
    x_struct = np.zeros((num_nodes, 4))
    for i in range(num_nodes):
        x_struct[i] = [degrees[i], pagerank[i], clustering[i], betweenness[i]]
    
    # Normalize
    x_struct = (x_struct - x_struct.mean(0)) / (x_struct.std(0) + 1e-8)
    
    # Random text embeddings (simulating SBERT)
    x_text = np.random.randn(num_nodes, text_dim).astype(np.float32)
    
    # Edge index
    edges = list(G.edges())
    edge_index = np.array([
        [e[0] for e in edges] + [e[1] for e in edges],
        [e[1] for e in edges] + [e[0] for e in edges]
    ])
    
    return G, x_struct, x_text, edge_index


def main():
    print("=" * 60)
    print("GraphRAG-IM Quick Test")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    G, x_struct, x_text, edge_index = create_synthetic_data(
        num_nodes=100, 
        num_edges=300,
        text_dim=384
    )
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Convert to tensors
    x_struct = torch.tensor(x_struct, dtype=torch.float32)
    x_text = torch.tensor(x_text, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Initialize model
    print("\n2. Initializing GraphRAG-IM model...")
    model = GraphRAGIM(
        struct_input_dim=4,
        text_dim=384,
        hidden_dim=128,
        num_gnn_layers=3,
        num_projections=6
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        scores = model(x_struct, edge_index, x_text)
    
    print(f"   Output shape: {scores.shape}")
    print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Select seeds
    print("\n4. Selecting top-k seeds...")
    k = 10
    _, top_indices = torch.topk(scores, k)
    seeds = top_indices.tolist()
    print(f"   Selected seeds (k={k}): {seeds}")
    
    # Evaluate with IC simulation
    print("\n5. Evaluating with IC simulation (1000 runs)...")
    result = evaluate_seeds(G, seeds, p=0.1, num_simulations=1000)
    print(f"   Spread: {result['mean']:.1f} ± {result['std']:.1f}")
    
    # Compare with random baseline
    print("\n6. Comparing with random baseline...")
    random_seeds = np.random.choice(list(G.nodes()), size=k, replace=False).tolist()
    random_result = evaluate_seeds(G, random_seeds, p=0.1, num_simulations=1000)
    print(f"   Random spread: {random_result['mean']:.1f} ± {random_result['std']:.1f}")
    
    improvement = (result['mean'] - random_result['mean']) / random_result['mean'] * 100
    print(f"   Improvement: {improvement:+.1f}%")
    
    print("\n" + "=" * 60)
    print("Quick test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
