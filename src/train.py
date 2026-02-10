"""
Training script for GraphRAG-IM.
"""

import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from model import GraphRAGIM, GraphRAGIMTopologyOnly, GraphRAGIMConcatFusion
from data import GraphRAGIMDataset, create_influence_labels
from ic_simulation import evaluate_seeds, single_node_influence


def train_epoch(model, data, labels, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    scores = model(data.x_struct, data.edge_index, data.x_text)
    
    # Get labels tensor
    label_tensor = torch.tensor([labels.get(data.idx_to_node[i], 0) 
                                  for i in range(data.num_nodes)], 
                                 dtype=torch.float, device=device)
    
    # Loss
    loss = criterion(scores, label_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate(model, data, G_val, k=50, p=0.1, num_simulations=1000, device='cpu'):
    """Validate by evaluating seed selection."""
    model.eval()
    
    with torch.no_grad():
        scores = model(data.x_struct, data.edge_index, data.x_text)
        
        # Select top-k seeds
        _, top_indices = torch.topk(scores, k)
        seeds = [data.idx_to_node[i.item()] for i in top_indices]
    
    # Evaluate spread
    result = evaluate_seeds(G_val, seeds, p=p, num_simulations=num_simulations)
    
    return result['mean'], result['std'], seeds


def compute_label_correlation(model, data, G, p=0.1, device='cpu'):
    """Compute Spearman correlation between scores and single-node influence."""
    model.eval()
    
    with torch.no_grad():
        scores = model(data.x_struct, data.edge_index, data.x_text)
        scores = scores.cpu().numpy()
    
    # Get single-node influence (cached if possible)
    influence = single_node_influence(G, p=p, num_simulations=100, show_progress=False)
    
    # Compute correlation
    pred_scores = []
    true_influence = []
    for i in range(data.num_nodes):
        node = data.idx_to_node[i]
        pred_scores.append(scores[i])
        true_influence.append(influence.get(node, 0))
    
    rho, pval = spearmanr(pred_scores, true_influence)
    return rho


def train(
    dataset: GraphRAGIMDataset,
    model_type: str = 'full',
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k: int = 50,
    p: float = 0.1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = './checkpoints'
):
    """
    Full training pipeline.
    
    Args:
        dataset: GraphRAGIMDataset instance
        model_type: 'full', 'topo_only', or 'concat'
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        k: Seed budget for evaluation
        p: Propagation probability
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Temporal split
    print("Creating temporal split...")
    G_train, G_val, G_test = dataset.temporal_split()
    print(f"Train edges: {G_train.number_of_edges()}")
    print(f"Val edges: {G_val.number_of_edges()}")
    print(f"Test edges: {G_test.number_of_edges()}")
    
    # Prepare data (use G_train to prevent temporal leakage!)
    print("Preparing training data...")
    data = dataset.prepare_pytorch_data(G_train, device=device)
    
    # Create labels
    print("Creating influence labels...")
    labels = create_influence_labels(G_train, num_seed_sets=100, num_simulations=1000, p=p)
    pos_count = sum(labels.values())
    print(f"Positive labels: {pos_count}/{len(labels)} ({100*pos_count/len(labels):.1f}%)")
    
    # Initialize model
    if model_type == 'full':
        model = GraphRAGIM(
            struct_input_dim=data.x_struct.shape[1],
            text_dim=data.x_text.shape[1]
        ).to(device)
    elif model_type == 'topo_only':
        model = GraphRAGIMTopologyOnly(
            struct_input_dim=data.x_struct.shape[1]
        ).to(device)
    elif model_type == 'concat':
        model = GraphRAGIMConcatFusion(
            struct_input_dim=data.x_struct.shape[1],
            text_dim=data.x_text.shape[1]
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Training loop
    best_spread = 0
    best_epoch = 0
    history = []
    
    for epoch in range(epochs):
        # Train
        loss = train_epoch(model, data, labels, optimizer, criterion, device)
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            spread_mean, spread_std, seeds = validate(
                model, data, G_val, k=k, p=p, num_simulations=1000, device=device
            )
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | "
                  f"Val Spread: {spread_mean:.1f}±{spread_std:.1f}")
            
            history.append({
                'epoch': epoch + 1,
                'loss': loss,
                'val_spread_mean': spread_mean,
                'val_spread_std': spread_std
            })
            
            # Save best model
            if spread_mean > best_spread:
                best_spread = spread_mean
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'spread': spread_mean,
                }, os.path.join(save_dir, f'{model_type}_best.pt'))
    
    print(f"\nBest validation spread: {best_spread:.1f} at epoch {best_epoch}")
    
    # Final test evaluation
    print("\nFinal test evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, f'{model_type}_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare test data (still using train graph for features to prevent leakage)
    spread_mean, spread_std, seeds = validate(
        model, data, G_test, k=k, p=p, num_simulations=10000, device=device
    )
    
    print(f"Test Spread: {spread_mean:.1f}±{spread_std:.1f}")
    
    # Compute correlation
    rho = compute_label_correlation(model, data, G_train, p=p, device=device)
    print(f"Spearman ρ (scores vs single-node influence): {rho:.3f}")
    
    # Save results
    results = {
        'model_type': model_type,
        'test_spread_mean': spread_mean,
        'test_spread_std': spread_std,
        'best_epoch': best_epoch,
        'spearman_rho': rho,
        'history': history,
        'seeds': seeds,
        'config': {
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'k': k,
            'p': p
        }
    }
    
    with open(os.path.join(save_dir, f'{model_type}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train GraphRAG-IM')
    parser.add_argument('--graph', type=str, required=True, help='Path to graph file')
    parser.add_argument('--text', type=str, required=True, help='Path to node text JSON')
    parser.add_argument('--summaries', type=str, required=True, help='Path to LLM summaries')
    parser.add_argument('--timestamps', type=str, default=None, help='Path to edge timestamps')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'topo_only', 'concat'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=50, help='Seed budget')
    parser.add_argument('--p', type=float, default=0.1, help='Propagation probability')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = GraphRAGIMDataset(
        graph_path=args.graph,
        text_path=args.text,
        summaries_path=args.summaries,
        edge_timestamps_path=args.timestamps
    )
    
    # Train
    results = train(
        dataset=dataset,
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        k=args.k,
        p=args.p,
        device=args.device,
        save_dir=args.save_dir
    )
    
    print("\nTraining complete!")
    print(f"Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
