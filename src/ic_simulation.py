"""
Independent Cascade (IC) Model Simulation for Influence Maximization.
"""

import random
from typing import Set, List, Dict, Optional
from collections import deque

import numpy as np
import networkx as nx
from tqdm import tqdm


def ic_simulation(G: nx.Graph, seeds: Set[int], p: float = 0.1) -> int:
    """
    Single Independent Cascade simulation.
    
    Args:
        G: NetworkX graph
        seeds: Set of seed node IDs
        p: Propagation probability
        
    Returns:
        Number of activated nodes
    """
    activated = set(seeds)
    newly_activated = set(seeds)
    
    while newly_activated:
        next_activated = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated:
                    if random.random() < p:
                        next_activated.add(neighbor)
        
        activated |= next_activated
        newly_activated = next_activated
    
    return len(activated)


def monte_carlo_spread(
    G: nx.Graph, 
    seeds: Set[int], 
    p: float = 0.1, 
    num_simulations: int = 10000
) -> float:
    """
    Estimate expected spread via Monte Carlo simulation.
    
    Args:
        G: NetworkX graph
        seeds: Set of seed node IDs
        p: Propagation probability
        num_simulations: Number of MC simulations
        
    Returns:
        Mean spread across simulations
    """
    spreads = [ic_simulation(G, seeds, p) for _ in range(num_simulations)]
    return np.mean(spreads), np.std(spreads)


def evaluate_seeds(
    G: nx.Graph,
    seeds: List[int],
    p: float = 0.1,
    num_simulations: int = 10000,
    show_progress: bool = False
) -> Dict:
    """
    Comprehensive evaluation of seed set.
    
    Args:
        G: NetworkX graph
        seeds: List of seed node IDs
        p: Propagation probability
        num_simulations: Number of MC simulations
        
    Returns:
        Dict with mean, std, and individual runs
    """
    seed_set = set(seeds)
    
    if show_progress:
        spreads = [ic_simulation(G, seed_set, p) for _ in tqdm(range(num_simulations))]
    else:
        spreads = [ic_simulation(G, seed_set, p) for _ in range(num_simulations)]
    
    return {
        'mean': np.mean(spreads),
        'std': np.std(spreads),
        'min': np.min(spreads),
        'max': np.max(spreads),
        'median': np.median(spreads)
    }


def single_node_influence(
    G: nx.Graph,
    p: float = 0.1,
    num_simulations: int = 1000,
    show_progress: bool = True
) -> Dict[int, float]:
    """
    Compute single-node influence σ({v}) for all nodes.
    Used for label validation (Spearman correlation).
    
    Args:
        G: NetworkX graph
        p: Propagation probability
        num_simulations: MC simulations per node
        
    Returns:
        Dict mapping node ID to influence score
    """
    nodes = list(G.nodes())
    influence = {}
    
    iterator = tqdm(nodes, desc="Computing single-node influence") if show_progress else nodes
    
    for node in iterator:
        spreads = [ic_simulation(G, {node}, p) for _ in range(num_simulations)]
        influence[node] = np.mean(spreads)
    
    return influence


class GreedyIM:
    """
    Greedy baseline for Influence Maximization.
    """
    
    def __init__(self, G: nx.Graph, p: float = 0.1, num_simulations: int = 1000):
        self.G = G
        self.p = p
        self.num_simulations = num_simulations
        
    def select_seeds(self, k: int, show_progress: bool = True) -> List[int]:
        """
        Greedy seed selection with lazy evaluation (CELF-style).
        
        Args:
            k: Number of seeds to select
            
        Returns:
            List of selected seed node IDs
        """
        seeds = []
        candidates = set(self.G.nodes())
        
        # Initial marginal gains
        gains = {}
        for node in tqdm(candidates, desc="Initial gains") if show_progress else candidates:
            gains[node] = np.mean([
                ic_simulation(self.G, {node}, self.p) 
                for _ in range(self.num_simulations)
            ])
        
        for i in range(k):
            if show_progress:
                print(f"Selecting seed {i+1}/{k}")
            
            # Find node with max marginal gain
            best_node = max(candidates, key=lambda n: gains.get(n, 0))
            seeds.append(best_node)
            candidates.remove(best_node)
            
            # Update marginal gains (lazy evaluation would be faster)
            current_seeds = set(seeds)
            current_spread = np.mean([
                ic_simulation(self.G, current_seeds, self.p)
                for _ in range(self.num_simulations // 10)  # Reduced for speed
            ])
            
            # Recompute for remaining candidates (simplified, not full CELF)
            for node in list(candidates)[:100]:  # Only top candidates
                new_seeds = current_seeds | {node}
                new_spread = np.mean([
                    ic_simulation(self.G, new_seeds, self.p)
                    for _ in range(self.num_simulations // 10)
                ])
                gains[node] = new_spread - current_spread
        
        return seeds


class CELF:
    """
    CELF (Cost-Effective Lazy Forward) baseline.
    """
    
    def __init__(self, G: nx.Graph, p: float = 0.1, num_simulations: int = 1000):
        self.G = G
        self.p = p
        self.num_simulations = num_simulations
        
    def _marginal_gain(self, seeds: Set[int], candidate: int) -> float:
        """Compute marginal gain of adding candidate to seeds."""
        current = np.mean([ic_simulation(self.G, seeds, self.p) 
                          for _ in range(self.num_simulations)])
        with_candidate = np.mean([ic_simulation(self.G, seeds | {candidate}, self.p) 
                                  for _ in range(self.num_simulations)])
        return with_candidate - current
    
    def select_seeds(self, k: int, show_progress: bool = True) -> List[int]:
        """CELF seed selection with lazy evaluation."""
        import heapq
        
        nodes = list(self.G.nodes())
        seeds = []
        
        # Initialize with marginal gains
        gains = []
        for node in tqdm(nodes, desc="CELF init") if show_progress else nodes:
            mg = self._marginal_gain(set(), node)
            heapq.heappush(gains, (-mg, 0, node))  # max-heap via negation
        
        iteration = 0
        while len(seeds) < k:
            iteration += 1
            
            while True:
                neg_mg, last_update, node = heapq.heappop(gains)
                
                if last_update == len(seeds):
                    # Valid marginal gain
                    seeds.append(node)
                    if show_progress:
                        print(f"Selected seed {len(seeds)}/{k}: node {node}, mg={-neg_mg:.2f}")
                    break
                else:
                    # Recompute marginal gain
                    new_mg = self._marginal_gain(set(seeds), node)
                    heapq.heappush(gains, (-new_mg, len(seeds), node))
        
        return seeds
