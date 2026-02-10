"""
Stage 1: Neighborhood Text Retrieval and LLM Summarization
"""

import os
import json
import hashlib
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


class NeighborhoodRetriever:
    """
    Retrieves 2-hop neighborhood text with inverse distance weighting.
    """
    
    def __init__(self, max_neighbors: int = 50, max_hops: int = 2, max_tokens: int = 2048):
        self.max_neighbors = max_neighbors
        self.max_hops = max_hops
        self.max_tokens = max_tokens
        
    def get_neighborhood(self, G: nx.Graph, node: int) -> List[Tuple[int, int]]:
        """
        BFS to get 2-hop neighbors with distances.
        
        Returns:
            List of (neighbor_id, distance) tuples, limited to max_neighbors
        """
        visited = {node: 0}
        queue = deque([node])
        neighbors = []
        
        while queue:
            current = queue.popleft()
            current_dist = visited[current]
            
            if current_dist >= self.max_hops:
                continue
                
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    neighbors.append((neighbor, current_dist + 1))
                    
                    if len(neighbors) >= self.max_neighbors:
                        return neighbors
        
        return neighbors
    
    def aggregate_text(
        self, 
        node: int, 
        node_text: Dict[int, str], 
        neighbors: List[Tuple[int, int]],
        probabilistic_sampling: bool = True
    ) -> str:
        """
        Aggregate neighborhood text with inverse distance weighting.
        
        Args:
            node: Target node ID
            node_text: Dict mapping node IDs to their text content
            neighbors: List of (neighbor_id, distance) tuples
            probabilistic_sampling: If True, sample tokens probabilistically
            
        Returns:
            Aggregated text T_v truncated to max_tokens
        """
        # Start with node's own text
        aggregated = [f"[USER]: {node_text.get(node, '')}"]
        
        # Add neighbor text with distance weighting
        for neighbor_id, distance in neighbors:
            weight = 1.0 / (distance + 1)
            text = node_text.get(neighbor_id, '')
            
            if probabilistic_sampling and text:
                # Probabilistic token sampling
                tokens = text.split()
                sampled_tokens = [t for t in tokens if random.random() < weight]
                text = ' '.join(sampled_tokens)
            
            if text:
                aggregated.append(f"[NEIGHBOR d={distance}]: {text}")
        
        # Join and truncate
        full_text = '\n'.join(aggregated)
        tokens = full_text.split()[:self.max_tokens]
        return ' '.join(tokens)
    
    def retrieve_context(self, G: nx.Graph, node: int, node_text: Dict[int, str]) -> str:
        """Full retrieval pipeline for a single node."""
        neighbors = self.get_neighborhood(G, node)
        return self.aggregate_text(node, node_text, neighbors)


class LLMSummarizer:
    """
    Generates influence-aware summaries using LLM.
    """
    
    PROMPT_TEMPLATE = """Summarize this user's influence potential in 100 words based on:
(1) expertise/topics
(2) engagement quality  
(3) community standing

User content: {user_content}

Neighbor content: {neighbor_content}

Summary:"""
    
    def __init__(
        self, 
        model: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.0,
        max_summary_tokens: int = 150,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_summary_tokens = max_summary_tokens
        
        # Set API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
    def generate_summary(self, user_content: str, neighbor_content: str) -> str:
        """Generate summary using OpenAI API."""
        try:
            import openai
            
            prompt = self.PROMPT_TEMPLATE.format(
                user_content=user_content[:1000],  # Truncate for prompt
                neighbor_content=neighbor_content[:3000]
            )
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_summary_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            print("Warning: openai package not installed. Using placeholder summary.")
            return self._placeholder_summary(user_content)
        except Exception as e:
            print(f"Warning: API call failed ({e}). Using placeholder summary.")
            return self._placeholder_summary(user_content)
    
    def _placeholder_summary(self, user_content: str) -> str:
        """Fallback placeholder when API is unavailable."""
        words = user_content.split()[:20]
        return f"User discusses: {' '.join(words)}... [placeholder summary]"


class Stage1Pipeline:
    """
    Complete Stage 1 pipeline: Retrieval + LLM Summarization.
    """
    
    def __init__(
        self,
        max_neighbors: int = 50,
        max_hops: int = 2,
        llm_model: str = "gpt-3.5-turbo-0125",
        cache_dir: str = "./cache/summaries"
    ):
        self.retriever = NeighborhoodRetriever(max_neighbors, max_hops)
        self.summarizer = LLMSummarizer(model=llm_model)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, dataset_name: str) -> str:
        return os.path.join(self.cache_dir, f"{dataset_name}_summaries.json")
    
    def _compute_checksum(self, summaries: Dict[int, str]) -> str:
        """Compute MD5 checksum for reproducibility."""
        content = json.dumps(summaries, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_graph(
        self,
        G: nx.Graph,
        node_text: Dict[int, str],
        dataset_name: str = "dataset",
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Dict[int, str]:
        """
        Process entire graph to generate summaries.
        
        Args:
            G: NetworkX graph (use training-period edges only!)
            node_text: Dict mapping node IDs to text content
            dataset_name: Name for caching
            use_cache: Whether to use cached summaries
            show_progress: Show progress bar
            
        Returns:
            Dict mapping node IDs to LLM-generated summaries
        """
        cache_path = self._get_cache_path(dataset_name)
        
        # Try loading from cache
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached summaries from {cache_path}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
                print(f"Checksum: {data['checksum']}")
                return {int(k): v for k, v in data['summaries'].items()}
        
        # Generate summaries
        summaries = {}
        nodes = list(G.nodes())
        
        iterator = tqdm(nodes, desc="Generating summaries") if show_progress else nodes
        
        for node in iterator:
            # Retrieve neighborhood context
            context = self.retriever.retrieve_context(G, node, node_text)
            user_content = node_text.get(node, "")
            
            # Generate summary
            summary = self.summarizer.generate_summary(user_content, context)
            summaries[node] = summary
        
        # Save with checksum
        checksum = self._compute_checksum(summaries)
        with open(cache_path, 'w') as f:
            json.dump({
                'checksum': checksum,
                'summaries': summaries,
                'metadata': {
                    'num_nodes': len(summaries),
                    'model': self.summarizer.model
                }
            }, f, indent=2)
        
        print(f"Saved summaries to {cache_path}")
        print(f"Checksum: {checksum}")
        
        return summaries


def load_precomputed_summaries(path: str) -> Tuple[Dict[int, str], str]:
    """
    Load pre-computed summaries with checksum verification.
    
    Returns:
        Tuple of (summaries_dict, checksum)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    summaries = {int(k): v for k, v in data['summaries'].items()}
    checksum = data['checksum']
    
    # Verify checksum
    computed = hashlib.md5(json.dumps(data['summaries'], sort_keys=True).encode()).hexdigest()
    if computed != checksum:
        raise ValueError(f"Checksum mismatch! Expected {checksum}, got {computed}")
    
    return summaries, checksum
