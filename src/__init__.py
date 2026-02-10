"""
GraphRAG-IM: Retrieval-Augmented Graph Neural Networks for Influence Maximization
"""

from .model import GraphRAGIM, GraphRAGIMTopologyOnly, GraphRAGIMConcatFusion
from .stage1_retrieval import Stage1Pipeline, NeighborhoodRetriever, LLMSummarizer
from .data import GraphRAGIMDataset, TextEncoder, StructuralFeatureExtractor
from .ic_simulation import ic_simulation, monte_carlo_spread, evaluate_seeds

__version__ = "1.0.0"
__all__ = [
    "GraphRAGIM",
    "GraphRAGIMTopologyOnly", 
    "GraphRAGIMConcatFusion",
    "Stage1Pipeline",
    "NeighborhoodRetriever",
    "LLMSummarizer",
    "GraphRAGIMDataset",
    "TextEncoder",
    "StructuralFeatureExtractor",
    "ic_simulation",
    "monte_carlo_spread",
    "evaluate_seeds"
]
