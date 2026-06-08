from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from data.data_utils import edge_prob_map, structural_features
from inference.diffusion import ContentICModel
from inference.selection import select_seeds
from models.model_utils import OfflineTextEncoder
from models.retriever import InfluencePrizedRetriever, get_summarizer, summary_checksum
from models.unified_model import HeuristicScorer


@dataclass
class InferenceResult:
    seeds: List[int]
    scores: np.ndarray
    text_embeddings: np.ndarray
    summaries: List[str] = field(default_factory=list, repr=False)
    checksums: List[str] = field(default_factory=list, repr=False)


class GraphRAGIMInference:
    """End-to-end seed-selection engine.

    Runs the three stages on a ContentGraph: (1) influence-prized retrieval +
    summarization, (2) content encoding + scoring (offline heuristic or neural),
    (3) content-conditioned DPP selection. Returns the selected seeds and all
    intermediate artifacts.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

    def _log(self, msg, *args):
        if self.logger:
            self.logger.info(msg, *args)

    def run(self, cg, scoring: Optional[str] = None, summarizer: Optional[str] = None,
            rng: Optional[np.random.Generator] = None) -> InferenceResult:
        cfg = self.config
        scoring = scoring or cfg.training.scoring
        summarizer = summarizer or cfg.retrieval.summarizer
        if rng is None:
            rng = np.random.default_rng(cfg.seed)

        cg.validate()
        cg.ensure_embeddings(dim=cfg.data.embedding_dim, seed=cfg.seed)

        model = ContentICModel(cg.graph, cg.embeddings, p0=cfg.diffusion.p0,
                               rho_mode=cfg.diffusion.rho_mode,
                               rho_temperature=cfg.diffusion.rho_temperature,
                               directed=cfg.diffusion.directed)
        edge_probs = edge_prob_map(model, cg.graph)

        # Stage 1: retrieval + summarization
        self._log("Stage 1: retrieval + summarization over %d nodes", cg.n_nodes)
        retriever = InfluencePrizedRetriever(
            n_hops=cfg.retrieval.n_hops, alpha=cfg.retrieval.alpha,
            gamma=cfg.retrieval.gamma, structural_prior=cfg.retrieval.structural_prior,
            max_nodes=cfg.retrieval.max_nodes, backend=cfg.retrieval.backend)
        summ = get_summarizer(summarizer)
        summaries = []
        for v in range(cg.n_nodes):
            res = retriever.retrieve(cg.graph, cg.embeddings, v, edge_probs,
                                     texts=cg.texts, rng=rng)
            summaries.append(summ.summarize(cg.texts[v], res.composed_text))
        checksums = [summary_checksum(s) for s in summaries]

        # Stage 2: encode summaries + score
        self._log("Stage 2: encoding (%s) + scoring (%s)", summarizer, scoring)
        text_emb = OfflineTextEncoder(dim=cfg.data.embedding_dim,
                                      seed=cfg.seed).encode(summaries)
        if scoring == "neural":
            scores = self._neural_scores(cg, text_emb)
        else:
            scores = HeuristicScorer(
                structural_prior=cfg.retrieval.structural_prior).score(
                cg.graph, text_emb)

        # Stage 3: content-conditioned selection
        seeds = select_seeds(scores, text_emb, k=cfg.selection.k,
                             method=cfg.selection.method, theta=cfg.selection.theta)
        self._log("Selected %d seeds via %s", len(seeds), cfg.selection.method)
        return InferenceResult(seeds=seeds, scores=scores, text_embeddings=text_emb,
                               summaries=summaries, checksums=checksums)

    def _neural_scores(self, cg, text_emb) -> np.ndarray:  # pragma: no cover - torch
        import torch

        from models.unified_model import UnifiedGraphRAGIM
        from training.trainer import GraphRAGIMTrainer

        cfg = self.config
        feats = structural_features(cg.graph)
        edges = np.array(list(cg.graph.edges())).T
        edges = np.concatenate([edges, edges[::-1]], axis=1)
        x_struct = torch.tensor(feats, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long)
        h_txt = torch.tensor(text_emb, dtype=torch.float32)

        net = UnifiedGraphRAGIM(
            struct_dim=feats.shape[1], text_dim=text_emb.shape[1],
            graph_hidden=cfg.model.graph_hidden, graph_layers=cfg.model.graph_layers,
            attn_dim=cfg.model.attn_dim, attn_slots=cfg.model.attn_slots,
            dropout=cfg.model.dropout, mlp_hidden=cfg.model.mlp_hidden,
            mlp_dropout=cfg.model.mlp_dropout)
        trainer = GraphRAGIMTrainer(cfg, logger=self.logger)
        trainer.fit(net, cg, x_struct, edge_index, h_txt)
        with torch.no_grad():
            return net(x_struct, edge_index, h_txt).cpu().numpy()
