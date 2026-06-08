#!/usr/bin/env python3
"""Train the GraphRAG-IM neural influence scorer (requires the 'neural' extras).

Generates weak influence labels under the content-conditioned IC model, fits the
GraphSAGE + cross-attention + MLP scorer with BCE (+ L2 via weight decay), and
saves the weights and per-node scores under <output_dir>.

Example
-------
    python scripts/train.py --config configs/default.yaml --output_dir outputs/model
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_dataset  # noqa: E402
from data.data_utils import structural_features  # noqa: E402
from models.model_utils import OfflineTextEncoder, seed_everything  # noqa: E402
from utils.config import GraphRAGIMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
LOG = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--output_dir", default="outputs/model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = GraphRAGIMConfig.from_yaml(args.config) if args.config else GraphRAGIMConfig()
    cfg.seed = args.seed

    try:
        import torch
    except ImportError:
        LOG.error("torch is not installed. Install the 'neural' extras to train.")
        sys.exit(1)

    from models.unified_model import UnifiedGraphRAGIM
    from training.trainer import GraphRAGIMTrainer

    cg = load_dataset(args.dataset, root=cfg.data.root, seed=args.seed)
    cg.ensure_embeddings(dim=cfg.data.embedding_dim, seed=args.seed)
    LOG.info("Graph: %d nodes, %d edges", cg.n_nodes, cg.n_edges)

    feats = structural_features(cg.graph)
    edges = np.array(list(cg.graph.edges())).T
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    x_struct = torch.tensor(feats, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long)
    summaries = [f"node {v}: {cg.texts[v]}" for v in range(cg.n_nodes)]
    h_txt = torch.tensor(
        OfflineTextEncoder(dim=cfg.model.text_dim, seed=args.seed).encode(summaries),
        dtype=torch.float32)

    net = UnifiedGraphRAGIM(
        struct_dim=feats.shape[1], text_dim=cfg.model.text_dim,
        graph_hidden=cfg.model.graph_hidden, graph_layers=cfg.model.graph_layers,
        attn_dim=cfg.model.attn_dim, attn_slots=cfg.model.attn_slots,
        dropout=cfg.model.dropout, mlp_hidden=cfg.model.mlp_hidden,
        mlp_dropout=cfg.model.mlp_dropout)
    trainer = GraphRAGIMTrainer(cfg, logger=LOG)
    trainer.fit(net, cg, x_struct, edge_index, h_txt)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.output_dir, "graphrag_im.pt"))
    with torch.no_grad():
        scores = net(x_struct, edge_index, h_txt).cpu().numpy()
    np.save(os.path.join(args.output_dir, "scores.npy"), scores)
    LOG.info("Saved model + scores to %s/", args.output_dir)


if __name__ == "__main__":
    main()
