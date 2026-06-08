import numpy as np

from inference.diffusion import ContentICModel
from training.labels import generate_influence_labels
from training.losses import influence_bce_loss


class GraphRAGIMTrainer:
    """Trains the influence scorer against weak labels (BCE + L2).

    Labels come from top-spread membership under the content-conditioned IC
    model (the same process used at evaluation). The neural path requires torch;
    label generation itself is dependency-free.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

    def _log(self, msg, *args):
        if self.logger:
            self.logger.info(msg, *args)

    def make_labels(self, cg, rng=None) -> np.ndarray:
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng(cfg.seed)
        model = ContentICModel(cg.graph, cg.embeddings, p0=cfg.diffusion.p0,
                               rho_mode=cfg.diffusion.rho_mode,
                               rho_temperature=cfg.diffusion.rho_temperature)
        labels = generate_influence_labels(
            model, k=cfg.selection.k, n_configs=cfg.training.label_n_configs,
            n_sims=cfg.training.label_sims, top_pct=cfg.training.label_top_pct,
            membership=cfg.training.label_membership, rng=rng)
        self._log("Generated labels: %d positives / %d nodes",
                  int(labels.sum()), len(labels))
        return labels

    def fit(self, net, cg, x_struct, edge_index, h_txt, labels=None):  # pragma: no cover
        import torch

        cfg = self.config
        if labels is None:
            labels = self.make_labels(cg)
        y = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr,
                                     weight_decay=cfg.training.weight_decay)
        net.train()
        for epoch in range(cfg.training.epochs):
            optimizer.zero_grad()
            preds = net(x_struct, edge_index, h_txt)
            loss = influence_bce_loss(preds, y)
            loss.backward()
            optimizer.step()
            if epoch % cfg.logging.log_interval == 0 or epoch == cfg.training.epochs - 1:
                self._log("epoch %3d | bce %.4f", epoch, float(loss))
        net.eval()
        return net
