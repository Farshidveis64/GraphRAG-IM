try:
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def influence_bce_loss(preds, labels, mask=None):
    """Binary cross-entropy between predicted influence probabilities and labels.

    Classification (is-this-node-influential) empirically separates top
    influencers better than regressing sigma({v}). L2 regularization (lambda)
    is applied via the optimizer's weight_decay, following the paper.
    """
    if not _TORCH_OK:  # pragma: no cover
        raise ImportError("influence_bce_loss requires torch.")
    criterion = nn.BCELoss()
    labels = labels.float()
    if mask is not None:
        return criterion(preds[mask], labels[mask])
    return criterion(preds, labels)
