from typing import List

import numpy as np


def build_dpp_kernel(scores: np.ndarray, embeddings: np.ndarray,
                     theta: float = 0.1) -> np.ndarray:
    """Quality-times-dissimilarity DPP kernel L_ij = s_i s_j exp(-theta ||h_i-h_j||^2)."""
    scores = np.asarray(scores, dtype=np.float64).clip(1e-8, None)
    emb = np.asarray(embeddings, dtype=np.float64)
    sq = np.sum(emb ** 2, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (emb @ emb.T)
    np.maximum(dist2, 0.0, out=dist2)
    return (scores[:, None] * scores[None, :]) * np.exp(-theta * dist2)


def dpp_greedy_map(kernel: np.ndarray, k: int) -> List[int]:
    """Greedy MAP inference for a DPP (Chen et al., 2018) via incremental Cholesky.

    Limiting behaviour of theta in build_dpp_kernel: theta -> inf recovers
    top-k by quality; theta -> 0 is rank-1 (a single representative).
    """
    n = kernel.shape[0]
    k = min(k, n)
    cis = np.zeros((k, n), dtype=np.float64)
    d2 = np.copy(np.diag(kernel)).astype(np.float64)

    selected: List[int] = [int(np.argmax(d2))]
    for it in range(1, k):
        prev = selected[-1]
        d_prev = np.sqrt(max(d2[prev], 1e-12))
        inner = cis[:it, prev] @ cis[:it, :] if it > 0 else 0.0
        ei = (kernel[prev, :] - inner) / d_prev
        cis[it - 1, :] = ei
        d2 = d2 - ei ** 2
        d2[selected] = -np.inf
        nxt = int(np.argmax(d2))
        if not np.isfinite(d2[nxt]) or d2[nxt] <= 1e-12:
            break
        selected.append(nxt)
    return selected


def topk_select(scores: np.ndarray, k: int) -> List[int]:
    """Indices of the k highest scores (descending)."""
    scores = np.asarray(scores)
    k = min(k, scores.shape[0])
    return list(np.argsort(scores)[::-1][:k].astype(int))


def select_seeds(scores: np.ndarray, embeddings: np.ndarray, k: int,
                 method: str = "dpp", theta: float = 0.1) -> List[int]:
    """Select k seeds by content-conditioned DPP or the top-k baseline."""
    if method == "topk":
        return topk_select(scores, k)
    if method == "dpp":
        return dpp_greedy_map(build_dpp_kernel(scores, embeddings, theta), k)
    raise ValueError(f"Unknown selection method {method!r}")
