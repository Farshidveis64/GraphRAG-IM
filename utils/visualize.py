import os
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "steel": "#3D6E8E", "teal": "#2A9D8F", "ours": "#E2603B",
    "sand": "#E3B23C", "slate": "#9AA7B5",
}
METHOD_COLORS = {
    "ToupleGDD": PALETTE["steel"],
    "ToupleGDD+Text": PALETTE["teal"],
    "GraphRAG-IM": PALETTE["ours"],
}
_METHODS = ["ToupleGDD", "ToupleGDD+Text", "GraphRAG-IM"]


def set_publication_style():
    """Apply a clean, vector-friendly Matplotlib style (idempotent)."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.axisbelow": True, "legend.fontsize": 9, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(PALETTE.values()))


def _save(fig, out_path):
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        fig.savefig(os.path.splitext(out_path)[0] + ".png")


def plot_diversity(results: Mapping[str, Mapping[str, float]], out_path: str = None):
    """Three-panel seed-diversity comparison across methods."""
    set_publication_style()
    panels = [
        ("emb_distance", "Emb. Dist. " + r"$\uparrow$", "Pairwise content distance"),
        ("community_coverage", "Comm. Cov. " + r"$\uparrow$", "Communities covered"),
        ("cascade_overlap", "Cascade Overlap " + r"$\downarrow$", "Mean pairwise ovl."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.7))
    colors = [METHOD_COLORS[m] for m in _METHODS]
    for ax, (key, title, ylab) in zip(axes, panels):
        vals = [results[m][key] for m in _METHODS]
        ax.bar(range(len(_METHODS)), vals, color=colors, width=0.62,
               edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(_METHODS)))
        ax.set_xticklabels(["1", "2", "3"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 5 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in _METHODS]
    labels = [f"{i + 1}: {m}" for i, m in enumerate(_METHODS)]
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_sensitivity(k_values: Sequence[int],
                     spread_by_k: Mapping[str, Sequence[float]],
                     p_values: Sequence[float],
                     spread_by_p: Mapping[str, Sequence[float]],
                     out_path: str = None):
    """Two-panel sensitivity to seed budget k and propagation probability p."""
    set_publication_style()
    fig, (ax_k, ax_p) = plt.subplots(1, 2, figsize=(8.0, 3.0))
    styles = {"ToupleGDD+Text": ("--", "s", PALETTE["steel"]),
              "GraphRAG-IM": ("-", "o", PALETTE["ours"])}
    for method, (ls, mk, col) in styles.items():
        lab = "GraphRAG-IM (Ours)" if method == "GraphRAG-IM" else method
        ax_k.plot(k_values, spread_by_k[method], ls, marker=mk, color=col, label=lab)
    ax_k.set_title(r"(a) Varying $k$  ($p=0.1$)")
    ax_k.set_xlabel(r"Seed budget $k$")
    ax_k.set_ylabel("Influence spread")
    ax_k.legend()
    for method, (ls, mk, col) in styles.items():
        ax_p.plot(p_values, spread_by_p[method], ls, marker=mk, color=col, label=method)
    ax_p.set_title(r"(b) Varying $p$  ($k=50$)")
    ax_p.set_xlabel(r"Propagation probability $p$")
    ax_p.set_ylabel("Influence spread")
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_ablation(ablation: Mapping[str, Mapping[str, float]], out_path: str = None):
    """Bar chart of mean signed spread drop per removed component, with error bars."""
    set_publication_style()
    components = list(ablation.keys())
    means = [ablation[c]["mean"] for c in components]
    stds = [ablation[c]["std"] for c in components]
    x = np.arange(len(components))
    colors = [PALETTE["steel"], PALETTE["teal"], PALETTE["ours"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6,
                  color=[colors[i % len(colors)] for i in range(len(components))],
                  edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.2, "ecolor": PALETTE["slate"]})
    for rect, m in zip(bars, means):
        va = "bottom" if m >= 0 else "top"
        ax.text(rect.get_x() + rect.get_width() / 2, m + (0.02 if m >= 0 else -0.02),
                f"{m:+.1f}%", ha="center", va=va, fontsize=8)
    ax.axhline(0.0, color=PALETTE["slate"], linewidth=0.8)
    lo = min(0.0, min(m - s for m, s in zip(means, stds)))
    hi = max(m + s for m, s in zip(means, stds))
    pad = 0.15 * (hi - lo) + 0.5
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Ablation Study (mean over random graphs)")
    ax.set_ylabel(r"Influence-spread drop ($\Delta$%)")
    ax.set_xlabel("Removed component")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    fig.tight_layout()
    _save(fig, out_path)
    return fig
