"""Render the pipeline figure used at the top of the README.

    python docs/figures/make_hero.py

Writes hero_pipeline.png and hero_pipeline-dark.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THEMES = {
    "light": dict(bg="white", ink="#1c2530", muted="#5b6875", line="#b9c3cf",
                  fill="#eef2f6", s1="#4a7fb5", s2="#c8683f", out="#3f7d5a",
                  f1="#eaf1f8", f2="#fbeee7", fo="#e9f2ec"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", line="#3d444d",
                 fill="#161b22", s1="#6ea8dd", s2="#e08a5c", out="#5aa87a",
                 f1="#12202f", f2="#2a1c14", fo="#12241a"),
}

HERE = os.path.dirname(os.path.abspath(__file__))


def render(theme, out):
    T = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 4.3), dpi=170)
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 43)
    ax.axis("off")
    fig.patch.set_facecolor(T["bg"])

    def box(x, y, w, h, title, sub, edge=None, face=None, tcol=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    linewidth=1.4, edgecolor=edge or T["line"],
                                    facecolor=face or T["fill"], zorder=2))
        ax.text(x + w / 2, y + h / 2 + 1.9, title, ha="center", va="center",
                fontsize=11.5, color=tcol or T["ink"], fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                fontsize=9.2, color=T["muted"], zorder=3)

    def arrow(x0, y0, x1, y1, c=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.5, color=c or T["line"], shrinkA=0, shrinkB=0, zorder=1))

    W, H = 26.0, 11.0
    R1, R2 = 27.0, 5.0          # two rows, so no horizontal chain exceeds three boxes
    X0 = 3.0

    box(X0, R1, W, H, "UDP stream", "50 Hz, 6 watch channels")
    arrow(X0 + W, R1 + H / 2, X0 + W + 3, R1 + H / 2)
    box(X0 + W + 3, R1, W, H, "Stage 1", "binary entry detector", T["s1"], T["f1"])
    arrow(X0 + 2 * W + 3, R1 + H / 2, X0 + 2 * W + 6, R1 + H / 2, T["s1"])
    box(X0 + 2 * W + 6, R1, W, H, "Buffer 2.5 s", "collected on detection", T["s1"], T["f1"])

    # drop to the second row
    cx = X0 + 2 * W + 6 + W / 2
    mid = (R1 + R2 + H) / 2
    ax.plot([cx, cx], [R1, mid], color=T["line"], lw=1.5, zorder=1)
    ax.plot([X0 + W + 3 + W / 2, cx], [mid, mid], color=T["line"], lw=1.5, zorder=1)
    arrow(X0 + W + 3 + W / 2, mid, X0 + W + 3 + W / 2, R2 + H, T["s2"])

    box(X0 + W + 3, R2, W, H, "Stage 2", "15-class, best window", T["s2"], T["f2"])
    arrow(X0 + W + 3, R2 + H / 2, X0 + W, R2 + H / 2, T["out"])
    box(X0, R2, W, H, "Gesture", "one of fifteen", T["out"], T["fo"], T["out"])

    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    render("light", os.path.join(HERE, "hero_pipeline.png"))
    render("dark", os.path.join(HERE, "hero_pipeline-dark.png"))
