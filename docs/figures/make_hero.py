"""Draw the README hero: the two-stage pipeline.

    python docs/figures/make_hero.py

Writes hero_pipeline.png and hero_pipeline-dark.png. No numbers are committed to
this repository, so the pipeline is the hero rather than a results figure.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
sys.path.insert(0, HERE)

import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def pipeline(T):
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 4.3))
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 43)
    ax.axis("off")

    def box(x, y, w, h, title, sub, edge=None, face=None, tcol=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    linewidth=1.4, edgecolor=edge or T["line"],
                                    facecolor=face or T["fill"], zorder=2))
        ax.text(x + w / 2, y + h / 2 + 1.9, title, ha="center", va="center",
                fontsize=figstyle.TITLE, color=tcol or T["ink"], fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                fontsize=figstyle.SMALL, color=T["muted"], zorder=3)

    def arrow(x0, y0, x1, y1, c=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.5, color=c or T["line"],
                                     shrinkA=0, shrinkB=0, zorder=1))

    def line(xs, ys, c=None):
        ax.plot(xs, ys, color=c or T["line"], linewidth=1.5, zorder=1)

    G, GF, D, DF = T["green"], T["green_fill"], T["gold"], T["gold_fill"]
    W, H = 26.0, 11.0
    R1, R2 = 27.0, 5.0          # two rows, so no horizontal chain exceeds three boxes
    X0 = 3.0

    box(X0, R1, W, H, "UDP stream", "50 Hz, 6 watch channels")
    arrow(X0 + W, R1 + H / 2, X0 + W + 3, R1 + H / 2)
    box(X0 + W + 3, R1, W, H, "Stage 1", "entry detector, always on", G, GF)
    arrow(X0 + 2 * W + 3, R1 + H / 2, X0 + 2 * W + 6, R1 + H / 2, G)
    box(X0 + 2 * W + 6, R1, W, H, "Buffer 2.5 s", "opened on detection", G, GF)

    # drop to the second row
    cx = X0 + 2 * W + 6 + W / 2
    mid = (R1 + R2 + H) / 2
    line([cx, cx], [R1, mid])
    line([X0 + W + 3 + W / 2, cx], [mid, mid])
    arrow(X0 + W + 3 + W / 2, mid, X0 + W + 3 + W / 2, R2 + H, D)

    box(X0 + W + 3, R2, W, H, "Stage 2", "15-class, best window", D, DF)
    arrow(X0 + W + 3, R2 + H / 2, X0 + W, R2 + H / 2, G)
    box(X0, R2, W, H, "Gesture", "one of fifteen", G, GF, G)
    return fig


if __name__ == "__main__":
    figstyle.save_both(pipeline, HERE + "hero_pipeline")
