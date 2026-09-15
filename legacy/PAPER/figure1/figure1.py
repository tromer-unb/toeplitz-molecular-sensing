"""
Gera Figura 1 do paper:
A - DNA one-hot
B - Janela local k-mer
C - Kernel Toeplitz multicanal
D - Pipeline completo

Saída:
figure_1_toeplitz_model.svg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "svg.fonttype": "none",
})


def add_panel_label(ax, label):
    ax.text(
        -0.08, 1.08, label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left"
    )


def arrow(ax, xy1, xy2):
    ax.add_patch(
        FancyArrowPatch(
            xy1, xy2,
            arrowstyle="-|>",
            mutation_scale=25,
            linewidth=3,
            color="black"
        )
    )


def panel_a(ax):
    ax.set_title("DNA one-hot", fontsize=22, fontweight="bold")
    add_panel_label(ax, "A")

    bases = ["A", "T", "G", "C"]
    seq = ["A", "C", "G", "T", "C"]

    mat = np.zeros((4, len(seq)))
    base_to_row = {"A": 0, "T": 1, "G": 2, "C": 3}

    for j, b in enumerate(seq):
        mat[base_to_row[b], j] = 1

    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1)

    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels(seq, fontsize=20, fontweight="bold")
    ax.set_yticks(range(4))
    ax.set_yticklabels(bases, fontsize=20, fontweight="bold")

    ax.set_xlabel("position", fontsize=18, fontweight="bold")
    ax.set_ylabel("base", fontsize=18, fontweight="bold")

    for i in range(4):
        for j in range(len(seq)):
            ax.text(
                j, i, int(mat[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="red" if mat[i, j] == 1 else "black"
            )

    ax.tick_params(width=2, length=6)

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_b(ax):
    ax.set_title("Local k-mer window", fontsize=22, fontweight="bold")
    add_panel_label(ax, "B")
    ax.axis("off")

    seq = list("ACGTCAAG")
    x0 = 0.5
    y = 0.55
    dx = 0.75

    for i, b in enumerate(seq):
        x = x0 + i * dx

        color = "#f0f0f0"
        edge = "black"
        lw = 2

        if 2 <= i <= 6:
            color = "#d9ecff"
            edge = "#005bbb"
            lw = 3

        ax.add_patch(Rectangle((x, y), 0.55, 0.35, facecolor=color, edgecolor=edge, linewidth=lw))
        ax.text(x + 0.275, y + 0.175, b, ha="center", va="center", fontsize=22, fontweight="bold")

    ax.text(2.6, 0.25, "K = 5", fontsize=24, fontweight="bold")
    arrow(ax, (3.0, 0.45), (3.0, 0.55))

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.2)


def panel_c(ax):
    ax.set_title("Multichannel Toeplitz kernel", fontsize=22, fontweight="bold")
    add_panel_label(ax, "C")

    kernel = np.array([
        [0.1, 0.8, 0.3, 1.2],
        [1.1, 0.2, 1.5, 0.4],
        [0.6, 1.3, 0.1, 0.9],
        [1.4, 0.3, 0.7, 1.1],
        [0.2, 1.0, 1.4, 0.2],
    ])

    im = ax.imshow(kernel, cmap="viridis")

    ax.set_xticks(range(4))
    ax.set_xticklabels(["A", "T", "G", "C"], fontsize=20, fontweight="bold")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=20, fontweight="bold")

    ax.set_xlabel("base", fontsize=18, fontweight="bold")
    ax.set_ylabel("offset", fontsize=18, fontweight="bold")

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax.text(
                j, i, f"{kernel[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white"
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14, width=2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_d(ax):
    ax.set_title("Learning, decoding, residual", fontsize=22, fontweight="bold")
    add_panel_label(ax, "D")
    ax.axis("off")

    boxes = {
        "DNA\nknown": (0.05, 0.68),
        "signal": (0.35, 0.68),
        "learn\nkernel": (0.65, 0.68),
        "new\nsignal": (0.05, 0.25),
        "Viterbi": (0.35, 0.25),
        "DNA\npred": (0.65, 0.25),
        "residual": (0.35, -0.08),
        "modification": (0.65, -0.08),
    }

    for text, (x, y) in boxes.items():
        ax.add_patch(
            Rectangle(
                (x, y), 0.22, 0.18,
                facecolor="#f3f3f3",
                edgecolor="black",
                linewidth=3
            )
        )
        ax.text(
            x + 0.11, y + 0.09, text,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold"
        )

    arrow(ax, (0.27, 0.77), (0.35, 0.77))
    arrow(ax, (0.57, 0.77), (0.65, 0.77))

    arrow(ax, (0.27, 0.34), (0.35, 0.34))
    arrow(ax, (0.57, 0.34), (0.65, 0.34))

    arrow(ax, (0.46, 0.25), (0.46, 0.10))
    arrow(ax, (0.57, 0.01), (0.65, 0.01))

    ax.text(0.50, 0.55, r"$s_i=\sum_j K_j x_{i+j}$", fontsize=24, fontweight="bold", ha="center")
    ax.text(0.50, 0.13, r"$r_i=\|s_i-\hat{s}_i\|$", fontsize=24, fontweight="bold", ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.18, 1.0)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_1_toeplitz_model.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_1_toeplitz_model.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    print("Figura salva como:")
    print("figure_1_toeplitz_model.svg")
    print("figure_1_toeplitz_model.png")


if __name__ == "__main__":
    main()
