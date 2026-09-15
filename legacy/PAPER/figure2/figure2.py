"""
Gera Figura 2 do paper:
Identificabilidade k-mer para diferentes K e C.

Painéis:
A - Heatmap do número de colisões
B - Heatmap de log10(d_min)
C - Fração de assinaturas únicas
D - Distância mínima por número de canais

Saídas:
figure_2_identifiability.svg
figure_2_identifiability.png

Observação:
Para K=9, existem 4^9 = 262144 k-mers.
O cálculo ainda é viável, mas pode demorar um pouco dependendo do computador.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "svg.fonttype": "none",
})


BASES = ["A", "T", "G", "C"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

K_VALUES = [3, 5, 7, 9]
C_VALUES = [1, 2, 3, 4]

SEED = 7
ROUND_DECIMALS = 8


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left"
    )


def one_hot_kmer(kmer):
    X = np.zeros((len(kmer), 4))

    for i, b in enumerate(kmer):
        X[i, BASE_TO_IDX[b]] = 1

    return X


def generate_kernel(k, c, seed):
    rng = np.random.default_rng(seed + 100 * k + 10 * c)
    return rng.random((k, 4, c))


def kmer_signature(kmer, kernel):
    X = one_hot_kmer(kmer)
    _, _, c = kernel.shape

    sig = np.zeros(c)

    for ch in range(c):
        sig[ch] = np.sum(X * kernel[:, :, ch])

    return sig


def compute_signatures(k, c):
    kernel = generate_kernel(k, c, SEED)
    kmers = ["".join(p) for p in product(BASES, repeat=k)]

    signatures = np.zeros((len(kmers), c))

    for i, km in enumerate(kmers):
        signatures[i] = kmer_signature(km, kernel)

    return signatures


def min_pairwise_distance(signatures):
    """
    Calcula distância mínima entre assinaturas.

    Para até 262144 pontos, calcula em blocos para evitar matriz gigante.
    """

    n = signatures.shape[0]
    min_dist_sq = np.inf
    block = 2048

    for i in range(0, n, block):
        A = signatures[i:i + block]

        for j in range(i, n, block):
            B = signatures[j:j + block]

            diff = A[:, None, :] - B[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)

            if i == j:
                np.fill_diagonal(dist_sq, np.inf)

            local_min = np.min(dist_sq)

            if local_min < min_dist_sq:
                min_dist_sq = local_min

    return np.sqrt(min_dist_sq)


def analyze_identifiability():
    collisions = np.zeros((len(K_VALUES), len(C_VALUES)))
    unique_fraction = np.zeros((len(K_VALUES), len(C_VALUES)))
    dmin = np.zeros((len(K_VALUES), len(C_VALUES)))

    for i, k in enumerate(K_VALUES):
        for j, c in enumerate(C_VALUES):
            print(f"Calculando K={k}, C={c}...")

            signatures = compute_signatures(k, c)

            rounded = [
                tuple(np.round(row, ROUND_DECIMALS))
                for row in signatures
            ]

            total = len(rounded)
            unique = len(set(rounded))

            collisions[i, j] = total - unique
            unique_fraction[i, j] = unique / total
            dmin[i, j] = min_pairwise_distance(signatures)

            print(
                f"  total={total}, unique={unique}, "
                f"collisions={int(collisions[i, j])}, "
                f"dmin={dmin[i, j]:.6e}"
            )

    return collisions, unique_fraction, dmin


def heatmap(ax, data, title, cbar_label, fmt, cmap="viridis"):
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xticks(range(len(C_VALUES)))
    ax.set_xticklabels(C_VALUES, fontsize=20, fontweight="bold")
    ax.set_yticks(range(len(K_VALUES)))
    ax.set_yticklabels(K_VALUES, fontsize=20, fontweight="bold")

    ax.set_xlabel("channels C", fontsize=20, fontweight="bold")
    ax.set_ylabel("window K", fontsize=20, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i, fmt(data[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white"
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=14, width=2)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(width=2, length=6)


def panel_d(ax, dmin):
    add_panel_label(ax, "D")

    for idx, k in enumerate(K_VALUES):
        y = dmin[idx, :]
        ax.plot(
            C_VALUES,
            y,
            marker="o",
            linewidth=4,
            markersize=10,
            label=f"K={k}"
        )

    ax.set_title("Minimum separation", fontsize=22, fontweight="bold")
    ax.set_xlabel("channels C", fontsize=20, fontweight="bold")
    ax.set_ylabel("d_min", fontsize=20, fontweight="bold")

    ax.set_xticks(C_VALUES)
    ax.set_xticklabels(C_VALUES, fontsize=18, fontweight="bold")

    ax.set_yscale("log")

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.legend(
        fontsize=14,
        frameon=False,
        prop={"weight": "bold", "size": 14}
    )

    ax.grid(True, linewidth=1.5, alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    collisions, unique_fraction, dmin = analyze_identifiability()

    log_dmin = np.log10(dmin + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    add_panel_label(ax, "A")
    heatmap(
        ax,
        collisions,
        "Collision count",
        "collisions",
        fmt=lambda x: f"{int(x)}",
        cmap="magma"
    )

    ax = axes[0, 1]
    add_panel_label(ax, "B")
    heatmap(
        ax,
        log_dmin,
        "Log minimum distance",
        "log10 d_min",
        fmt=lambda x: f"{x:.1f}",
        cmap="viridis"
    )

    ax = axes[1, 0]
    add_panel_label(ax, "C")
    heatmap(
        ax,
        unique_fraction,
        "Unique signature fraction",
        "unique fraction",
        fmt=lambda x: f"{x:.2f}",
        cmap="plasma"
    )

    panel_d(axes[1, 1], dmin)

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_2_identifiability.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_2_identifiability.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    np.savez(
        "figure_2_identifiability_data.npz",
        K_VALUES=np.array(K_VALUES),
        C_VALUES=np.array(C_VALUES),
        collisions=collisions,
        unique_fraction=unique_fraction,
        dmin=dmin,
        log_dmin=log_dmin
    )

    print("\nArquivos salvos:")
    print("figure_2_identifiability.svg")
    print("figure_2_identifiability.png")
    print("figure_2_identifiability_data.npz")


if __name__ == "__main__":
    main()
