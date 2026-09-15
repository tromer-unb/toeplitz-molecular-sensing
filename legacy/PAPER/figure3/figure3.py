"""
Gera Figura 3 do paper:
Aprendizado/calibração do kernel Toeplitz multicanal.

Painéis:
A - Kernel verdadeiro, canal 1
B - Kernel aprendido, canal 1
C - Erro de predição do sinal vs número de sequências de treino
D - Acurácia de reconstrução vs número de sequências de treino

Saídas:
figure_3_kernel_learning.svg
figure_3_kernel_learning.png
figure_3_kernel_learning_data.npz
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

SEED = 7

N_TEST = 120
TRAIN_LEN = 150
K = 5
C = 3
NOISE = 0.05

TRAIN_SIZES = [5, 10, 25, 50, 100, 200, 500]
REPEATS = 15

TRUE_KERNEL = np.random.default_rng(SEED).random((K, 4, C))


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left"
    )


def generate_random_dna(n, rng):
    return "".join(rng.choice(BASES, size=n))


def one_hot_dna(seq):
    X = np.zeros((len(seq), 4))

    for i, b in enumerate(seq):
        X[i, BASE_TO_IDX[b]] = 1

    return X


def generate_signal(seq, kernel, noise=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X = one_hot_dna(seq)
    k, _, c = kernel.shape
    pad = k // 2

    Xpad = np.pad(X, ((pad, pad), (0, 0)), mode="constant")

    signal = []

    for i in range(len(seq)):
        window = Xpad[i:i + k]
        value = np.zeros(c)

        for ch in range(c):
            value[ch] = np.sum(window * kernel[:, :, ch])

        signal.append(value)

    signal = np.array(signal)

    if noise > 0:
        signal += rng.normal(0, noise, size=signal.shape)

    return signal


def build_design_matrix(seq, k):
    X = one_hot_dna(seq)
    pad = k // 2

    Xpad = np.pad(X, ((pad, pad), (0, 0)), mode="constant")

    rows = []

    for i in range(len(seq)):
        window = Xpad[i:i + k]
        rows.append(window.flatten())

    return np.array(rows)


def learn_kernel_from_data(seqs, signals, k, c):
    A_all = []
    Y_all = []

    for seq, sig in zip(seqs, signals):
        A_all.append(build_design_matrix(seq, k))
        Y_all.append(sig)

    A_all = np.vstack(A_all)
    Y_all = np.vstack(Y_all)

    W, _, _, _ = np.linalg.lstsq(A_all, Y_all, rcond=None)

    return W.reshape(k, 4, c)


def kmer_signal(kmer, kernel):
    X = one_hot_dna(kmer)
    _, _, c = kernel.shape

    out = np.zeros(c)

    for ch in range(c):
        out[ch] = np.sum(X * kernel[:, :, ch])

    return out


def decode_signal_viterbi(signal, kernel):
    k = kernel.shape[0]
    pad = k // 2

    all_kmers = ["".join(p) for p in product(BASES, repeat=k)]
    kmer_values = {km: kmer_signal(km, kernel) for km in all_kmers}

    def score(km, s):
        diff = kmer_values[km] - s
        return np.sum(diff ** 2)

    dp = [{} for _ in range(len(signal))]
    back = [{} for _ in range(len(signal))]

    for km in all_kmers:
        dp[0][km] = score(km, signal[0])
        back[0][km] = None

    for t in range(1, len(signal)):
        for km in all_kmers:
            prefix = km[:-1]

            best_prev = None
            best_score = float("inf")

            for prev in all_kmers:
                if prev[1:] == prefix:
                    val = dp[t - 1][prev] + score(km, signal[t])

                    if val < best_score:
                        best_score = val
                        best_prev = prev

            dp[t][km] = best_score
            back[t][km] = best_prev

    last = min(dp[-1], key=dp[-1].get)

    path = [last]

    for t in range(len(signal) - 1, 0, -1):
        last = back[t][last]
        path.append(last)

    path = path[::-1]

    seq = path[0]

    for km in path[1:]:
        seq += km[-1]

    return seq[pad:pad + len(signal)]


def accuracy(real, pred):
    return np.mean([a == b for a, b in zip(real, pred)])


def train_kernel(num_train, train_len, noise, rng):
    seqs = []
    signals = []

    for _ in range(num_train):
        seq = generate_random_dna(train_len, rng)
        sig = generate_signal(seq, TRUE_KERNEL, noise=noise, rng=rng)

        seqs.append(seq)
        signals.append(sig)

    return learn_kernel_from_data(seqs, signals, K, C)


def signal_prediction_mse(seq, true_kernel, learned_kernel):
    s_true = generate_signal(seq, true_kernel, noise=0.0, rng=np.random.default_rng(1))
    s_pred = generate_signal(seq, learned_kernel, noise=0.0, rng=np.random.default_rng(2))

    return np.mean((s_true - s_pred) ** 2)


def evaluate_learning():
    mse_mean = []
    mse_std = []
    acc_mean = []
    acc_std = []

    example_learned_kernel = None

    pad = K // 2

    for train_size in TRAIN_SIZES:
        print(f"Treinando com {train_size} sequências...")

        mses = []
        accs = []

        for rep in range(REPEATS):
            rng = np.random.default_rng(SEED + 1000 * train_size + rep)

            learned = train_kernel(
                num_train=train_size,
                train_len=TRAIN_LEN,
                noise=NOISE,
                rng=rng
            )

            if train_size == TRAIN_SIZES[-1] and rep == 0:
                example_learned_kernel = learned.copy()

            test_seq = generate_random_dna(N_TEST, rng)
            test_signal = generate_signal(test_seq, TRUE_KERNEL, noise=NOISE, rng=rng)

            pred_seq = decode_signal_viterbi(test_signal, learned)

            mse = signal_prediction_mse(test_seq, TRUE_KERNEL, learned)
            acc = accuracy(test_seq[pad:-pad], pred_seq[pad:-pad])

            mses.append(mse)
            accs.append(acc)

        mse_mean.append(np.mean(mses))
        mse_std.append(np.std(mses))
        acc_mean.append(np.mean(accs))
        acc_std.append(np.std(accs))

        print(
            f"  MSE={np.mean(mses):.6e} ± {np.std(mses):.6e} | "
            f"ACC={np.mean(accs):.2%} ± {np.std(accs):.2%}"
        )

    return (
        np.array(mse_mean),
        np.array(mse_std),
        np.array(acc_mean),
        np.array(acc_std),
        example_learned_kernel
    )


def kernel_heatmap(ax, kernel_channel, title):
    im = ax.imshow(kernel_channel, cmap="viridis", aspect="auto")

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(BASES, fontsize=20, fontweight="bold")
    ax.set_yticks(range(K))
    ax.set_yticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=20, fontweight="bold")

    ax.set_xlabel("base", fontsize=20, fontweight="bold")
    ax.set_ylabel("offset", fontsize=20, fontweight="bold")

    for i in range(kernel_channel.shape[0]):
        for j in range(kernel_channel.shape[1]):
            ax.text(
                j,
                i,
                f"{kernel_channel[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white"
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14, width=2)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(width=2, length=6)


def learning_curve_mse(ax, train_sizes, mean, std):
    ax.set_title("Signal prediction error", fontsize=22, fontweight="bold")

    ax.errorbar(
        train_sizes,
        mean,
        yerr=std,
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("training sequences", fontsize=20, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=20, fontweight="bold")

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True, linewidth=1.5, alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def learning_curve_accuracy(ax, train_sizes, mean, std):
    ax.set_title("Sequence reconstruction", fontsize=22, fontweight="bold")

    ax.errorbar(
        train_sizes,
        100 * mean,
        yerr=100 * std,
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6
    )

    ax.set_xscale("log")
    ax.set_ylim(80, 101)

    ax.set_xlabel("training sequences", fontsize=20, fontweight="bold")
    ax.set_ylabel("accuracy (%)", fontsize=20, fontweight="bold")

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True, linewidth=1.5, alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    mse_mean, mse_std, acc_mean, acc_std, learned_kernel = evaluate_learning()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    add_panel_label(ax, "A")
    kernel_heatmap(
        ax,
        TRUE_KERNEL[:, :, 0],
        "True kernel"
    )

    ax = axes[0, 1]
    add_panel_label(ax, "B")
    kernel_heatmap(
        ax,
        learned_kernel[:, :, 0],
        "Learned kernel"
    )

    ax = axes[1, 0]
    add_panel_label(ax, "C")
    learning_curve_mse(
        ax,
        TRAIN_SIZES,
        mse_mean,
        mse_std
    )

    ax = axes[1, 1]
    add_panel_label(ax, "D")
    learning_curve_accuracy(
        ax,
        TRAIN_SIZES,
        acc_mean,
        acc_std
    )

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_3_kernel_learning.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_3_kernel_learning.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    np.savez(
        "figure_3_kernel_learning_data.npz",
        train_sizes=np.array(TRAIN_SIZES),
        mse_mean=mse_mean,
        mse_std=mse_std,
        acc_mean=acc_mean,
        acc_std=acc_std,
        true_kernel=TRUE_KERNEL,
        learned_kernel=learned_kernel
    )

    print("\nArquivos salvos:")
    print("figure_3_kernel_learning.svg")
    print("figure_3_kernel_learning.png")
    print("figure_3_kernel_learning_data.npz")


if __name__ == "__main__":
    main()
