"""
Gera Figura 4 do paper:
Robustez ao ruído do modelo Toeplitz multicanal.

Painéis:
A - Acurácia vs ruído
B - Erro de predição do sinal vs ruído
C - Diferença entre kernel verdadeiro e aprendido
D - Distribuição de acurácia por nível de ruído

Saídas:
figure_4_noise_robustness.svg
figure_4_noise_robustness.png
figure_4_noise_robustness_data.npz
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
NUM_TRAIN = 150

K = 5
C = 3

NOISE_LEVELS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30]
REPEATS = 20

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
        rows.append(Xpad[i:i + k].flatten())

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


def train_kernel(noise, rng):
    seqs = []
    signals = []

    for _ in range(NUM_TRAIN):
        seq = generate_random_dna(TRAIN_LEN, rng)
        sig = generate_signal(seq, TRUE_KERNEL, noise=noise, rng=rng)
        seqs.append(seq)
        signals.append(sig)

    return learn_kernel_from_data(seqs, signals, K, C)


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


def signal_prediction_mse(seq, true_kernel, learned_kernel):
    s_true = generate_signal(seq, true_kernel, noise=0.0, rng=np.random.default_rng(1))
    s_pred = generate_signal(seq, learned_kernel, noise=0.0, rng=np.random.default_rng(2))
    return np.mean((s_true - s_pred) ** 2)


def evaluate_noise():
    pad = K // 2

    acc_true_all = []
    acc_learned_all = []
    signal_mse_all = []
    kernel_error_all = []

    for noise in NOISE_LEVELS:
        print(f"Ruído={noise:.2f}")

        acc_true_reps = []
        acc_learned_reps = []
        signal_mse_reps = []
        kernel_error_reps = []

        for rep in range(REPEATS):
            rng = np.random.default_rng(SEED + rep + int(noise * 10000))

            learned = train_kernel(noise, rng)

            dna = generate_random_dna(N_TEST, rng)
            signal = generate_signal(dna, TRUE_KERNEL, noise=noise, rng=rng)

            pred_true = decode_signal_viterbi(signal, TRUE_KERNEL)
            pred_learned = decode_signal_viterbi(signal, learned)

            acc_true = accuracy(dna[pad:-pad], pred_true[pad:-pad])
            acc_learned = accuracy(dna[pad:-pad], pred_learned[pad:-pad])

            sig_mse = signal_prediction_mse(dna, TRUE_KERNEL, learned)
            kernel_error = np.linalg.norm(TRUE_KERNEL - learned)

            acc_true_reps.append(acc_true)
            acc_learned_reps.append(acc_learned)
            signal_mse_reps.append(sig_mse)
            kernel_error_reps.append(kernel_error)

        acc_true_all.append(acc_true_reps)
        acc_learned_all.append(acc_learned_reps)
        signal_mse_all.append(signal_mse_reps)
        kernel_error_all.append(kernel_error_reps)

        print(
            f"  true={np.mean(acc_true_reps):.2%}, "
            f"learned={np.mean(acc_learned_reps):.2%}, "
            f"signal_mse={np.mean(signal_mse_reps):.3e}"
        )

    return (
        np.array(acc_true_all),
        np.array(acc_learned_all),
        np.array(signal_mse_all),
        np.array(kernel_error_all)
    )


def panel_accuracy(ax, acc_true, acc_learned):
    add_panel_label(ax, "A")

    x = np.array(NOISE_LEVELS)

    true_mean = np.mean(acc_true, axis=1) * 100
    true_std = np.std(acc_true, axis=1) * 100

    learned_mean = np.mean(acc_learned, axis=1) * 100
    learned_std = np.std(acc_learned, axis=1) * 100

    ax.errorbar(
        x,
        true_mean,
        yerr=true_std,
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6,
        label="true kernel"
    )

    ax.errorbar(
        x,
        learned_mean,
        yerr=learned_std,
        marker="s",
        linewidth=4,
        markersize=10,
        capsize=6,
        label="learned kernel"
    )

    ax.set_title("Reconstruction accuracy", fontsize=22, fontweight="bold")
    ax.set_xlabel("noise level", fontsize=20, fontweight="bold")
    ax.set_ylabel("accuracy (%)", fontsize=20, fontweight="bold")

    ax.set_ylim(80, 101)
    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.legend(frameon=False, prop={"weight": "bold", "size": 15})

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_signal_mse(ax, signal_mse):
    add_panel_label(ax, "B")

    x = np.array(NOISE_LEVELS)
    mean = np.mean(signal_mse, axis=1)
    std = np.std(signal_mse, axis=1)

    ax.errorbar(
        x,
        mean,
        yerr=std,
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6
    )

    ax.set_yscale("log")

    ax.set_title("Signal prediction error", fontsize=22, fontweight="bold")
    ax.set_xlabel("noise level", fontsize=20, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=20, fontweight="bold")

    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_kernel_error(ax, kernel_error):
    add_panel_label(ax, "C")

    x = np.array(NOISE_LEVELS)
    mean = np.mean(kernel_error, axis=1)
    std = np.std(kernel_error, axis=1)

    ax.errorbar(
        x,
        mean,
        yerr=std,
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6
    )

    ax.set_title("Kernel deviation", fontsize=22, fontweight="bold")
    ax.set_xlabel("noise level", fontsize=20, fontweight="bold")
    ax.set_ylabel("norm error", fontsize=20, fontweight="bold")

    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_boxplot(ax, acc_learned):
    add_panel_label(ax, "D")

    data = [100 * row for row in acc_learned]

    bp = ax.boxplot(
        data,
        labels=[str(x) for x in NOISE_LEVELS],
        patch_artist=True,
        widths=0.6
    )

    for box in bp["boxes"]:
        box.set_linewidth(2)

    for median in bp["medians"]:
        median.set_linewidth(3)
        median.set_color("black")

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(2)

    for cap in bp["caps"]:
        cap.set_linewidth(2)

    ax.set_title("Learned-kernel accuracy spread", fontsize=22, fontweight="bold")
    ax.set_xlabel("noise level", fontsize=20, fontweight="bold")
    ax.set_ylabel("accuracy (%)", fontsize=20, fontweight="bold")

    ax.set_ylim(80, 101)

    ax.grid(True, axis="y", linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    acc_true, acc_learned, signal_mse, kernel_error = evaluate_noise()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    panel_accuracy(axes[0, 0], acc_true, acc_learned)
    panel_signal_mse(axes[0, 1], signal_mse)
    panel_kernel_error(axes[1, 0], kernel_error)
    panel_boxplot(axes[1, 1], acc_learned)

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_4_noise_robustness.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_4_noise_robustness.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    np.savez(
        "figure_4_noise_robustness_data.npz",
        noise_levels=np.array(NOISE_LEVELS),
        acc_true=acc_true,
        acc_learned=acc_learned,
        signal_mse=signal_mse,
        kernel_error=kernel_error
    )

    print("\nArquivos salvos:")
    print("figure_4_noise_robustness.svg")
    print("figure_4_noise_robustness.png")
    print("figure_4_noise_robustness_data.npz")


if __name__ == "__main__":
    main()
