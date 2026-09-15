"""
Gera Figura 6 do paper:
Limite de resolução espacial para modificações próximas.

Painéis:
A - Exemplo de duas modificações próximas
B - Número médio de eventos detectados vs distância
C - Taxa de separação correta vs distância
D - F1 de detecção vs distância

Saídas:
figure_6_resolution_limit.svg
figure_6_resolution_limit.png
figure_6_resolution_limit_data.npz
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.patches import Rectangle

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

N = 120
K = 5
C = 3
NOISE = 0.05

SAFE_MARGIN = 10
TRAIN_NUM = 200
TRAIN_LEN = 150

DISTANCES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
REPEATS = 40

TOP_N_PEAKS = 12
MAX_EVENTS = 2
CLUSTER_DISTANCE = 2

TRUE_KERNEL = np.random.default_rng(SEED).random((K, 4, C))
METHYLATION_SHIFT = np.array([0.35, -0.25, 0.20])


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


def force_c_pair(seq, pos1, pos2):
    seq = list(seq)
    seq[pos1] = "C"
    seq[pos2] = "C"
    return "".join(seq)


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


def generate_methylated_signal(seq, kernel, methylated_positions, noise=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    signal = generate_signal(seq, kernel, noise=0.0, rng=rng)

    for pos in methylated_positions:
        for offset in range(-(K // 2), K // 2 + 1):
            t = pos + offset

            if 0 <= t < len(seq):
                weight = 1.0 / (1.0 + abs(offset))
                signal[t] += weight * METHYLATION_SHIFT

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


def train_kernel(rng):
    seqs = []
    signals = []

    for _ in range(TRAIN_NUM):
        seq = generate_random_dna(TRAIN_LEN, rng)
        sig = generate_signal(seq, TRUE_KERNEL, noise=NOISE, rng=rng)

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
        return np.sum((kmer_values[km] - s) ** 2)

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


def residual_map(signal, decoded_seq, kernel):
    predicted_signal = generate_signal(decoded_seq, kernel, noise=0.0, rng=np.random.default_rng(1))
    residual = np.sqrt(np.sum((signal - predicted_signal) ** 2, axis=1))
    return residual, predicted_signal


def smooth_signal_1d(x, window=3):
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.zeros_like(x)

    for i in range(len(x)):
        y[i] = np.mean(xpad[i:i + window])

    return y


def raw_peak_candidates(residual, top_n=TOP_N_PEAKS, margin=SAFE_MARGIN):
    candidates = []

    for i, r in enumerate(residual):
        if i < margin or i >= len(residual) - margin:
            continue
        candidates.append((i, float(r)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def cluster_peak_candidates(candidates, residual, max_events=MAX_EVENTS, cluster_distance=CLUSTER_DISTANCE):
    clusters = []

    for pos, score in candidates:
        placed = False

        for cluster in clusters:
            if abs(pos - cluster["center"]) <= cluster_distance:
                cluster["positions"].append(pos)

                best_pos = max(cluster["positions"], key=lambda p: residual[p])
                cluster["center"] = int(best_pos)
                cluster["score"] = float(residual[best_pos])

                placed = True
                break

        if not placed:
            clusters.append({
                "positions": [pos],
                "center": int(pos),
                "score": float(score),
            })

    clusters.sort(key=lambda c: c["score"], reverse=True)
    clusters = clusters[:max_events]

    return [
        (c["center"], c["score"], sorted(c["positions"]))
        for c in clusters
    ]


def detection_metrics(true_positions, predicted_positions, tolerance=1):
    true_positions = [int(x) for x in true_positions]
    predicted_positions = [int(x) for x in predicted_positions]

    matched_true = set()
    matched_pred = set()

    for pi, p in enumerate(predicted_positions):
        best_ti = None
        best_dist = None

        for ti, t in enumerate(true_positions):
            if ti in matched_true:
                continue

            dist = abs(p - t)

            if dist <= tolerance:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_ti = ti

        if best_ti is not None:
            matched_true.add(best_ti)
            matched_pred.add(pi)

    tp = len(matched_true)
    fp = len(predicted_positions) - len(matched_pred)
    fn = len(true_positions) - len(matched_true)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1


def make_pair_example(distance, learned_kernel, rng):
    center = N // 2
    pos1 = center
    pos2 = center + distance

    dna = generate_random_dna(N, rng)
    dna = force_c_pair(dna, pos1, pos2)

    methylated_positions = [pos1, pos2]

    signal = generate_methylated_signal(
        dna,
        TRUE_KERNEL,
        methylated_positions,
        noise=NOISE,
        rng=rng
    )

    decoded = decode_signal_viterbi(signal, learned_kernel)
    residual_raw, predicted_signal = residual_map(signal, decoded, learned_kernel)
    residual_smooth = smooth_signal_1d(residual_raw, window=3)

    candidates = raw_peak_candidates(residual_smooth)
    events = cluster_peak_candidates(candidates, residual_smooth)

    return {
        "dna": dna,
        "signal": signal,
        "predicted_signal": predicted_signal,
        "residual": residual_smooth,
        "methylated_positions": methylated_positions,
        "events": events,
        "predicted_positions": [int(e[0]) for e in events],
    }


def evaluate_resolution():
    rng_train = np.random.default_rng(SEED)
    learned_kernel = train_kernel(rng_train)

    mean_events = []
    std_events = []
    separation_rate = []
    f1_mean = []
    f1_std = []

    example_data = None

    for d in DISTANCES:
        print(f"Distância={d}")

        n_events = []
        separated = []
        f1s = []

        for rep in range(REPEATS):
            rng = np.random.default_rng(SEED + 1000 * d + rep)

            result = make_pair_example(
                distance=d,
                learned_kernel=learned_kernel,
                rng=rng
            )

            pred = result["predicted_positions"]
            true = result["methylated_positions"]

            tp, fp, fn, precision, recall, f1 = detection_metrics(
                true,
                pred,
                tolerance=1
            )

            n_events.append(len(pred))
            f1s.append(f1)

            separated.append(1.0 if tp == 2 and len(pred) == 2 else 0.0)

            if d == 2 and rep == 0:
                example_data = result

        mean_events.append(np.mean(n_events))
        std_events.append(np.std(n_events))
        separation_rate.append(np.mean(separated))
        f1_mean.append(np.mean(f1s))
        f1_std.append(np.std(f1s))

        print(
            f"  events={np.mean(n_events):.2f}, "
            f"sep={np.mean(separated):.2%}, "
            f"F1={np.mean(f1s):.2%}"
        )

    return {
        "learned_kernel": learned_kernel,
        "mean_events": np.array(mean_events),
        "std_events": np.array(std_events),
        "separation_rate": np.array(separation_rate),
        "f1_mean": np.array(f1_mean),
        "f1_std": np.array(f1_std),
        "example": example_data,
    }


def panel_a(ax, example):
    add_panel_label(ax, "A")
    ax.set_title("Close modifications", fontsize=22, fontweight="bold")

    dna = example["dna"]
    methylated = set(example["methylated_positions"])

    start = N // 2 - 8
    end = N // 2 + 14

    for idx, i in enumerate(range(start, end)):
        b = dna[i]

        face = "#f0f0f0"
        edge = "black"
        lw = 1.5

        if i in methylated:
            face = "#ffb3b3"
            edge = "red"
            lw = 3

        ax.add_patch(
            Rectangle(
                (idx, 0),
                0.85,
                0.8,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw
            )
        )

        ax.text(
            idx + 0.425,
            0.4,
            b,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold"
        )

        if i in methylated:
            ax.text(
                idx + 0.425,
                1.05,
                "M",
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                color="red"
            )

    ax.set_xlim(0, end - start)
    ax.set_ylim(-0.2, 1.55)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(
        [str(start), str(start + 5), str(start + 10), str(start + 15), str(start + 20)],
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("position", fontsize=20, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(axis="x", width=2, length=6)


def panel_b(ax, results):
    add_panel_label(ax, "B")
    ax.set_title("Detected event count", fontsize=22, fontweight="bold")

    ax.errorbar(
        DISTANCES,
        results["mean_events"],
        yerr=results["std_events"],
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6
    )

    ax.axhline(2, linestyle="--", linewidth=3)

    ax.set_xlabel("distance", fontsize=20, fontweight="bold")
    ax.set_ylabel("events", fontsize=20, fontweight="bold")

    ax.set_ylim(0.8, 2.4)
    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_c(ax, results):
    add_panel_label(ax, "C")
    ax.set_title("Separation rate", fontsize=22, fontweight="bold")

    ax.plot(
        DISTANCES,
        100 * results["separation_rate"],
        marker="o",
        linewidth=4,
        markersize=10
    )

    ax.axvline(K, linestyle="--", linewidth=3)

    ax.set_xlabel("distance", fontsize=20, fontweight="bold")
    ax.set_ylabel("separated (%)", fontsize=20, fontweight="bold")

    ax.set_ylim(-5, 105)
    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_d(ax, results):
    add_panel_label(ax, "D")
    ax.set_title("Detection F1", fontsize=22, fontweight="bold")

    ax.errorbar(
        DISTANCES,
        100 * results["f1_mean"],
        yerr=100 * results["f1_std"],
        marker="o",
        linewidth=4,
        markersize=10,
        capsize=6
    )

    ax.axvline(K, linestyle="--", linewidth=3)

    ax.set_xlabel("distance", fontsize=20, fontweight="bold")
    ax.set_ylabel("F1 (%)", fontsize=20, fontweight="bold")

    ax.set_ylim(-5, 105)
    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    results = evaluate_resolution()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    panel_a(axes[0, 0], results["example"])
    panel_b(axes[0, 1], results)
    panel_c(axes[1, 0], results)
    panel_d(axes[1, 1], results)

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_6_resolution_limit.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_6_resolution_limit.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    np.savez(
        "figure_6_resolution_limit_data.npz",
        distances=np.array(DISTANCES),
        mean_events=results["mean_events"],
        std_events=results["std_events"],
        separation_rate=results["separation_rate"],
        f1_mean=results["f1_mean"],
        f1_std=results["f1_std"],
        example_dna=np.array(list(results["example"]["dna"])),
        example_residual=results["example"]["residual"],
        example_methylated_positions=np.array(results["example"]["methylated_positions"]),
        example_predicted_positions=np.array(results["example"]["predicted_positions"])
    )

    print("\nArquivos salvos:")
    print("figure_6_resolution_limit.svg")
    print("figure_6_resolution_limit.png")
    print("figure_6_resolution_limit_data.npz")


if __name__ == "__main__":
    main()
