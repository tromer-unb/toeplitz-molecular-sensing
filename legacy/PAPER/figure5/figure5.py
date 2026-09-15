"""
Gera Figura 5 do paper:
Detecção de modificação por resíduo do modelo Toeplitz.

Painéis:
A - DNA com posições modificadas
B - Sinal observado vs sinal previsto
C - Resíduo por posição
D - Clustering de picos e eventos finais

Saídas:
figure_5_modification_residual.svg
figure_5_modification_residual.png
figure_5_modification_residual_data.npz
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

N = 80
K = 5
C = 3
NOISE = 0.05

SAFE_MARGIN = K
NUM_METHYLATIONS = 3
TOP_N_PEAKS = 8
MAX_EVENTS = 3
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


def train_kernel(num_train=200, train_len=150, noise=NOISE, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)

    seqs = []
    signals = []

    for _ in range(num_train):
        seq = generate_random_dna(train_len, rng)
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


def residual_map(signal, decoded_seq, kernel):
    predicted_signal = generate_signal(decoded_seq, kernel, noise=0.0, rng=np.random.default_rng(1))
    residual = np.sqrt(np.sum((signal - predicted_signal) ** 2, axis=1))
    return residual, predicted_signal


def smooth_signal_1d(x, window=3):
    if window <= 1:
        return x.copy()

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


def detection_metrics(true_positions, predicted_positions, tolerance=0):
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


def prepare_example():
    rng = np.random.default_rng(SEED)

    learned_kernel = train_kernel(
        num_train=200,
        train_len=150,
        noise=NOISE,
        rng=rng
    )

    dna = generate_random_dna(N, rng)

    candidate_c_positions = [
        i for i, b in enumerate(dna)
        if b == "C" and SAFE_MARGIN <= i < N - SAFE_MARGIN
    ]

    if len(candidate_c_positions) < NUM_METHYLATIONS:
        raise RuntimeError("Poucas bases C. Tente outro SEED.")

    methylated_positions = sorted(
        int(x)
        for x in rng.choice(
            candidate_c_positions,
            size=NUM_METHYLATIONS,
            replace=False
        )
    )

    signal = generate_methylated_signal(
        dna,
        TRUE_KERNEL,
        methylated_positions,
        noise=NOISE,
        rng=rng
    )

    decoded = decode_signal_viterbi(signal, learned_kernel)

    residual_raw, predicted_signal = residual_map(
        signal,
        decoded,
        learned_kernel
    )

    residual_smooth = smooth_signal_1d(residual_raw, window=3)

    raw_candidates = raw_peak_candidates(
        residual_smooth,
        top_n=TOP_N_PEAKS,
        margin=SAFE_MARGIN
    )

    events = cluster_peak_candidates(
        raw_candidates,
        residual_smooth,
        max_events=MAX_EVENTS,
        cluster_distance=CLUSTER_DISTANCE
    )

    predicted_positions = [int(p) for p, _, _ in events]

    tp, fp, fn, precision, recall, f1 = detection_metrics(
        methylated_positions,
        predicted_positions,
        tolerance=0
    )

    return {
        "dna": dna,
        "signal": signal,
        "predicted_signal": predicted_signal,
        "decoded": decoded,
        "methylated_positions": methylated_positions,
        "residual_raw": residual_raw,
        "residual_smooth": residual_smooth,
        "raw_candidates": raw_candidates,
        "events": events,
        "predicted_positions": predicted_positions,
        "metrics": (tp, fp, fn, precision, recall, f1),
        "learned_kernel": learned_kernel,
    }


def panel_a(ax, data):
    add_panel_label(ax, "A")
    ax.set_title("Modified positions", fontsize=22, fontweight="bold")

    dna = data["dna"]
    methylated = set(data["methylated_positions"])

    y = 0

    for i, b in enumerate(dna):
        face = "#f0f0f0"
        edge = "black"
        lw = 1.5

        if i in methylated:
            face = "#ffb3b3"
            edge = "red"
            lw = 3

        ax.add_patch(
            Rectangle(
                (i, y),
                0.9,
                0.8,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw
            )
        )

        if i % 5 == 0 or i in methylated:
            ax.text(
                i + 0.45,
                y + 0.4,
                b,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    for pos in methylated:
        ax.text(
            pos + 0.45,
            1.05,
            "M",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="red"
        )

    ax.set_xlim(0, len(dna))
    ax.set_ylim(-0.2, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("position", fontsize=20, fontweight="bold")

    ax.tick_params(axis="x", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_b(ax, data):
    add_panel_label(ax, "B")
    ax.set_title("Observed and predicted signal", fontsize=22, fontweight="bold")

    x = np.arange(len(data["dna"]))

    observed = data["signal"][:, 0]
    predicted = data["predicted_signal"][:, 0]

    ax.plot(x, observed, linewidth=3, label="observed")
    ax.plot(x, predicted, linewidth=3, linestyle="--", label="predicted")

    for pos in data["methylated_positions"]:
        ax.axvline(pos, linewidth=2.5, alpha=0.45)

    ax.set_xlabel("position", fontsize=20, fontweight="bold")
    ax.set_ylabel("channel 1", fontsize=20, fontweight="bold")

    ax.legend(frameon=False, prop={"weight": "bold", "size": 15})
    ax.grid(True, linewidth=1.5, alpha=0.3)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_c(ax, data):
    add_panel_label(ax, "C")
    ax.set_title("Residual map", fontsize=22, fontweight="bold")

    x = np.arange(len(data["dna"]))
    residual = data["residual_smooth"]

    ax.plot(x, residual, linewidth=4)
    ax.scatter(
        data["methylated_positions"],
        residual[data["methylated_positions"]],
        s=120,
        zorder=5,
        label="modified"
    )

    for pos in data["methylated_positions"]:
        ax.axvline(pos, linewidth=2.5, alpha=0.45)

    ax.set_xlabel("position", fontsize=20, fontweight="bold")
    ax.set_ylabel("residual", fontsize=20, fontweight="bold")

    ax.legend(frameon=False, prop={"weight": "bold", "size": 15})
    ax.grid(True, linewidth=1.5, alpha=0.3)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_d(ax, data):
    add_panel_label(ax, "D")
    ax.set_title("Peak clustering", fontsize=22, fontweight="bold")

    x = np.arange(len(data["dna"]))
    residual = data["residual_smooth"]

    ax.plot(x, residual, linewidth=3)

    for pos, score in data["raw_candidates"]:
        ax.scatter(pos, score, s=80, marker="o")

    for pos, score, cluster in data["events"]:
        left = min(cluster) - 0.5
        width = max(cluster) - min(cluster) + 1.0

        ax.add_patch(
            Rectangle(
                (left, 0),
                width,
                np.max(residual) * 1.05,
                fill=False,
                edgecolor="black",
                linewidth=3
            )
        )

        ax.scatter(pos, score, s=180, marker="*", zorder=6)
        ax.text(
            pos,
            score + 0.035,
            f"{pos}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold"
        )

    tp, fp, fn, precision, recall, f1 = data["metrics"]

    ax.text(
        0.03,
        0.92,
        f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", linewidth=2)
    )

    ax.set_xlabel("position", fontsize=20, fontweight="bold")
    ax.set_ylabel("residual", fontsize=20, fontweight="bold")

    ax.grid(True, linewidth=1.5, alpha=0.3)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    data = prepare_example()

    print("DNA:")
    print(data["dna"])
    print("Modified positions:", data["methylated_positions"])
    print("Predicted events:", data["predicted_positions"])
    print("Metrics:", data["metrics"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    panel_a(axes[0, 0], data)
    panel_b(axes[0, 1], data)
    panel_c(axes[1, 0], data)
    panel_d(axes[1, 1], data)

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_5_modification_residual.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_5_modification_residual.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    np.savez(
        "figure_5_modification_residual_data.npz",
        dna=np.array(list(data["dna"])),
        signal=data["signal"],
        predicted_signal=data["predicted_signal"],
        residual_raw=data["residual_raw"],
        residual_smooth=data["residual_smooth"],
        methylated_positions=np.array(data["methylated_positions"]),
        predicted_positions=np.array(data["predicted_positions"])
    )

    print("\nArquivos salvos:")
    print("figure_5_modification_residual.svg")
    print("figure_5_modification_residual.png")
    print("figure_5_modification_residual_data.npz")


if __name__ == "__main__":
    main()
