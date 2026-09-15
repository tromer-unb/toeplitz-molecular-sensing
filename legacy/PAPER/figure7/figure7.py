"""
Gera Figura 7 do paper:
Inserções/deleções simuladas no sinal como limitação do modelo.

Painéis:
A - Sinal normal, inserção e deleção
B - Resíduo para inserção
C - Resíduo para deleção
D - Localização do evento estimado vs evento real

Saídas:
figure_7_indel_limitation.svg
figure_7_indel_limitation.png
figure_7_indel_limitation_data.npz
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

N = 120
K = 5
C = 3
NOISE = 0.05

TRAIN_NUM = 200
TRAIN_LEN = 150
SAFE_MARGIN = 10

EVENT_POS = 60

REPEATS = 40
TOP_N_PEAKS = 10
MAX_EVENTS = 1
CLUSTER_DISTANCE = 4

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
    predicted_signal = generate_signal(
        decoded_seq,
        kernel,
        noise=0.0,
        rng=np.random.default_rng(1)
    )

    m = min(len(signal), len(predicted_signal))

    residual = np.sqrt(
        np.sum((signal[:m] - predicted_signal[:m]) ** 2, axis=1)
    )

    return residual, predicted_signal[:m]


def smooth_signal_1d(x, window=5):
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")

    y = np.zeros_like(x)

    for i in range(len(x)):
        y[i] = np.mean(xpad[i:i + window])

    return y


def insert_signal_event(signal, pos):
    extra = signal[pos:pos + 1].copy()
    return np.vstack([signal[:pos], extra, signal[pos:]])


def delete_signal_event(signal, pos):
    return np.vstack([signal[:pos], signal[pos + 1:]])


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


def detect_event(residual):
    candidates = raw_peak_candidates(residual)
    events = cluster_peak_candidates(candidates, residual)

    if len(events) == 0:
        return None, candidates, events

    return int(events[0][0]), candidates, events


def analyze_single_example(learned_kernel, rng):
    dna = generate_random_dna(N, rng)
    clean_signal = generate_signal(dna, TRUE_KERNEL, noise=NOISE, rng=rng)

    signal_insert = insert_signal_event(clean_signal, EVENT_POS)
    signal_delete = delete_signal_event(clean_signal, EVENT_POS)

    decoded_insert = decode_signal_viterbi(signal_insert, learned_kernel)
    decoded_delete = decode_signal_viterbi(signal_delete, learned_kernel)

    residual_insert_raw, predicted_insert = residual_map(
        signal_insert,
        decoded_insert,
        learned_kernel
    )

    residual_delete_raw, predicted_delete = residual_map(
        signal_delete,
        decoded_delete,
        learned_kernel
    )

    residual_insert = smooth_signal_1d(residual_insert_raw, window=5)
    residual_delete = smooth_signal_1d(residual_delete_raw, window=5)

    pred_insert, candidates_insert, events_insert = detect_event(residual_insert)
    pred_delete, candidates_delete, events_delete = detect_event(residual_delete)

    return {
        "dna": dna,
        "clean_signal": clean_signal,
        "signal_insert": signal_insert,
        "signal_delete": signal_delete,
        "decoded_insert": decoded_insert,
        "decoded_delete": decoded_delete,
        "residual_insert": residual_insert,
        "residual_delete": residual_delete,
        "predicted_insert_signal": predicted_insert,
        "predicted_delete_signal": predicted_delete,
        "pred_insert": pred_insert,
        "pred_delete": pred_delete,
        "events_insert": events_insert,
        "events_delete": events_delete,
    }


def evaluate_indels(learned_kernel):
    insert_errors = []
    delete_errors = []
    insert_positions = []
    delete_positions = []

    for rep in range(REPEATS):
        rng = np.random.default_rng(SEED + 5000 + rep)

        result = analyze_single_example(learned_kernel, rng)

        if result["pred_insert"] is not None:
            insert_positions.append(result["pred_insert"])
            insert_errors.append(result["pred_insert"] - EVENT_POS)

        if result["pred_delete"] is not None:
            delete_positions.append(result["pred_delete"])
            delete_errors.append(result["pred_delete"] - EVENT_POS)

    return {
        "insert_positions": np.array(insert_positions),
        "delete_positions": np.array(delete_positions),
        "insert_errors": np.array(insert_errors),
        "delete_errors": np.array(delete_errors),
    }


def panel_a(ax, example):
    add_panel_label(ax, "A")
    ax.set_title("Signal perturbations", fontsize=22, fontweight="bold")

    start = EVENT_POS - 15
    end = EVENT_POS + 16

    x_clean = np.arange(len(example["clean_signal"]))
    x_insert = np.arange(len(example["signal_insert"]))
    x_delete = np.arange(len(example["signal_delete"]))

    ax.plot(
        x_clean[start:end],
        example["clean_signal"][start:end, 0],
        linewidth=4,
        label="normal"
    )

    ax.plot(
        x_insert[start:end + 1],
        example["signal_insert"][start:end + 1, 0],
        linewidth=3,
        linestyle="--",
        label="insertion"
    )

    ax.plot(
        x_delete[start:end - 1],
        example["signal_delete"][start:end - 1, 0],
        linewidth=3,
        linestyle=":",
        label="deletion"
    )

    ax.axvline(EVENT_POS, linewidth=3, alpha=0.5)

    ax.set_xlabel("position", fontsize=20, fontweight="bold")
    ax.set_ylabel("channel 1", fontsize=20, fontweight="bold")

    ax.legend(frameon=False, prop={"weight": "bold", "size": 15})
    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_b(ax, example):
    add_panel_label(ax, "B")
    ax.set_title("Insertion residual", fontsize=22, fontweight="bold")

    residual = example["residual_insert"]
    x = np.arange(len(residual))

    ax.plot(x, residual, linewidth=4)
    ax.axvline(EVENT_POS, linewidth=3, linestyle="--")
    ax.axvline(example["pred_insert"], linewidth=3, linestyle=":")

    ax.scatter(
        [example["pred_insert"]],
        [residual[example["pred_insert"]]],
        s=180,
        marker="*",
        zorder=6
    )

    ax.set_xlabel("position", fontsize=20, fontweight="bold")
    ax.set_ylabel("residual", fontsize=20, fontweight="bold")

    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_c(ax, example):
    add_panel_label(ax, "C")
    ax.set_title("Deletion residual", fontsize=22, fontweight="bold")

    residual = example["residual_delete"]
    x = np.arange(len(residual))

    ax.plot(x, residual, linewidth=4)
    ax.axvline(EVENT_POS, linewidth=3, linestyle="--")
    ax.axvline(example["pred_delete"], linewidth=3, linestyle=":")

    ax.scatter(
        [example["pred_delete"]],
        [residual[example["pred_delete"]]],
        s=180,
        marker="*",
        zorder=6
    )

    ax.set_xlabel("position", fontsize=20, fontweight="bold")
    ax.set_ylabel("residual", fontsize=20, fontweight="bold")

    ax.grid(True, linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def panel_d(ax, indel_stats):
    add_panel_label(ax, "D")
    ax.set_title("Localization error", fontsize=22, fontweight="bold")

    data = [
        indel_stats["insert_errors"],
        indel_stats["delete_errors"]
    ]

    bp = ax.boxplot(
        data,
        labels=["insertion", "deletion"],
        patch_artist=True,
        widths=0.55
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

    ax.axhline(0, linewidth=3, linestyle="--")

    ax.set_ylabel("predicted - real", fontsize=20, fontweight="bold")

    ax.grid(True, axis="y", linewidth=1.5, alpha=0.35)

    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    rng_train = np.random.default_rng(SEED)
    learned_kernel = train_kernel(rng_train)

    rng_example = np.random.default_rng(SEED + 999)
    example = analyze_single_example(learned_kernel, rng_example)

    indel_stats = evaluate_indels(learned_kernel)

    print("Event position:", EVENT_POS)
    print("Example predicted insertion:", example["pred_insert"])
    print("Example predicted deletion:", example["pred_delete"])
    print("Insertion errors:", indel_stats["insert_errors"])
    print("Deletion errors:", indel_stats["delete_errors"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    panel_a(axes[0, 0], example)
    panel_b(axes[0, 1], example)
    panel_c(axes[1, 0], example)
    panel_d(axes[1, 1], indel_stats)

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "figure_7_indel_limitation.svg",
        format="svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "figure_7_indel_limitation.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    np.savez(
        "figure_7_indel_limitation_data.npz",
        event_pos=EVENT_POS,
        clean_signal=example["clean_signal"],
        signal_insert=example["signal_insert"],
        signal_delete=example["signal_delete"],
        residual_insert=example["residual_insert"],
        residual_delete=example["residual_delete"],
        pred_insert=example["pred_insert"],
        pred_delete=example["pred_delete"],
        insert_errors=indel_stats["insert_errors"],
        delete_errors=indel_stats["delete_errors"]
    )

    print("\nArquivos salvos:")
    print("figure_7_indel_limitation.svg")
    print("figure_7_indel_limitation.png")
    print("figure_7_indel_limitation_data.npz")


if __name__ == "__main__":
    main()
