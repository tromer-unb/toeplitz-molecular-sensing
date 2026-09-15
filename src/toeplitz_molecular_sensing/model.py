from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

ALPHABET = ("A", "T", "G", "C")
BASE_TO_INDEX = {base: i for i, base in enumerate(ALPHABET)}


def one_hot_encode(sequence: str) -> np.ndarray:
    """Encode a DNA sequence as an (N, 4) one-hot matrix."""
    x = np.zeros((len(sequence), 4), dtype=float)
    for i, base in enumerate(sequence):
        try:
            x[i, BASE_TO_INDEX[base]] = 1.0
        except KeyError as exc:
            raise ValueError(f"Unsupported nucleotide {base!r}; expected one of {ALPHABET}") from exc
    return x


def random_sequence(length: int, rng: np.random.Generator) -> str:
    """Sample an iid uniform DNA sequence over A,T,G,C."""
    if length < 1:
        raise ValueError("length must be positive")
    return "".join(rng.choice(ALPHABET, size=length).tolist())


def random_kernel(k: int = 5, channels: int = 3, seed: int = 7) -> np.ndarray:
    """Generate the manuscript's default U[0,1) finite-support sensing kernel."""
    _validate_kernel_shape_parameters(k, channels)
    return np.random.default_rng(seed).random((k, 4, channels))


def forward_signal(
    sequence: str,
    kernel: np.ndarray,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the translation-invariant finite-support sensing operator with zero padding."""
    kernel = np.asarray(kernel, dtype=float)
    k, channels = _kernel_dims(kernel)
    r = k // 2
    x = one_hot_encode(sequence)
    padded = np.pad(x, ((r, r), (0, 0)), mode="constant")
    signal = np.empty((len(sequence), channels), dtype=float)
    for i in range(len(sequence)):
        window = padded[i : i + k]
        signal[i] = np.einsum("kb,kbc->c", window, kernel)
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    if noise_sigma:
        rng = np.random.default_rng() if rng is None else rng
        signal = signal + rng.normal(0.0, noise_sigma, size=signal.shape)
    return signal


def build_design_matrix(sequence: str, k: int) -> np.ndarray:
    """Construct the local-window design matrix A in R^(N x 4K)."""
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer")
    r = k // 2
    x = one_hot_encode(sequence)
    padded = np.pad(x, ((r, r), (0, 0)), mode="constant")
    return np.vstack([padded[i : i + k].reshape(-1) for i in range(len(sequence))])


def calibrate_kernel(sequences: Sequence[str], signals: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Estimate the sensing kernel by unregularized least squares."""
    if len(sequences) != len(signals) or not sequences:
        raise ValueError("sequences and signals must be non-empty and have equal length")
    a_blocks, y_blocks = [], []
    channels = None
    for sequence, signal in zip(sequences, signals):
        y = np.asarray(signal, dtype=float)
        if y.ndim != 2 or y.shape[0] != len(sequence):
            raise ValueError("each signal must have shape (len(sequence), channels)")
        channels = y.shape[1] if channels is None else channels
        if y.shape[1] != channels:
            raise ValueError("all signals must have the same channel count")
        a_blocks.append(build_design_matrix(sequence, k))
        y_blocks.append(y)
    a = np.vstack(a_blocks)
    y = np.vstack(y_blocks)
    w, *_ = np.linalg.lstsq(a, y, rcond=None)
    return w.reshape(k, 4, channels)


def context_signature(context: str, kernel: np.ndarray) -> np.ndarray:
    """Return the local multichannel signature for a length-K context."""
    kernel = np.asarray(kernel, dtype=float)
    k, _ = _kernel_dims(kernel)
    if len(context) != k:
        raise ValueError(f"context length must equal kernel width ({k})")
    return np.einsum("kb,kbc->c", one_hot_encode(context), kernel)


def enumerate_signatures(kernel: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Enumerate all 4^K contexts and their multichannel signatures."""
    k, _ = _kernel_dims(np.asarray(kernel, dtype=float))
    contexts = ["".join(q) for q in product(ALPHABET, repeat=k)]
    signatures = np.vstack([context_signature(q, kernel) for q in contexts])
    return contexts, signatures


def decode_sequence(signal: np.ndarray, kernel: np.ndarray) -> str:
    """Decode by Viterbi-style dynamic programming over overlapping K-mers."""
    y = np.asarray(signal, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    k, channels = _kernel_dims(kernel)
    if y.ndim != 2 or y.shape[1] != channels:
        raise ValueError(f"signal must have shape (N, {channels})")
    n = y.shape[0]
    if n < 1:
        raise ValueError("signal must contain at least one sample")

    contexts, signatures = enumerate_signatures(kernel)
    suffix_to_states: dict[str, list[int]] = {}
    for idx, ctx in enumerate(contexts):
        suffix_to_states.setdefault(ctx[1:], []).append(idx)

    predecessors: list[np.ndarray] = []
    for ctx in contexts:
        predecessors.append(np.asarray(suffix_to_states[ctx[:-1]], dtype=np.int64))

    costs = np.sum((signatures[None, :, :] - y[:, None, :]) ** 2, axis=2)
    dp = costs[0].copy()
    back = np.empty((n, len(contexts)), dtype=np.int32)
    back[0].fill(-1)
    for i in range(1, n):
        nxt = np.empty_like(dp)
        for state, pred in enumerate(predecessors):
            local = dp[pred]
            j = int(np.argmin(local))
            nxt[state] = costs[i, state] + local[j]
            back[i, state] = int(pred[j])
        dp = nxt

    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    state_contexts = [contexts[idx] for idx in path]
    r = k // 2
    decoded = [state_contexts[0][j] for j in range(r)]
    decoded.extend(ctx[r] for ctx in state_contexts)
    decoded.extend(state_contexts[-1][r + 1 :])
    return "".join(decoded[r : r + n])


def interior_accuracy(reference: str, predicted: str, k: int) -> float:
    """Compute accuracy excluding r=(K-1)/2 positions at both ends."""
    if len(reference) != len(predicted):
        raise ValueError("reference and predicted sequences must have equal length")
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer")
    r = k // 2
    if len(reference) <= 2 * r:
        raise ValueError("sequence is too short for the requested interior evaluation")
    return float(np.mean([a == b for a, b in zip(reference[r:-r], predicted[r:-r])]))


def _validate_kernel_shape_parameters(k: int, channels: int) -> None:
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer")
    if channels < 1:
        raise ValueError("channels must be positive")


def _kernel_dims(kernel: np.ndarray) -> tuple[int, int]:
    if kernel.ndim != 3 or kernel.shape[1] != 4:
        raise ValueError("kernel must have shape (K, 4, C)")
    k, _, channels = kernel.shape
    _validate_kernel_shape_parameters(k, channels)
    return k, channels
