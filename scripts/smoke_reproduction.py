"""Fast end-to-end check of calibration and decoding without reproducing full figures."""

import numpy as np

from toeplitz_molecular_sensing import (
    calibrate_kernel,
    decode_sequence,
    forward_signal,
    interior_accuracy,
    random_kernel,
    random_sequence,
)


def main() -> None:
    kernel = random_kernel(k=5, channels=3, seed=7)
    rng = np.random.default_rng(20260915)
    train = [random_sequence(100, rng) for _ in range(20)]
    signals = [forward_signal(seq, kernel, noise_sigma=0.02, rng=rng) for seq in train]
    learned = calibrate_kernel(train, signals, k=5)

    test = random_sequence(80, rng)
    observed = forward_signal(test, kernel, noise_sigma=0.02, rng=rng)
    decoded = decode_sequence(observed, learned)
    accuracy = interior_accuracy(test, decoded, k=5)
    kernel_error = np.linalg.norm(learned - kernel)
    print(f"kernel_frobenius_error={kernel_error:.6g}")
    print(f"interior_accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
