import numpy as np

from toeplitz_molecular_sensing import (
    build_design_matrix,
    calibrate_kernel,
    decode_sequence,
    forward_signal,
    interior_accuracy,
    one_hot_encode,
    random_kernel,
    random_sequence,
    residual_magnitude,
    select_events,
    smooth_residual,
)


def test_one_hot_order_matches_manuscript_alphabet():
    x = one_hot_encode("ATGC")
    np.testing.assert_array_equal(x, np.eye(4))


def test_forward_equals_design_matrix_product():
    kernel = random_kernel(k=5, channels=3, seed=7)
    sequence = "ATGCCGTA"
    direct = forward_signal(sequence, kernel)
    matrix = build_design_matrix(sequence, 5) @ kernel.reshape(20, 3)
    np.testing.assert_allclose(direct, matrix, atol=1e-12)


def test_noiseless_kernel_recovery():
    rng = np.random.default_rng(12)
    kernel = random_kernel(k=3, channels=2, seed=7)
    sequences = [random_sequence(80, rng) for _ in range(8)]
    signals = [forward_signal(seq, kernel) for seq in sequences]
    estimated = calibrate_kernel(sequences, signals, k=3)
    np.testing.assert_allclose(estimated, kernel, atol=1e-10)


def test_noiseless_decoding_is_exact_interior():
    rng = np.random.default_rng(21)
    kernel = random_kernel(k=3, channels=3, seed=7)
    sequence = random_sequence(30, rng)
    signal = forward_signal(sequence, kernel)
    decoded = decode_sequence(signal, kernel)
    assert interior_accuracy(sequence, decoded, k=3) == 1.0


def test_residual_smoothing_and_event_ranking():
    observed = np.zeros((9, 2))
    predicted = observed.copy()
    observed[4] = [3.0, 4.0]
    residual = residual_magnitude(observed, predicted)
    smooth = smooth_residual(residual, window=3)
    events = select_events(smooth, n_candidates=3, radius=1, max_events=1)
    assert events[0].position in {3, 4, 5}
    assert events[0].score > 0
