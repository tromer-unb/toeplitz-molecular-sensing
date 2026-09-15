"""Interpretable Toeplitz models for sequence-dependent molecular sensing."""

from .model import (
    ALPHABET,
    build_design_matrix,
    calibrate_kernel,
    context_signature,
    decode_sequence,
    enumerate_signatures,
    forward_signal,
    interior_accuracy,
    one_hot_encode,
    random_kernel,
    random_sequence,
)
from .residuals import DetectedEvent, residual_magnitude, select_events, smooth_residual

__all__ = [
    "ALPHABET", "DetectedEvent", "build_design_matrix", "calibrate_kernel",
    "context_signature", "decode_sequence", "enumerate_signatures", "forward_signal",
    "interior_accuracy", "one_hot_encode", "random_kernel", "random_sequence",
    "residual_magnitude", "select_events", "smooth_residual",
]
