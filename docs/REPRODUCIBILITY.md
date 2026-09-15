# Reproducibility and audit trail

This repository is organized so that the scientific model can be inspected independently of figure-specific plotting code.

## Scientific provenance

The accepted manuscript **A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing** defines the computational model used here: finite-support translation-invariant sensing, zero boundary padding, iid Gaussian noise, least-squares kernel calibration, overlapping K-mer dynamic programming, and residual-based anomaly localization.

The original submission-era source bundle is retained as `PAPER.tar.gz`. It is treated as a provenance artifact rather than as the primary execution interface. The refactored package under `src/toeplitz_molecular_sensing/` implements the manuscript equations in directly testable functions.

## Deterministic parameters

The default ground-truth kernel uses NumPy's default random generator with seed `7`. Figure-specific seed formulae and experimental grids should remain visible in figure reproduction scripts rather than hidden inside the reusable package.

## Figure migration map

| Figure | Scientific task | Manuscript parameters that must remain explicit |
|---|---|---|
| 1 | Conceptual workflow | K=5, C=3 |
| 2 | Local-signature identifiability | K in {3,5,7,9}; C in {1,2,3,4}; seed 7+100K+10C; 8-decimal collision rule |
| 3 | Kernel calibration | M in {5,10,25,50,100,200,500}; length 150; sigma=0.05; 15 realizations |
| 4 | Noise robustness | sigma in {0,0.02,0.05,0.10,0.20,0.30}; 20 realizations |
| 5 | Residual localization | 80 bases; Delta=(0.35,-0.25,0.20); w=3; top 8; rho=2; max 3 |
| 6 | Two-event separation | d in {1,2,3,4,5,6,7,8,10,12}; 40 realizations; w=3; top 12; rho=2; delta=1 |
| 7 | Signal-sample insertion/deletion | index 60; 40 realizations; w=5; top 10; rho=4; boundary exclusion |

## Audit policy

1. Scientific functions live in `src/` and are covered by unit tests.
2. Figure scripts should only configure parameters, call library functions, and create outputs.
3. Generated figures/data belong in `results/` and should not be hand-edited.
4. Changes affecting reported values must include a test and an updated reproduction note.
5. `PAPER.tar.gz` is never silently replaced; if regenerated, its SHA-256 must be recorded in release notes.

## Minimal verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check src tests
python scripts/smoke_reproduction.py
```

A successful test suite verifies the core operator identity, noiseless least-squares recovery, noiseless interior decoding, and residual event selection.

## Legacy-source migration

The historical archive should be extracted only on a dedicated migration branch. Each original figure script should then be compared against the refactored library function-by-function. Differences in random-number draw order, boundary handling, candidate ranking, or plotting-only transformations must be documented before any historical script is retired.
