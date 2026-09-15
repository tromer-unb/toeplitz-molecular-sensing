# Toeplitz Molecular Sensing

Reproducible code for the accepted PCCP manuscript **“A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing.”**

This repository is organized for scientific auditability: reusable mathematics is implemented as a tested Python package, figure workflows are separated from model code, random seeds remain explicit, and the original submission-era archive is retained for provenance.

## Scientific scope

The framework represents sequence-dependent molecular sensing as a finite-support, translation-invariant multichannel operator. It supports synthetic DNA generation and one-hot encoding, forward signal generation, least-squares kernel calibration, constrained K-mer dynamic-programming reconstruction, residual-field construction, anomaly localization, and controlled identifiability/noise/alignment studies.

The experiments are synthetic proof-of-concept calculations. They are not direct performance measurements of a specific nanopore instrument.

## Repository layout

```text
.
├── src/toeplitz_molecular_sensing/   # reusable scientific implementation
│   ├── model.py                      # encoding, forward model, calibration, decoder
│   └── residuals.py                  # residual smoothing and event selection
├── tests/                            # deterministic unit tests
├── scripts/                          # executable workflows and smoke checks
├── docs/REPRODUCIBILITY.md           # provenance and audit policy
├── PAPER.tar.gz                      # original publication-era source bundle
├── pyproject.toml                    # package metadata and dependencies
├── requirements.txt                  # minimal runtime dependencies
└── .github/workflows/                # automated verification
```

`PAPER.tar.gz` is preserved as a provenance artifact. New development should not depend on extracting the archive manually.

## Quick start

```bash
git clone https://github.com/tromer-unb/toeplitz-molecular-sensing.git
cd toeplitz-molecular-sensing
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
pytest
python scripts/smoke_reproduction.py
```

The smoke workflow generates a ground-truth kernel with seed `7`, calibrates it from noisy synthetic sequences, decodes an independent sequence, and reports kernel error and interior sequence accuracy.

## Core equations implemented in `src/`

For a one-hot sequence vector, finite-support kernel blocks, and multichannel signal, the package implements the manuscript's zero-padded translation-invariant forward operator. Calibration uses `numpy.linalg.lstsq`. Reconstruction minimizes cumulative squared emission cost over overlapping K-mer states while enforcing a K−1 nucleotide overlap between successive contexts.

## Verification

The test suite checks four refactoring invariants:

1. one-hot ordering is exactly `A, T, G, C`;
2. direct forward evaluation equals the matrix design representation;
3. noiseless least-squares calibration recovers the ground-truth kernel numerically;
4. noiseless dynamic-programming reconstruction is exact over manuscript-defined interior positions.

Run:

```bash
pytest
ruff check src tests
```

GitHub Actions runs the same verification on Python 3.10, 3.11, and 3.12.

## Reproducing paper figures

The accepted manuscript reports Figures 1–7 covering the conceptual workflow, local-signature identifiability, kernel calibration, noise robustness, residual localization, two-event separation, and signal-sample insertion/deletion. The historical figure scripts are retained in `PAPER.tar.gz` until they are individually migrated into auditable `scripts/figure_XX_*.py` entry points.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the migration and provenance policy.

## Branching convention

Use focused branches such as `model/decoder`, `figure/2-identifiability`, `figure/5-residual-localization`, and `docs/reproducibility`. Changes that affect reported numerical values should state the figure/section affected, seed rule, execution command, and before/after values in the pull request.

## Citation

Please cite the associated article:

> R. M. Tromer *et al.*, **A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing**, accepted for publication in *Physical Chemistry Chemical Physics* (2026).

Update the citation metadata with the final DOI, volume, issue, and pages when the version of record is available.

## License

MIT. See [`LICENSE`](LICENSE).
