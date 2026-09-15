"""Re-run a publication-era figure script and compare its numerical output.

Usage:
    python scripts/verify_legacy_figure.py 3

Figures 2--7 have tracked NPZ outputs. Figure 1 is checked for successful
execution and expected plot files only.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "PAPER"

NPZ_BY_FIGURE = {
    2: "figure_2_identifiability_data.npz",
    3: "figure_3_kernel_learning_data.npz",
    4: "figure_4_noise_robustness_data.npz",
    5: "figure_5_modification_residual_data.npz",
    6: "figure_6_resolution_limit_data.npz",
    7: "figure_7_indel_limitation_data.npz",
}


def compare_npz(reference: Path, regenerated: Path) -> None:
    with np.load(reference, allow_pickle=False) as ref, np.load(regenerated, allow_pickle=False) as new:
        if set(ref.files) != set(new.files):
            raise AssertionError(f"NPZ keys differ: {ref.files} vs {new.files}")
        for key in ref.files:
            a = ref[key]
            b = new[key]
            if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
                np.testing.assert_array_equal(a, b, err_msg=f"Mismatch in {key}")
            else:
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=f"Mismatch in {key}")


def verify_figure(number: int) -> None:
    source_dir = LEGACY / f"figure{number}"
    script = source_dir / f"figure{number}.py"
    if not script.exists():
        raise FileNotFoundError(script)

    with tempfile.TemporaryDirectory(prefix=f"toeplitz-figure{number}-") as tmp:
        work = Path(tmp)
        shutil.copy2(script, work / script.name)
        subprocess.run([sys.executable, script.name], cwd=work, check=True)

        if number == 1:
            expected = [
                work / "figure_1_toeplitz_model.png",
                work / "figure_1_toeplitz_model.svg",
            ]
            missing = [str(path.name) for path in expected if not path.exists()]
            if missing:
                raise AssertionError(f"Missing generated outputs: {missing}")
            return

        filename = NPZ_BY_FIGURE[number]
        compare_npz(source_dir / filename, work / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", type=int, choices=range(1, 8))
    args = parser.parse_args()
    verify_figure(args.figure)
    print(f"Figure {args.figure}: legacy reproduction verified.")


if __name__ == "__main__":
    main()
