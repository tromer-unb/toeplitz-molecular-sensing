# A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing

This repository contains the Python source codes used to reproduce the computational results and figures presented in the paper:

> **A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing**

The complete source code is provided as a compressed archive:

```text
PAPER.tar.gz
```

After downloading the repository, users must extract this archive and execute the Python script associated with each figure.

---

## Overview

The paper presents an interpretable multichannel Toeplitz framework for sequence-dependent molecular sensing.

The computational workflow includes:

1. generation of synthetic DNA sequences;
2. one-hot encoding of nucleotide sequences;
3. generation of multichannel signals using a finite-support Toeplitz kernel;
4. calibration of the sensing kernel using least squares;
5. sequence reconstruction using constrained dynamic programming;
6. signal reconstruction from the predicted sequence;
7. residual calculation;
8. localization of synthetic signal anomalies;
9. evaluation of insertion and deletion of signal samples.

The scripts in this repository reproduce the figures and numerical experiments described in the manuscript.

---

## Repository structure

The GitHub repository contains:

```text
.
├── PAPER.tar.gz
├── README.md
└── LICENSE
```

The file `PAPER.tar.gz` contains all source-code directories.

After extraction, the directory structure should be similar to:

```text
PAPER/
├── figure1/
│   └── figure1.py
│
├── figure2/
│   └── figure2.py
│
├── figure3/
│   └── figure3.py
│
├── figure4/
│   └── figure4.py
│
├── figure5/
│   └── figure5.py
│
├── figure6/
│   └── figure6.py
│
└── figure7/
    └── figure7.py
```

Depending on the final version of the archive, each figure directory may also contain:

```text
input data
generated data
text output
PNG figures
PDF figures
NumPy arrays
auxiliary Python files
```

Each directory is intended to be self-contained and contains the code required to reproduce the corresponding figure.

---

# Quick start

The basic reproduction procedure is:

```bash
git clone https://github.com/USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
tar -xzf PAPER.tar.gz
cd PAPER
pip install numpy matplotlib
python figure1/figure1.py
```

Replace:

```text
USERNAME
```

with the GitHub username or organization name, and replace:

```text
REPOSITORY-NAME
```

with the repository name.

---

# 1. Downloading the repository

## Option A: Clone with Git

Open a terminal and run:

```bash
git clone https://github.com/USERNAME/REPOSITORY-NAME.git
```

Enter the downloaded repository:

```bash
cd REPOSITORY-NAME
```

Check that the archive is present:

```bash
ls
```

The output should contain:

```text
PAPER.tar.gz
README.md
LICENSE
```

## Option B: Download from the GitHub website

1. Open the repository page.
2. Click the green **Code** button.
3. Select **Download ZIP**.
4. Extract the downloaded ZIP file.
5. Open the extracted repository directory.

The file `PAPER.tar.gz` will be located inside that directory.

---

# 2. Extracting the source codes

## Linux

Run:

```bash
tar -xzf PAPER.tar.gz
```

## macOS

Run:

```bash
tar -xzf PAPER.tar.gz
```

## Windows PowerShell

Recent versions of Windows usually support the `tar` command:

```powershell
tar -xzf PAPER.tar.gz
```

## Windows with 7-Zip

The archive can also be extracted using 7-Zip:

1. Right-click `PAPER.tar.gz`.
2. Select **7-Zip**.
3. Select **Extract Here**.
4. A `.tar` file may be created.
5. Extract the `.tar` file again.

After extraction, a directory named `PAPER` should be available.

Enter the extracted directory:

```bash
cd PAPER
```

---

# 3. Finding the source codes

Each source code is stored inside the directory corresponding to its figure.

The naming convention is:

```text
figure1/figure1.py
figure2/figure2.py
figure3/figure3.py
figure4/figure4.py
figure5/figure5.py
figure6/figure6.py
figure7/figure7.py
```

To list all directories on Linux or macOS:

```bash
ls
```

Expected output:

```text
figure1
figure2
figure3
figure4
figure5
figure6
figure7
```

To locate all Python files recursively:

```bash
find . -name "*.py"
```

Expected output:

```text
./figure1/figure1.py
./figure2/figure2.py
./figure3/figure3.py
./figure4/figure4.py
./figure5/figure5.py
./figure6/figure6.py
./figure7/figure7.py
```

On Windows PowerShell, use:

```powershell
Get-ChildItem -Recurse -Filter *.py
```

To inspect the contents of one figure directory:

```bash
ls figure3
```

On Windows PowerShell:

```powershell
Get-ChildItem figure3
```

---

# 4. Python requirements

The simulations were implemented in Python.

The main required packages are:

```text
Python
NumPy
Matplotlib
```

Recommended versions:

```text
Python 3.10 or newer
NumPy 1.24 or newer
Matplotlib 3.7 or newer
```

The code may also use modules from the Python standard library, including:

```text
itertools
os
pathlib
time
```

These standard-library modules do not need to be installed separately.

---

# 5. Creating a virtual environment

Using a virtual environment is recommended to avoid conflicts with other Python installations.

## Linux or macOS

Create the environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install numpy matplotlib
```

## Windows PowerShell

Create the environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install the dependencies:

```powershell
pip install numpy matplotlib
```

## Windows Command Prompt

Create the environment:

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate
```

Install the dependencies:

```cmd
pip install numpy matplotlib
```

---

# 6. Checking the installation

Check the Python version:

```bash
python --version
```

Check the NumPy installation:

```bash
python -c "import numpy; print(numpy.__version__)"
```

Check the Matplotlib installation:

```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

Test both packages together:

```bash
python -c "import numpy; import matplotlib; print('Environment ready')"
```

Expected output:

```text
Environment ready
```

---

# 7. Compiling and executing the codes

Python scripts are normally executed directly rather than manually compiled.

In this repository, the expression “compile the code” means running the corresponding Python script with the Python interpreter.

The general command is:

```bash
python figureN/figureN.py
```

where `N` is the number of the figure.

For example:

```bash
python figure3/figure3.py
```

Some scripts may read or write files relative to their own directory. For maximum compatibility, it is recommended to enter the figure directory before running the code.

Example:

```bash
cd figure3
python figure3.py
```

After completion, return to the main `PAPER` directory:

```bash
cd ..
```

---

# 8. Reproducing each figure

## Figure 1 — Conceptual workflow

Figure 1 illustrates:

- DNA one-hot encoding;
- local nucleotide window;
- multichannel Toeplitz kernel;
- kernel learning;
- sequence reconstruction;
- residual analysis.

Run:

```bash
cd figure1
python figure1.py
cd ..
```

Or directly:

```bash
python figure1/figure1.py
```

---

## Figure 2 — Numerical identifiability

Figure 2 evaluates:

- local sequence signatures;
- signature collisions after numerical rounding;
- unique-signature fraction;
- minimum Euclidean distance between signatures;
- effects of context length and number of channels.

Run:

```bash
cd figure2
python figure2.py
cd ..
```

Or directly:

```bash
python figure2/figure2.py
```

This calculation may require more memory and processing time because all nucleotide contexts are enumerated for different context lengths.

---

## Figure 3 — Kernel calibration

Figure 3 evaluates:

- ground-truth Toeplitz kernel;
- calibrated Toeplitz kernel;
- least-squares estimation;
- signal-prediction error;
- sequence-reconstruction accuracy;
- effect of calibration-set size.

Run:

```bash
cd figure3
python figure3.py
cd ..
```

Or directly:

```bash
python figure3/figure3.py
```

---

## Figure 4 — Reconstruction under noise

Figure 4 evaluates:

- Gaussian measurement noise;
- reconstruction using the ground-truth kernel;
- reconstruction using the learned kernel;
- signal-prediction error;
- kernel deviation;
- distribution of reconstruction accuracy.

Run:

```bash
cd figure4
python figure4.py
cd ..
```

Or directly:

```bash
python figure4/figure4.py
```

---

## Figure 5 — Residual localization

Figure 5 evaluates:

- synthetic localized signal perturbations;
- observed and reconstructed signals;
- multichannel residual magnitude;
- residual smoothing;
- candidate ranking;
- event clustering;
- precision, recall, and F1 score.

Run:

```bash
cd figure5
python figure5.py
cd ..
```

Or directly:

```bash
python figure5/figure5.py
```

---

## Figure 6 — Separation of nearby perturbations

Figure 6 evaluates:

- two nearby synthetic perturbations;
- event-cluster count;
- event-separation rate;
- detection F1 score;
- effect of positional distance.

Run:

```bash
cd figure6
python figure6.py
cd ..
```

Or directly:

```bash
python figure6/figure6.py
```

This script performs multiple independent realizations and may take longer than scripts used only for schematic figures.

---

## Figure 7 — Signal-sample insertion and deletion

Figure 7 evaluates:

- duplication of a signal sample;
- removal of a signal sample;
- alignment sensitivity;
- residual response;
- event localization error.

Run:

```bash
cd figure7
python figure7.py
cd ..
```

Or directly:

```bash
python figure7/figure7.py
```

---

# 9. Recommended complete workflow

The following sequence reproduces the complete workflow.

## Step 1 — Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

## Step 2 — Extract the source-code archive

```bash
tar -xzf PAPER.tar.gz
```

## Step 3 — Enter the source-code directory

```bash
cd PAPER
```

## Step 4 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Step 5 — Install the dependencies

```bash
pip install numpy matplotlib
```

## Step 6 — Locate the Python codes

```bash
find . -name "*.py"
```

## Step 7 — Execute the codes

```bash
python figure1/figure1.py
python figure2/figure2.py
python figure3/figure3.py
python figure4/figure4.py
python figure5/figure5.py
python figure6/figure6.py
python figure7/figure7.py
```

When relative file paths are used, execute each script from inside its own directory:

```bash
cd figure1
python figure1.py
cd ../figure2
python figure2.py
cd ../figure3
python figure3.py
cd ../figure4
python figure4.py
cd ../figure5
python figure5.py
cd ../figure6
python figure6.py
cd ../figure7
python figure7.py
cd ..
```

---

# 10. Running all figures automatically

On Linux or macOS, all scripts can be executed sequentially with:

```bash
for i in 1 2 3 4 5 6 7
do
    echo "Running Figure ${i}"
    cd figure${i}
    python figure${i}.py
    cd ..
done
```

A version that stops when an error occurs is:

```bash
set -e

for i in 1 2 3 4 5 6 7
do
    echo "Running Figure ${i}"
    (
        cd figure${i}
        python figure${i}.py
    )
done
```

On Windows PowerShell:

```powershell
1..7 | ForEach-Object {
    Write-Host "Running Figure $_"
    Push-Location "figure$_"
    python "figure$_.py"
    Pop-Location
}
```

---

# 11. Output files

Each script generates the output associated with its corresponding figure.

Depending on the script configuration, outputs may include:

```text
PNG images
PDF images
numerical values printed in the terminal
NumPy data files
text files
intermediate simulation results
```

Generated files are normally saved:

```text
inside the corresponding figure directory
```

or:

```text
inside an output subdirectory
```

For example:

```text
figure3/figure3.png
```

or:

```text
figure3/output/figure3.png
```

To find all generated PNG files:

```bash
find . -name "*.png"
```

To find all generated PDF files:

```bash
find . -name "*.pdf"
```

On Windows PowerShell:

```powershell
Get-ChildItem -Recurse -Filter *.png
```

---

# 12. Computational workflow

The general computational procedure implemented by the scripts is:

```text
DNA sequence
      |
      v
One-hot encoding
      |
      v
Local sequence windows
      |
      v
Multichannel Toeplitz kernel
      |
      v
Synthetic molecular-sensing signal
      |
      +------------------------------+
      |                              |
      v                              v
Kernel calibration            New observed signal
      |                              |
      v                              v
Learned kernel          Dynamic-programming decoding
                                     |
                                     v
                           Reconstructed DNA sequence
                                     |
                                     v
                         Reconstructed canonical signal
                                     |
                                     v
                                  Residual
                                     |
                                     v
                         Anomaly/event localization
```

---

# 13. Main model parameters

The default computational model uses:

```text
DNA alphabet: A, T, G, C
Sequence representation: one-hot encoding
Context length: K = 5
Interaction radius: r = 2
Signal channels: C = 3
Boundary condition: zero padding
Noise model: independent Gaussian noise
Kernel calibration: unregularized least squares
Sequence reconstruction: Viterbi-style dynamic programming
```

Individual figures may use different:

```text
sequence lengths
noise amplitudes
numbers of calibration sequences
numbers of independent realizations
smoothing windows
clustering radii
event-matching tolerances
```

The exact values are defined inside the corresponding Python scripts and described in the paper.

---

# 14. Reproducibility

Random seeds are defined in the scripts to reproduce the computational experiments.

For example, the paper uses a default NumPy random-number generator seed of:

```text
7
```

Some experiments derive additional seeds from:

```text
context length
channel count
calibration-set size
noise level
event separation
realization number
```

Minor numerical differences may occur due to:

```text
Python version
NumPy version
Matplotlib version
operating system
BLAS or LAPACK implementation
processor architecture
floating-point arithmetic
```

These differences should not substantially alter the qualitative results.

---

# 15. Expected execution time

Execution time depends on:

```text
computer hardware
number of processor cores
available memory
Python and NumPy versions
number of independent realizations
number of enumerated nucleotide contexts
```

Figures involving context enumeration or repeated simulations may require more time.

In particular:

```text
Figure 2 may require substantial memory and processing time for K = 9.
Figures 3 and 4 perform repeated calibration and reconstruction experiments.
Figure 6 performs multiple realizations for several event separations.
Figure 7 performs repeated insertion and deletion experiments.
```

The terminal should remain open until the script finishes.

---

# 16. Troubleshooting

## Error: `python: command not found`

Try:

```bash
python3 figure1.py
```

Check the installation:

```bash
python3 --version
```

---

## Error: `No module named numpy`

Install NumPy:

```bash
pip install numpy
```

Or:

```bash
python -m pip install numpy
```

---

## Error: `No module named matplotlib`

Install Matplotlib:

```bash
pip install matplotlib
```

Or:

```bash
python -m pip install matplotlib
```

---

## Error: script cannot find a file

Enter the directory containing the script before executing it.

Example:

```bash
cd figure5
python figure5.py
```

This is important when the script uses relative paths.

---

## Error: permission denied

Run the script through Python:

```bash
python figure2.py
```

It is not necessary to make the `.py` file directly executable.

---

## Error while extracting `PAPER.tar.gz`

Verify the file:

```bash
ls -lh PAPER.tar.gz
```

Test the archive:

```bash
tar -tzf PAPER.tar.gz
```

Extract it again:

```bash
tar -xzf PAPER.tar.gz
```

---

## Matplotlib cannot open a display

On a remote server or computing cluster, use a non-interactive backend:

```bash
export MPLBACKEND=Agg
```

Then execute the script:

```bash
python figure4.py
```

Alternatively, add this before importing `matplotlib.pyplot`:

```python
import matplotlib
matplotlib.use("Agg")
```

---

## The simulation is taking a long time

This may be expected for experiments involving:

```text
large context spaces
many calibration sequences
multiple noise levels
multiple independent realizations
pairwise signature-distance calculations
```

Monitor memory and CPU usage before interrupting the process.

---

# 17. Verifying the extracted archive

To view the contents without extraction:

```bash
tar -tzf PAPER.tar.gz
```

To show only Python files:

```bash
tar -tzf PAPER.tar.gz | grep "\.py$"
```

To check the archive size:

```bash
du -h PAPER.tar.gz
```

To calculate a SHA-256 checksum:

```bash
sha256sum PAPER.tar.gz
```

On macOS:

```bash
shasum -a 256 PAPER.tar.gz
```

On Windows PowerShell:

```powershell
Get-FileHash PAPER.tar.gz -Algorithm SHA256
```

The checksum can be used to verify that the downloaded archive is complete and unchanged.

---

# 18. Updating the archive

When updating the source codes, recreate the archive from the directory containing `PAPER`:

```bash
tar -czf PAPER.tar.gz PAPER/
```

Check its contents:

```bash
tar -tzf PAPER.tar.gz
```

Then update the GitHub repository:

```bash
git add PAPER.tar.gz README.md
git commit -m "Update paper source codes"
git push origin main
```

---

# 19. Paper figures and corresponding directories

| Paper figure | Source-code directory | Main script | Main purpose |
|---|---|---|---|
| Figure 1 | `figure1/` | `figure1.py` | Conceptual Toeplitz sensing workflow |
| Figure 2 | `figure2/` | `figure2.py` | Numerical identifiability of local signatures |
| Figure 3 | `figure3/` | `figure3.py` | Kernel calibration and reconstruction |
| Figure 4 | `figure4/` | `figure4.py` | Noise robustness |
| Figure 5 | `figure5/` | `figure5.py` | Residual localization of perturbations |
| Figure 6 | `figure6/` | `figure6.py` | Separation of nearby perturbations |
| Figure 7 | `figure7/` | `figure7.py` | Signal-sample insertion and deletion |

---

# 20. Scientific scope

The repository implements controlled synthetic experiments.

The model assumes:

```text
a linear sensing response
finite local context
translation-invariant coefficients
fixed signal-to-sequence alignment
Gaussian independent noise
synthetic DNA sequences
```

The framework is intended as an interpretable computational model for investigating:

```text
identifiability
kernel calibration
sequence reconstruction
noise sensitivity
residual analysis
anomaly localization
alignment limitations
```

The synthetic results should not be interpreted as direct performance measurements of a specific experimental nanopore device.

---

# 21. Citation

If this source code or methodology is used in academic work, please cite the associated paper:

```text
Raphael Matozo Tromer,
Alysson Martins Almeida Silva,
Ketankumar A. Ganure,
Santosh K. Tiwari,
and Luiz Antonio Ribeiro Junior.

A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing.
```

The complete journal citation, DOI, volume, issue, and page information should be added after publication.

A BibTeX entry can be included when the publication information becomes available.

Example placeholder:

```bibtex
@article{tromer_toeplitz_molecular_sensing,
  title   = {A Unified Toeplitz Framework for Sequence-Dependent Molecular Sensing},
  author  = {Tromer, Raphael Matozo and
             Silva, Alysson Martins Almeida and
             Ganure, Ketankumar A. and
             Tiwari, Santosh K. and
             Ribeiro Junior, Luiz Antonio},
  journal = {To be added},
  year    = {To be added},
  doi     = {To be added}
}
```

---

# 22. Authors

- Raphael Matozo Tromer
- Alysson Martins Almeida Silva
- Ketankumar A. Ganure
- Santosh K. Tiwari
- Luiz Antonio Ribeiro Junior

---

# 23. Corresponding author

**Luiz Antonio Ribeiro Junior**

Computational Materials Laboratory  
Institute of Physics  
University of Brasília  
Brasília, Federal District, Brazil

---

# 24. License

The source code is distributed under the license included in this repository.

Please read the `LICENSE` file before redistributing or modifying the source code.

---

# 25. Contact and issues

For scientific questions related to the methodology, contact the corresponding author.

For problems related to:

```text
archive extraction
missing files
Python dependencies
code execution
figure reproduction
```

open an issue in the GitHub repository and provide:

```text
operating system
Python version
NumPy version
Matplotlib version
figure number
complete error message
command used to execute the script
```

Example issue description:

```text
Operating system: Ubuntu 24.04
Python version: 3.12
NumPy version: 2.0
Matplotlib version: 3.9
Figure: Figure 4
Command: python figure4.py
Error: [paste the complete error message here]
```

---

## Minimal command summary

```bash
git clone https://github.com/USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
tar -xzf PAPER.tar.gz
cd PAPER
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python figure1/figure1.py
python figure2/figure2.py
python figure3/figure3.py
python figure4/figure4.py
python figure5/figure5.py
python figure6/figure6.py
python figure7/figure7.py
```
