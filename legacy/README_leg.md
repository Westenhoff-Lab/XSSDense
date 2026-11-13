# XSSDense

<!--[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](#installation)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)-->

## Reconstructing Electron Densities from X-ray Solution Scattering Data Using Deep Generative Models

XSSDense is a framework for reconstructing three-dimensional electron-density maps from X-ray solution scattering (XSS) data using a β-Variational Autoencoder (β-VAE) and latent-space optimization.

The framework combines:

- Voxelisation of structural models (PDB ensembles or MD trajectories)
- Deep generative learning using β-VAEs
- Latent-space statistical analysis
- Genetic-algorithm optimization
- Reconstruction of electron-density maps directly from experimental XSS measurements

## Workflow

Each step is a command installed with the package:

```text
PDB ensemble / MD trajectory
      ▼
xssdense-voxelize        →  <name>.h5         (3D density grids)
      ▼
xssdense-gen-tfrecord    →  <name>.tfrecord/  (train/test tfrecords)
      ▼
xssdense-train-vae       →  <name>.vae/       (encoder, decoder)
      ▼
xssdense-ga              →  results/          (ranked density maps)
```

## Method Overview

### Voxelisation

Atomic coordinates are centred and converted into fixed-size electron-density volumes on a cubic grid of `n_voxels`^3 points (52^3 by default). The voxel size follows from the bounding box: `dv = 2·bbmax / (n_voxels − 1)` where `bbmax` is the max dimensions of the system (including marging).

### β-VAE Training

```text
Density Map
      ▼
Encoder
      ▼
Latent Vector
      ▼
Decoder
      ▼
Reconstructed Density Map
```

The VAE learns a compressed latent representation of structural variability. At the end of training, the best model is saved and the latent-space statistics (mean, std, min, max per dimension) are saved in a txt file.

### Latent-Space Analysis

The latent-space statistics are used to guide reconstruction toward physically realistic regions of latent space.

### Reconstruction

```text
Latent Vector
      ▼
Decoder
      ▼
Density Map
      ▼
Scattering Calculation (solvent envelope + hydration shell)
      ▼
Comparison with measured ΔI(q)
      ▼
Fitness Score (fit quality + yield penalty)
      ▼
Evolution
```

Before the search, the dark-state curve is fitted (absolute scale, hydration-shell contrast and thickness). These solvent parameters are then held fixed while light-state candidates are evaluated against ΔI(q).

## System Requirements
All code has been tested on a system using the following setup:

### CUDA
- 13x or 12x

### Main Dependencies
- TensorFlow 2.20
- NumPy 1.26.4
- SciPy 1.17
- h5py
- Matplotlib 3.10
- scikit-learn 1.6.1
- Gemmi 0.7.4
- tqdm 4.67.1
- keras 3.13.1
- MDAnalysis 2.10

## Installation
Clone the repository:

```bash
git clone https://github.com/Westenhoff-Lab/XSSDense.git
cd XSSDense
```

TensorFlow and CuPy must match your CUDA setup, so always install with **exactly one** of these options (check your CUDA version with `nvidia-smi`). This also installs the `xssdense-*` commands:

```bash
# CUDA 12
pip install -e ".[cuda12]"

# CUDA 13
pip install -e ".[cuda13]"

# NVIDIA TensorFlow container: TensorFlow and CuPy come from the container
pip install -e ".[container]"
```

**A plain `pip install -e .` installs neither TensorFlow nor CuPy, and the `xssdense-*` commands will not run.** Versions other than those listed above may be required.

Then check the installation on a node with a GPU:

```bash
xssdense-check
```

It reports whether each dependency imports and whether CuPy and TensorFlow see a GPU, and exits with code 1 if anything fails.



## Example: Crambin

A complete small example is included in `examples/crambin.ipynb`. From an MD trajectory of crambin (PDB 1CRN), it clusters the frames into two classes, builds synthetic ΔI(q) data from each class, and recovers the light-state density with the genetic algorithm.

### Data(Crambin)

The psf and dcd files are available at https://doi.org/10.5281/zenodo.21915224.

```text
examples/crambin/
├── crambin.ipynb
├── crambin_saxs_class.py
└── inputs/
    ├── 1crn.psf
    └── 1crn_eq_aligned.dcd
```

All outputs go into a `run/` directory.

### Step 1: Voxelise the trajectory

```bash
xssdense-voxelize \
    --topology examples/crambin/inputs/1crn.psf \
    --trajectory examples/crambin/inputs/1crn_eq_aligned.dcd \
    --bbmax 41 \
    --save_path run \
    --system 1crn
```

### Step 2: Generate TFRecords

```bash
xssdense-gen-tfrecord \
    --h5 run/1crn_41bb_1d61dv_5co.h5 \
    --output_dir run/1crn_41bb_1d61dv_5co.tfrecord
```

### Step 3: Train the β-VAE

```bash
xssdense-train-vae \
    --datapath run/1crn_41bb_1d61dv_5co.tfrecord \
    --savepath run \
    --epochs 30 \
    --mode constant \
    --beta_input 1e-1 \
    --batchsize 64
```

### Step 4: Create synthetic scattering data

This step is only needed for this example and is not part of the general pipeline.It clusters the per-frame scattering curves (PCA + k-means) and writes one synthetic dataset per class, e.g. `run/synthetic_class1_vacuum/` with `ground_abs.csv` (dark-state I(q)) and `deltaI.csv` (ΔI(q)).

```bash
python examples/crambin/crambin_saxs_class.py \
    --h5 run/1crn_41bb_1d61dv_5co.h5 \
    --meta run/1crn_41bb_1d61dv_5co.tfrecord/meta/meta.json \
    --out_dir run \
    --rho_bulk 0 \
    --padding 3
```

### Step 5: Reconstruct the electron density

```bash
xssdense-ga \
    --model_path run/1crn_41bb_1d61dv_5co.vae \
    --meta run/1crn_41bb_1d61dv_5co.tfrecord/meta/meta.json \
    --h5 run/1crn_41bb_1d61dv_5co.h5 \
    --dark_curve run/synthetic_class1_vacuum/ground_abs.csv \
    --dI_curve run/synthetic_class1_vacuum/deltaI.csv \
    --rho_bulk 0 \
    --target_yield 1 \
    --padding 3 \
    --qmax 0.6 \
    --output_folder run/results/synthetic_class1_vacuum
```

## Datasets

The dataset available at https://doi.org/10.5281/zenodo.21915224 reproduces the Crambin synthetic and LOV2 unfolding case.

## Citation

```bibtex
@article{Monrroy2026.08.07.743437,
  author = {Monrroy, Leonardo and Cardoch, Sebastian and Westenhoff, Sebastian},
  title = {XSSDense: Time-resolved X-ray Solution Scattering Density Reconstruction Using a Variational Autoencoder},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.08.07.743437}
}
```

## License

XSSDense is distributed under the GNU General Public License v3.0 (GPL-3.0).

See the LICENSE file for the full license text.

## Code Availability

The XSSDense source code is freely available at: https://github.com/Westenhoff-Lab/XSSDense
