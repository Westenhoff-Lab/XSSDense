# XSSDense

<!--[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](#installation)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)-->

## Reconstructing Electron Densities from X-ray Solution Scattering Data Using Deep Generative Models

XSSDense is a framework for reconstructing three-dimensional electron-density maps from X-ray solution scattering (XSS) data using a β-Variational Autoencoder (β-VAE) and latent-space optimization.

The framework combines:

- Structural model voxelisation
- Deep generative learning using β-VAEs
- Latent-space statistical analysis
- Genetic-algorithm optimization
- Reconstruction of electron-density maps directly from experimental XSS measurements

---

## Workflow

```text
PDB Structures
      │
      ▼
Voxelisation
      │
      ▼
3D Density Grids
      │
      ▼
TFRecord Generation
      │
      ▼
β-VAE Training
      │
      ▼
Latent Space Statistics
      │
      ▼
Genetic Algorithm Reconstruction
      │
      ▼
3D Electron Density Map
```
# Method Overview

## Voxelisation

Atomic coordinates are converted into fixed-size electron-density volumes suitable for neural-network training.

## β-VAE Training

```text
Density Map
      │
      ▼
Encoder
      │
      ▼
Latent Vector
      │
      ▼
Decoder
      │
      ▼
Reconstructed Density Map
```

The VAE learns a compressed latent representation of structural variability.

## Latent-Space Analysis

Statistics of the encoded training distribution are computed and used to guide reconstruction toward physically realistic regions of latent space.

## Reconstruction

```text
Latent Vector
      │
      ▼
Decoder
      │
      ▼
Density Map
      │
      ▼
Scattering Calculation
      │
      ▼
Comparison with Experiment
      │
      ▼
Fitness Score
      │
      ▼
Evolution
```

# System Requirements
All code has been tested on a system using the following setup: 

### Operating System
- Rocky Linux 8.10

### CUDA version
- 13.2

### Python
- Python 3.11.3

### Main Dependencies
- TensorFlow 2.20
- NumPy 1.26.4
- SciPy 1.17
- Matplotlib 3.10
- scikit-learn 1.6.1
- CuPy-cuda13x 13.6.0
- Gemmi 0.7.4
- tqdm 4.67.1


### Hardware

- GPU: NVIDIA Tesla A100 HGX GPU 40GB VRAM
- CPU: 2 x 32 core Intel(R) Xeon(R) Gold 6338

# Installation
Clone the repository:

```bash
git clone https://github.com/your-username/XSSDense.git
cd XSSDense
```

Install dependencies:

```bash
pip install tensorflow numpy scipy matplotlib scikit-learn cupy-cuda13x gemmi tqdm
```

Depending on your CUDA version and GPU architecture, versions other than those listed above may be required. 

### Installation Time
Installation typically requires 10–20 minutes, depending on the TensorFlow and CUDA configuration.
 

---

# Quick Start Tutorial

The example dataset available at https://doi.org/10.5281/zenodo.21915224 reproduces the LOV2 unfolding case presented in the accompanying manuscript. By following the workflow described below, users can reproduce the complete XSSDense pipeline, validate the software installation, and verify the expected outputs.

## Example Dataset

```text
example/
├── voxel_maps.h5
├── ground_state.dat
├── difference_signal.dat
└── reference_results/
```

Expected outputs should resemble the results contained in:

```text
example/reference_results/
```

---

## Step 1: Generate TFRecords

Convert voxelized density maps into TensorFlow TFRecords.

Expected runtime: ~5 min.

```bash
python generate_tfrecord.py \
    example/voxel_maps.h5 \
    example_output/
```

Output:

```text
example_output/
├── tfrecords/
│   ├── train.tfrecord
│   └── test.tfrecord
└── meta/
    └── meta.json
```

---

## Step 2: Train the β-VAE

```bash
python Train_VAE.py \
    example_output/tfrecords/train.tfrecord \
    example_output/tfrecords/test.tfrecord \
    lov2_model \
    late \
    1
```

The output directory created contains the final encoder and decoder model as well as the weights of them over the different epochs and a log file that contains the losses over the epochs. 

Expected runtime: ~2 hrs .

Output:

```text
lov2_model/
├── encoder_model.keras
├── decoder_model.keras
├── vae_epoch_*.weights.h5
├── vae_final.weights.h5
└── log.txt
```

---

## Step 3: Generate Latent-Space Statistics

```bash
python process_training.py \
    lov2_model/ \
    example_output/tfrecords/train.tfrecord \
    lov2_model/
``` 

Output:

```text
lov2_model/
├── encoder_model.keras
├── decoder_model.keras
├── vae_epoch_*.weights.h5
├── vae_final.weights.h5
├── log.txt
├── latent_pca.png
└── latent_stats.txt
```

Expected runtime: 5 min.
---

## Step 4: Reconstruct Electron Density

```bash
python reconstruct.py \
    --model_path lov2_model/ \
    --ground_scattering example/ground_state.dat \
    --iq_path example/difference_signal.dat \
    --params example_output/meta/meta.json \
    --voxel  example/voxel_maps.h5\
    --output_folder lov2_reconstruction \
    --meta_json 2.4 \
    --latent_size 8 \
    --batch_size 16 \
    --max_iter 30 \
    --target_yield 0.15 \
    --yield_weight 2
```

Output:

```text
lov2_reconstruction/
├── final_rank0_diff.ccp4
├── final_rank0_extrapolated.ccp4
├── final_rank0_latent.npy
├── final_rank0_score.npy
└── log.txt
```

Expected runtime: 50 min.

# Using Your Own Data

## Instructions for Use

To analyze a new system, users should:

1. Generate voxelized density maps from structural models.
2. Convert the voxel maps to TFRecords.
3. Train a β-VAE model.
4. Compute latent-space statistics.
5. Supply experimental scattering data.
6. Run latent-space reconstruction.
7. Analyze the resulting CCP4 density maps.

## 1. Voxelize Structures

Input:

```text
pdb_structures/
├── structure1.pdb
├── structure2.pdb
└── ...
```

Output:

```text
voxel_maps.h5
```

---

## 2. Generate TFRecords

```bash
python generate_tfrecord.py voxel_maps.h5 project_data/
```

---

## 3. Train a β-VAE

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    experiment_name \
    late \
    1.0
```

### Training Modes

| Mode | Description |
|------|-------------|
| constant | Fixed β throughout training |
| late | Gradual β warm-up during training |

---

## 4. Generate Latent Statistics

```bash
python process_training_norms_absolute52.py \
    trained_model/ \
    train.tfrecord \
    latent_statistics/
```

---

## 5. Reconstruction

```bash
python main_reconstruction_ga_may25_absolute.py \
    --model_dir trained_model/ \
    --iq ground_state.dat \
    --diff difference_signal.dat \
    --ground_state dark.npy \
    --latent_stats latent_statistics/ \
    --output reconstruction/
```

### Important Parameters

| Parameter | Description |
|-----------|-------------|
| target yield | Excited-state population |
| yield weight | Weight applied to yield optimization |
| population size | Number of GA candidates |
| batch size | Candidates evaluated simultaneously |
| max iterations | Number of optimization generations |

---

A genetic algorithm searches latent space for density maps that best reproduce the experimental XSS signal.

# Citation

```bibtex
@article{Monrroy2026.08.07.743437,
  author = {Monrroy, Leonardo and Cardoch, Sebastian and Westenhoff, Sebastian},
  title = {XSSDense: Time-resolved X-ray Solution Scattering Density Reconstruction Using a Variational Autoencoder},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.08.07.743437}
}
```
# License

XSSDense is distributed under the GNU General Public License v3.0 (GPL-3.0).

See the LICENSE file for the full license text.


# Code Availability

The XSSDense source code is freely available at:

https://github.com/Westenhoff-Lab/XSSDense

The example dataset used to validate installation and reproduce the workflow is available at:

https://doi.org/10.5281/zenodo.21915224

The software is distributed under the GNU General Public License v3.0 (GPL-3.0).
