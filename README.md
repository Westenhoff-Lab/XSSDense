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

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/XSSDense.git
cd XSSDense
```

Install dependencies:

```bash
pip install tensorflow numpy scipy matplotlib scikit-learn cupy gemmi tqdm
```

Depending on your CUDA and TensorFlow configuration, additional packages may be required.

---

# Quick Start Tutorial

The example dataset found in: https://doi.org/10.5281/zenodo.21915224 reproduces the complete XSSDense pipeline and serves as an installation and workflow validation test.

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


# Using Your Own Data

## 1. Voxelise Structures

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

---

## Expected Outcome

After completing the workflow, users will obtain:

- A trained β-VAE model
- Latent-space statistics
- Optimized latent-space solutions
- Three-dimensional reconstructed electron-density maps
- Scattering profiles consistent with experimental XSS observations
