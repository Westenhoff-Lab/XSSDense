# XSSDense
## Reconstructing Electron Densities from X-ray Solution Scattering Data Using Deep Generative Models

XSSDense is a computational framework for reconstructing three-dimensional electron-density maps from X-ray solution scattering (XSS) data using a β-Variational Autoencoder (β-VAE) and genetic-algorithm optimization.

The framework combines structural modelling, deep generative learning, and experimental scattering data to reconstruct electron-density distributions of molecular systems.

## Workflow

The XSSDense workflow consists of four main stages:

- Structural voxelisation and electron-density calculation.
- TensorFlow TFRecord generation.
- β-VAE training and latent-space characterization.
- Genetic-algorithm reconstruction against experimental XSS measurements.
```
       PDB Structures / MD Trajectory
                    |
                    v
              voxelize.py
                    |
                    v
         Electron-Density Grids
                (HDF5)
                    |
                    v
          generate_tfrecord.py
                    |
                    v
             TFRecord Files
                    |
                    v
              train_vae.py
                    |
                    v
               Trained VAE
                    |
             +------+------+
             |             |
             v             v
       Encoder/Decoder  Latent Statistics
             |             |
             +------+------+
                    |
                    v
          genetic_algorithm.py
                    |
                    v
        GPU Scattering Calculation
                    |
                    v
          Comparison with Experiment
                    |
                    v
             Genetic-Algorithm Search
                    |
                    v
          Reconstructed Electron Density
```

## Method Overview
### 1. Structural Voxelisation

Atomic coordinates are converted into three-dimensional electron-density distributions using voxelize.py.

The script supports two input formats:

- Ensembles of aligned PDB structures.
- Molecular-dynamics trajectories with an associated topology.

Electron densities are calculated using atomic form factors and Gaussian representations of the atomic electron distributions.

The resulting density maps are stored in an HDF5 file containing:

| Dataset / Metadata | Description |
| --- | --- |
| rho | Three-dimensional electron-density distributions. |
| dv | Voxel spacing in Å. |
| bbmax | Bounding-box dimension. |
| Additional metadata | Input structures and voxelisation parameters. |

The default grid resolution is 52 × 52 × 52 voxels.

### 2. TFRecord Generation

The HDF5 density dataset is converted into TensorFlow TFRecords using generate_tfrecord.py.

The script performs global min–max normalization:

$$\rho_{\mathrm{norm}} = \frac{\rho - \rho_{\min}}{\rho_{\max} - \rho_{\min}}$$

The dataset is divided into training and testing subsets. For molecular-dynamics trajectories, the splitting is performed in blocks to reduce correlations between neighbouring frames.

The output includes the TFRecord files and a metadata JSON file containing the normalization and electron-conservation information required by subsequent stages.

### 3. β-VAE Training

The normalized density maps are used to train a three-dimensional β-VAE using train_vae.py.

```
      Electron-Density Grid
                |
                v
              Encoder
                |
                v
        Latent Distribution
                |
                v
         Latent Sampling
                |
                v
              Decoder
                |
                v
       Reconstructed Density
```


The training objective combines reconstruction loss and KL-divergence regularisation.

The β parameter controls the contribution of latent-space regularisation.

The implementation supports two β scheduling modes:

- constant: Fixed β throughout training.
- late: Gradual β warm-up during training.

Following training, the script automatically saves the encoder, decoder, training statistics, and latent representations of the training dataset.

### 4. Genetic-Algorithm Reconstruction

The reconstruction is performed using genetic_algorithm.py.

The algorithm searches the trained VAE latent space for density distributions that reproduce the experimentally measured difference-scattering signal.

```
           Training Latent Space
                    |
                    v
           Candidate Latents
                    |
                    v
                  Decoder
                    |
                    v
          Electron-Density Maps
                    |
                    v
          GPU Scattering Calculation
                    |
                    v
          Dark / Light State Comparison
                    |
                    v
          Experimental Difference Signal
                    |
                    v
              Fitness Evaluation
                    |
                    v
             Genetic Algorithm
                    |
                    v
          Optimised Latent Candidates
                    |
                    v
          Reconstructed Density Maps
```


The scattering calculation accounts for solvent contributions and hydration-shell effects.

**The genetic algorithm uses:**

- Latent-space population initialization.
- Tournament selection.
- Crossover.
- Gaussian mutation.
- Elitism.
- Introduction of new candidates.
- Early stopping based on convergence.

Candidate solutions are ranked according to their agreement with the experimental difference-scattering signal and the specified target excited-state yield.

## System Requirements

All code has been tested on a system using the following configuration.

## Software
| Component | Tested version |
| --- | --- |
| Operating System | Rocky Linux 8.10 |
| CUDA | 13.2 |
| Python | 3.11.3 |
| TensorFlow | 2.20 |
| Keras | 3.13.1 |
| NumPy | 1.26.4 |
| SciPy | 1.17 |
| Matplotlib | 3.10 |
| scikit-learn | 1.6.1 |
| CuPy-cuda13x | 13.6.0 |
| Gemmi | 0.7.4 |
| tqdm | 4.67.1 |
| MDAnalysis | 2.10 |
| h5py | Required |
| pandas | Required |
## Hardware

**Reference configuration:**

- GPU: NVIDIA Tesla A100 HGX GPU, 40 GB VRAM.
- CPU: 2 × 32-core Intel Xeon Gold 6338.

The software uses TensorFlow and CuPy for GPU acceleration.

Note: TensorFlow and CuPy should be installed with versions compatible with the local CUDA environment and GPU architecture. The optional GPU dependencies specified in pyproject.toml should also be checked against the installed CUDA version.
## Installation

Clone the repository:

```bash
git clone https://github.com/Westenhoff-Lab/XSSDense.git
```

cd XSSDense


Install XSSDense and its Python dependencies:

```bash
pip install -e .
```


Install TensorFlow and the appropriate CuPy distribution for your CUDA environment.

For the reference CUDA 13 configuration:

```bash
pip install tensorflow==2.20
pip install cupy-cuda13x==13.6.0
```


The installation provides four command-line entry points:

```bash
xssdense-voxelize
xssdense-gen-tfrecord
xssdense-train-vae
xssdense-ga
```


These correspond to the four main workflow stages.

## Quick Start Tutorial

The example dataset available on Zenodo reproduces the LOV2 unfolding case presented in the accompanying manuscript.

Example dataset: https://doi.org/10.5281/zenodo.21915224

The dataset can be used to reproduce the complete XSSDense workflow and validate the software installation.

## Example Dataset

**The example workflow requires:**

```
example/
├── LOV2_ref.dat.dat
├── LOV2_diff_intermediate.dat
├── LOV2_diff_unfolded.dat
├── reference_results/
└── pdb_structures/
```


The PDB structures should be aligned to a common reference structure.

The experimental scattering files should contain two-column CSV data without headers:

- First column: q (Å⁻¹).
- Second column: Scattering intensity.

The reconstruction script uses the ground-state scattering curve and the measured difference-scattering curve as separate inputs.

### Step 1: Voxelise PDB Structures

Convert the aligned PDB ensemble into three-dimensional electron-density maps.

**Run:**

```bash
xssdense-voxelize \
    --bbmax 61 \
    --n_voxels 52 \
    --pdb_path pdb_structures \
    --reference_pdb pdb_structures/Ref_pred_aligned.sup.pdb \
    --save_path example_output/voxel_maps \
    --system LOV2
```

#### Important Parameters
| Parameter | Description |
| --- | --- |
| bbmax | Maximum extent of the cubic bounding box (Å). |
| n_voxels | Number of voxels per dimension. Default: 52. |
| pdb_path | Directory containing aligned PDB structures. |
| reference_pdb | Reference structure defining the common coordinate frame. |
| save_path | Output directory for the HDF5 density dataset. |
| system | Identifier used for naming the output dataset. |
| cutoff | Atomic density calculation cutoff radius (Å). Default: 5.0. |
| selection | MDAnalysis atom selection. Default: protein and not element H. |

The voxel spacing is automatically calculated from the bounding-box size and grid resolution.

For the example configuration, the voxel spacing is approximately 2.4 Å.

**Expected output:**

```
example_output/
└── voxel_maps/
    └── LOV2_61bb_2d39dv_5co.h5
```


The HDF5 file contains the electron-density maps and associated voxelisation metadata.

### Alternative: Molecular-Dynamics Trajectories

XSSDense also supports molecular-dynamics trajectories.

Provide a topology and trajectory file:

```bash
xssdense-voxelize \
    --bbmax 61 \
    --n_voxels 52 \
    --trajectory trajectory.dcd \
    --topology topology.psf \
    --save_path example_output/voxel_maps \
    --system LOV2
```


The trajectory is processed frame by frame, with each frame converted into an electron-density distribution.

The resulting HDF5 dataset contains the density maps for all processed frames.

### Step 2: Generate TFRecords

Convert the HDF5 electron-density dataset into TensorFlow TFRecords.

**Run:**

```bash
xssdense-gen-tfrecord \
    --h5 example_output/voxel_maps/LOV2_61bb_2d39dv_5co.h5 \
    --output_dir example_output/LOV2_data
```

#### Important Parameters
Parameter	Description	Default
--h5	Input HDF5 density dataset.	Required
--output_dir	Output directory for TFRecords and metadata.	Required
--block	Number of consecutive frames per block.	100
--test_every	Every nth block is assigned to testing.	10
--num_proc	Number of processes used for serialization.	8

The script performs global min–max normalization and divides the dataset into training and testing subsets.

For MD trajectories, the block-based splitting helps prevent neighbouring correlated frames from being distributed between training and testing datasets.

**Expected output:**

```
example_output/
└── LOV2_data/
    ├── tfrecords/
    │   ├── train.tfrecords
    │   └── test.tfrecords
    └── meta/
        └── meta.json
```


The metadata file contains the normalization parameters and electron-conservation information required during VAE training and reconstruction.

### Step 3: Train the β-VAE

Train the β-VAE using the generated TFRecords.

**Run:**

```bash
xssdense-train-vae \
    --datapath example_output/LOV2_data \
    --savepath example_output/models \
    --mode late \
    --beta_input 1.0 \
    --epochs 50 \
    --batchsize 64
```

#### Important Parameters
Parameter	Description	Default
--datapath	Directory containing TFRecords and metadata.	Optional
--savepath	Directory where the trained model is saved.	.
--mode	β scheduling mode (constant or late).	Required
--beta_input	Target β regularisation parameter.	Required
--z_dim	Latent-space dimensionality.	8
--learning_rate	Initial Adam learning rate.	0.0003
--epochs	Maximum number of training epochs.	80
--batchsize	Training batch size.	64
--patience	Early-stopping patience.	5
--min_improvement	Minimum relative validation-MSE improvement.	0.005
--save_every_epoch	Save model weights after every epoch.	Disabled

Alternatively, the training and testing TFRecord files and metadata can be supplied individually using:

```
--train_datapath
--test_datapath
--meta_datapath
```


The --datapath option automatically resolves these three files.

## Training Outputs

The trained model is saved in a .vae directory.

For example:

```
example_output/
└── models/
    └── LOV2_data.vae/
        ├── encoder_model.keras
        ├── decoder_model.keras
        ├── vae_best.weights.h5
        ├── latent_stats.txt
        ├── latents_train.npy
        └── log.txt
```


**The output contains:**

- Trained encoder and decoder.
- Best-performing VAE weights.
- Training and validation loss information.
- Encoded training latent vectors.
- Latent-space statistics.
Important: Latent-space statistics are calculated automatically after training. No separate latent-processing script is required.

The latent_stats.txt file contains four columns for each latent dimension:

Mean.
Standard deviation.
Minimum.
Maximum.

These statistics are used by the genetic algorithm during reconstruction.

Expected runtime: Approximately 2 hours, depending on dataset size and hardware.

### Step 4: Reconstruct Electron Density

The final step reconstructs the excited-state electron density using the trained VAE and genetic algorithm.

The reconstruction script uses:

The trained VAE model.
- Latent-space statistics.
The original HDF5 electron-density dataset.
The TFRecord metadata.
Experimental ground-state scattering.
Experimental difference-scattering data.

**Run:**

```bash
xssdense-ga \
    --model_path example_output/models/LOV2_data.vae \
    --meta example_output/LOV2_data/meta/meta.json \
    --h5 example_output/voxel_maps/LOV2_61bb_2d39dv_5co.h5 \
    --dark_curve example/ground_state.dat \
    --dI_curve example/difference_signal.dat \
    --output_folder example_output/reconstruction \
    --rho_bulk 0.334 \
    --target_yield 0.15 \
    --yield_weight 5.0 \
    --qmax 0.6
```

#### Important Parameters
Parameter	Description	Default
--model_path	Trained VAE model directory.	Required
--meta	TFRecord metadata JSON file.	Required
--h5	Original HDF5 density dataset.	Required
--dark_index	Index of the ground-state density in the HDF5 dataset.	0
--dark_curve	Experimental ground-state scattering curve.	Required
--dI_curve	Experimental difference-scattering curve.	Required
--output_folder	Reconstruction output directory.	Required
--rho_bulk	Bulk solvent electron density (e/Å³).	0.334
--threshold	Density threshold for solvent displacement.	rho_bulk
--target_yield	Expected excited-state population.	0.5
--yield_weight	Weight of the yield penalty.	2.0
--qmin	Minimum q-value used for fitting.	0.0
--qmax	Maximum q-value used for fitting.	0.6
--padding	FFT padding factor.	3
--pop_size	Genetic-algorithm population size.	64
--max_generations	Maximum number of generations.	200
--patience	Generations without improvement before stopping.	25
--elite_fraction	Fraction of elite candidates retained.	0.1
--newcomer_fraction	Fraction of new candidates introduced per generation.	0.1
--pool_size	Number of training latents in the initial candidate pool.	1024
--mutation_rate	Fraction of latent dimensions mutated.	0.5
--step_min	Minimum mutation step in latent standard deviations.	0.03
--step_max	Maximum mutation step in latent standard deviations.	1.0
--top_k	Number of best-ranked candidates saved.	5
--seed	Random-number generator seed.	0
## Reconstruction Procedure

The reconstruction algorithm performs the following operations:

Loads the trained VAE and latent-space statistics.
Loads the ground-state density from the HDF5 dataset.
Encodes and decodes the ground-state density through the VAE.
Fits the ground-state scattering, including solvent and hydration-shell contributions.
Generates an initial population of candidate latent vectors from the training latent distribution.
Decodes the candidates into electron-density distributions.
Calculates their scattering curves using GPU-accelerated Fourier transforms.
Compares the calculated difference-scattering curves with the experimental measurements.
Evaluates candidates according to scattering agreement and the target-yield penalty.
Evolves the population using selection, crossover, and mutation.
Saves the highest-ranked reconstructed density maps.

The optimization terminates when the maximum number of generations is reached or the best fitness score has not improved for the specified patience interval.

## Reconstruction Outputs

The output directory contains:

```
reconstruction/
├── dark_state.ccp4
├── dark_state_vae.ccp4
├── dark_state_vae_density.npy
├── dark_latent.npy
├── rank0_latent.npy
├── rank0_density.npy
├── rank0_density.ccp4
├── rank0_deltaI.csv
├── rank1_latent.npy
├── rank1_density.npy
├── rank1_density.ccp4
├── rank1_deltaI.csv
├── ...
├── deltaI_fits.png
├── score_history.txt
└── summary.json
```


The number of ranked candidates is controlled by --top_k.

The CCP4 files can be visualized using molecular visualization software such as ChimeraX.

The summary.json file contains the reconstruction settings and information about each candidate, including:

Fitness score.
Estimated excited-state yield.
R² of the scattering fit.
Radius of gyration.
Distance from the dark-state latent representation.
## Using Your Own Data

To apply XSSDense to a new molecular system, follow the four-stage workflow:

Prepare an ensemble of aligned PDB structures or an MD trajectory with its topology.
Generate electron-density maps using voxelize.py.
Generate training and testing TFRecords using generate_tfrecord.py.
Train a β-VAE using train_vae.py.
Supply the experimental ground-state and difference-scattering curves.
Perform reconstruction using genetic_algorithm.py.
Analyse the reconstructed CCP4 density maps and scattering fits.
## Important Considerations
All PDB structures must be aligned to a common reference frame.
The grid dimensions must be consistent with the VAE architecture.
The voxel spacing must be retained in the original HDF5 file.
The original HDF5 dataset and corresponding metadata must be retained for reconstruction.
The experimental scattering curves must be supplied as two-column CSV files without headers.
The q-range used for fitting should be selected consistently with the experimental data.
The target yield should reflect the expected excited-state population.
## Citation

If you use XSSDense in your research, please cite the accompanying manuscript:

```bibtex
@article{Monrroy2026.08.07.743437,
  author = {Monrroy, Leonardo and Cardoch, Sebastian, Lee, Seoyoung and Westenhoff, Sebastian},
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

The XSSDense source code is freely available on GitHub:

Repository: https://github.com/Westenhoff-Lab/XSSDense

Source code: https://github.com/Westenhoff-Lab/XSSDense/tree/main/src/xssdense

The example dataset used to validate the installation and reproduce the LOV2 unfolding workflow is available on Zenodo:

Example dataset: https://doi.org/10.5281/zenodo.21915224
