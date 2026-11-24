# XSSDense

XSSDense is a computational framework for reconstructing three-dimensional electron-density maps from X-ray solution scattering (XSS) data using a β-Variational Autoencoder (β-VAE) and genetic-algorithm optimization.

## Workflow

The XSSDense workflow consists of four main stages:

- Electron-density calculation and voxelisation.
- TensorFlow TFRecord generation.
- β-VAE training and latent-space characterization.
- Genetic-algorithm reconstruction against experimental XSS measurements.

```
       PDB Structures / MD Trajectory
                    |
                    v
             xssdense-voxelize
                    |
                    v
         Electron-Density Grids
                (HDF5)
                    |
                    v
          xssdense-gen-tfrecord
                    |
                    v
             TFRecord Files
                    |
                    v
           xssdense-train-vae
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
               xssdense-ga
                    |
                    v
          Scattering Calculation
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

Atomic coordinates are converted into three-dimensional electron-density distributions using `xssdense-voxelize`.

The command supports two input formats:

- Ensembles of aligned PDB structures.
- Molecular-dynamics trajectories with an associated topology.

Electron densities are calculated using atomic form factors and Gaussian representations of the atomic electron distributions.

The resulting density maps are stored in an HDF5 file containing:

| Dataset  | Description |
| --- | --- |
| rho | Three-dimensional electron-density distributions. |
| dv | Voxel spacing in Å. |
| bbmax | Maximum dimension of the bounding box (Å). |

The default grid resolution is 52 × 52 × 52 voxels.

### 2. TFRecord Generation

The HDF5 density dataset is converted into TensorFlow TFRecords using `xssdense-gen-tfrecord`.

The script performs global min–max normalization:

$$\rho_{\mathrm{norm}} = \frac{\rho - \rho_{\min}}{\rho_{\max} - \rho_{\min}}$$

The dataset is divided into training and testing subsets. The splitting is performed in blocks. In molecular dynamics simulations this reduces correlations between neighbouring frames.

The output includes the TFRecord files and a metadata JSON file containing the normalization and electron-conservation information required by subsequent stages.

### 3. β-VAE Training

The normalized density maps are used to train a three-dimensional β-VAE using `xssdense-train-vae`.

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

The training objective combines square error loss and KL-divergence regularisation, where the β parameter controls the contribution of latent-space regularisation.

The implementation supports two β scheduling modes:

- constant: Fixed β throughout training.
- late: Gradual β warm-up during training.

Following training, the script automatically saves the encoder, decoder, training statistics, and latent representations of the training dataset.

### 4. Genetic-Algorithm Reconstruction

The reconstruction is performed using `xssdense-ga`. The algorithm searches the trained VAE latent space for density distributions that reproduce the experimentally measured difference-scattering signal.

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
          Scattering Calculation (FFT)
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

The genetic algorithm uses:

- Latent-space population initialization
- Tournament selection
- Crossover
- Gaussian mutation
- Elitism
- Introduction of new candidates
- Early stopping based on convergence

Candidate solutions are ranked according to their agreement with the experimental difference-scattering signal and the specified target excited-state yield.

## System Requirements

TensorFlow uses the GPU for VAE training and decoding. The scattering calculations in the genetic algorithm run on the CPU (NumPy/SciPy FFTs, multithreaded).

Note: TensorFlow should be installed with a version compatible with the local CUDA environment and GPU architecture.

### Python packages
| Component | Version |
| --- | --- |
| TensorFlow | 2.20 |
| Keras | 3.13.1 |
| NumPy | 1.26.4 |
| SciPy | 1.17 |
| Matplotlib | 3.10 |
| scikit-learn | 1.6.1 |
| Gemmi | 0.7.4 |
| tqdm | 4.67.1 |
| MDAnalysis | 2.10 |
| h5py | Required |
| pandas | Required |

## Installation

Clone the repository:

```bash
git clone https://github.com/Westenhoff-Lab/XSSDense.git
cd XSSDense
```

Install XSSDense and its Python dependencies:

```bash
pip install -e .
```

The installation provides four command-line entry points:

```bash
xssdense-voxelize
xssdense-gen-tfrecord
xssdense-train-vae
xssdense-ga
```


## Tutorial

The tutorial reproduces the LOV2 unfolding case from the accompanying manuscript. The example dataset is available on Zenodo (https://doi.org/10.5281/zenodo.21915223) and should be extracted to `example/`:

```
example/
├── LOV2_ref.dat
├── LOV2_diff_intermediate.dat
├── LOV2_diff_unfolded.dat
├── reference_results/
└── pdb_structures/
    ├── Ref_pred_aligned.sup.pdb
    └── ...
```

### Step 1: Voxelise PDB Structures

Convert the PDB ensemble into three-dimensional electron-density maps.

```bash
xssdense-voxelize \
    --pdb_path example/pdb_structures \
    --reference_pdb example/pdb_structures/Ref_pred_aligned.sup.pdb \
    --bbmax 61 \
    --save_path example_output \
    --system LOV2
```

| Parameter | Description | Default |
| --- | --- | --- |
| --pdb_path | Directory containing aligned PDB structures. | — |
| --reference_pdb | Reference structure defining the common coordinate frame. Required with `--pdb_path`. | — |
| --topology, --trajectory | MD topology and trajectory, used instead of `--pdb_path`. | — |
| --bbmax | Maximum dimension of the bounding box (Å). | Largest atom coordinate + 4 Å |
| --n_voxels | Number of voxels per dimension. Must be 52 unless the VAE architecture in `vae.py` is changed. | 52 |
| --cutoff | Atomic density calculation cutoff radius (Å). | 5.0 |
| --selection | MDAnalysis atom selection. | protein and not element H |
| --save_path | Output directory for the HDF5 file. | Required |
| --system | Identifier used for naming the output file. | Required |

The voxel spacing is 2bbmax / (n_voxels − 1), rounded up to 0.01 Å; here 2.40 Å. The reference structure is stored as frame 0, which is why `--dark_index 0` selects it as the ground state in Step 4.

**Expected output:**

```
example_output/
└── LOV2_61bb_2d4dv_5co.h5
```

The file is named `<system>_<bbmax>bb_<dv>dv_<cutoff>co.h5` and contains the density maps (`rho`), the voxel spacing (`dv`), the bounding box (`bbmax`) and the voxelisation settings.

For an MD trajectory, replace `--pdb_path` and `--reference_pdb` with `--topology topology.psf --trajectory trajectory.dcd`. Each frame becomes one density map.

### Step 2: Generate TFRecords

Normalise the densities and split them into training and test sets.

```bash
xssdense-gen-tfrecord \
    --h5 example_output/LOV2_61bb_2d4dv_5co.h5 \
    --output_dir example_output/LOV2_61bb_2d4dv_5co.tfrecord
```

| Parameter | Description | Default |
| --- | --- | --- |
| --h5 | Input HDF5 density dataset. | Required |
| --output_dir | Output directory for TFRecords and metadata. | Required |
| --block | Number of consecutive frames per block. | 100 |
| --test_every | Every nth block is assigned to testing. | 10 |
| --num_proc | Number of processes used for serialization. | 8 |

For MD trajectories, the block-based split keeps correlated neighbouring frames from ending up in both the training and test sets.

**Expected output:**

```
example_output/
└── LOV2_61bb_2d4dv_5co.tfrecord/
    ├── tfrecords/
    │   ├── train.tfrecords
    │   └── test.tfrecords
    └── meta/
        └── meta.json
```

`meta.json` holds the normalization parameters, the train/test frame ranges and the electron-conservation information used in Steps 3 and 4.

### Step 3: Train the β-VAE

```bash
xssdense-train-vae \
    --datapath example_output/LOV2_61bb_2d4dv_5co.tfrecord \
    --savepath example_output \
    --mode late \
    --beta_input 1 \
    --epochs 50 \
    --batchsize 64
```

| Parameter | Description | Default |
| --- | --- | --- |
| --datapath | Output directory of Step 2. Alternatively, give `--train_datapath`, `--test_datapath` and `--meta_datapath` separately. | — |
| --savepath | Directory where the `.vae` model directory is created. | . |
| --mode | β schedule: `constant`, or `late` (warm-up over the first 10% of training). | Required |
| --beta_input | Target β regularisation parameter. | Required |
| --z_dim | Latent-space dimensionality. | 8 |
| --learning_rate | Initial Adam learning rate. | 0.0003 |
| --epochs | Maximum number of training epochs. | 80 |
| --batchsize | Training batch size. | 64 |
| --patience | Early-stopping patience (epochs). | 5 |
| --min_improvement | Minimum relative validation-MSE improvement. | 0.005 |
| --save_every_epoch | Save model weights after every epoch. | Disabled |

**Expected output:**

```
example_output/
└── LOV2_61bb_2d4dv_5co.vae/
    ├── encoder_model.keras
    ├── decoder_model.keras
    ├── vae_best.weights.h5
    ├── latent_stats.txt
    ├── latents_train.npy
    └── log.txt
```

The `.vae` directory is named after `--datapath` and `log.txt` has the training and validation losses per epoch. `latents_train.npy` holds the encoded training set, and `latent_stats.txt` the mean, standard deviation, minimum and maximum of each latent dimension. Both are computed automatically after training and are used by the genetic algorithm.

### Step 4: Reconstruct Electron Density

Search the latent space for the density whose difference scattering matches the measured signal.

```bash
xssdense-ga \
    --model_path example_output/LOV2_61bb_2d4dv_5co.vae \
    --meta example_output/LOV2_61bb_2d4dv_5co.tfrecord/meta/meta.json \
    --h5 example_output/LOV2_61bb_2d4dv_5co.h5 \
    --dark_curve example/LOV2_ref.dat \
    --dI_curve example/LOV2_diff_unfolded.dat \
    --output_folder example_output/results/unfolded \
    --rho_bulk 0.334 \
    --target_yield 0.15 \
    --yield_weight 5 \
    --qmax 0.6 \
    --seed 1
```

Use `example/LOV2_diff_intermediate.dat` as `--dI_curve` (and a different `--output_folder`) to reconstruct the intermediate state. The manuscript results come from ten independent runs per state with `--seed 1` to `--seed 10`.

| Parameter | Description | Default |
| --- | --- | --- |
| --model_path | Trained VAE model directory. | Required |
| --meta | TFRecord metadata JSON file. | Required |
| --h5 | Original HDF5 density dataset. | Required |
| --dark_index | Index of the ground-state density in the HDF5 dataset. | 0 |
| --dark_curve | Experimental ground-state scattering curve. | Required |
| --dI_curve | Experimental difference-scattering curve. | Required |
| --output_folder | Reconstruction output directory. | Required |
| --rho_bulk | Bulk solvent electron density (e/Å^3); 0 for vacuum. | 0.334 |
| --threshold | Density threshold for solvent displacement. | rho_bulk |
| --target_yield | Expected excited-state population. | 0.5 |
| --yield_weight | Weight of the yield penalty. | 2.0 |
| --qmin | Minimum q-value used for fitting. | 0.0 |
| --qmax | Maximum q-value used for fitting. | 0.6 |
| --padding | FFT padding factor. | 3 |
| --pop_size | Genetic-algorithm population size. | 64 |
| --max_generations | Maximum number of generations. | 200 |
| --patience | Generations without improvement before stopping. | 25 |
| --elite_fraction | Fraction of elite candidates retained. | 0.1 |
| --newcomer_fraction | Fraction of new candidates introduced per generation. | 0.1 |
| --pool_size | Number of training latents in the initial candidate pool. | 1024 |
| --mutation_rate | Fraction of latent dimensions mutated. | 0.5 |
| --step_min | Minimum mutation step in latent standard deviations. | 0.03 |
| --step_max | Maximum mutation step in latent standard deviations. | 1.0 |
| --top_k | Number of best-ranked candidates saved. | 5 |
| --seed | Random-number generator seed. | 0 |

The reconstruction:

1. Loads the trained VAE and latent-space statistics.
2. Loads the ground-state density from the HDF5 dataset and passes it through the VAE.
3. Fits the ground-state scattering, including solvent and hydration-shell contributions.
4. Draws an initial population of candidate latents from the training latent distribution.
5. Decodes the candidates, calculates their scattering, and scores the difference curves against the data and the target yield.
6. Evolves the population by selection, crossover and mutation until `--max_generations` is reached or the best score has not improved for `--patience` generations.
7. Saves the highest-ranked density maps.

**Expected output:**

```
example_output/results/unfolded/
├── dark_state.ccp4
├── dark_state_vae.ccp4
├── dark_state_vae_density.npy
├── dark_latent.npy
├── rank0_latent.npy
├── rank0_density.npy
├── rank0_density.ccp4
├── rank0_deltaI.csv
├── ...
├── deltaI_fits.png
├── score_history.txt
└── summary.json
```

There is one set of `rank<N>_*` files per candidate (`--top_k`). The CCP4 maps can be viewed in ChimeraX. `summary.json` has the settings and, per candidate, the fitness score, estimated yield, R^2 of the fit, radius of gyration and distance from the dark-state latent.

## Using Your Own Data

To apply XSSDense to a new molecular system, prepare an ensemble of aligned PDB structures from an MD trajectory with its topology or there template for generating a structural ensemble with AFsample2 [run_afsample2.sh](run_afsample2.sh). Then follow the four-stage workflow.

*** Important Considerations ***

- All PDB structures must be aligned to a common reference frame.
- The grid must be 52 x 52 x 52 voxels unless the VAE architecture in `vae.py` is changed.
- The voxel spacing must be retained in the original HDF5 file.
- The original HDF5 dataset and corresponding metadata must be retained for reconstruction.
- The experimental scattering curves must be supplied as two-column CSV files without headers: q (Å^-1) and intensity.
- The ground-state curve and the difference-scattering curve are separate inputs to the reconstruction.
- The q-range used for fitting should be selected consistently with the experimental data.
- The target yield should reflect the expected excited-state population.

## Citation

If you use XSSDense in your research, please cite the accompanying manuscript:

```bibtex
@article{Monrroy2026.08.07.743437,
  author = {Monrroy, Leonardo and Cardoch, Sebastian and Lee, Seoyoung and Westenhoff, Sebastian},
  title = {XSSDense: X-ray Solution Scattering Density Reconstruction Using a Variational Autoencoder},
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
