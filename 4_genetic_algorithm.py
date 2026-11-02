"""
Genetic-algorithm reconstruction of a light-state density from a deltaI(q) curve
Usage:
    python 4_genetic_algorithm.py \
        --model_path data/1crn_41bb_1d61dv_5co.vae \
        --meta data/1crn_41bb_1d61dv_5co.tfrecord/meta/meta.json \
        --h5 data/1crn_41bb_1d61dv_5co.h5 \
        --dI_curve data/synthetic_1230_vacuum/deltaI.csv \
        --dark_curve data/synthetic_1230_vacuum/ground_abs.csv \
        --rho_bulk 0 \
        --target_yield 1 \
        --output_folder results/synthetic_1230_vacuum \
        --padding 3 \
        --qmax 0.6
"""

import os

from helper import redirect_tf_cpp_logs

redirect_tf_cpp_logs.redirect()  # before TensorFlow is imported

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)  # leave GPU memory for CuPy

import argparse
import json

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np

from density_manipulation import create_ccp4_map_
from helper.scattering import scattering_curve_gpu, envelope_voxels
from helper.VAE import load_vae


# =========================================================
# Curves and fitting
# =========================================================
def load_curve(path):
    """Two-column CSV (q, I) without header."""
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0], data[:, 1]


def fit_scale(I_model, I_data):
    """Least-squares factor k minimizing sum (I_data - k * I_model)^2."""
    return np.sum(I_data * I_model) / np.sum(I_model**2)


def r_squared(I_model, I_data):
    res = np.sum((I_data - I_model) ** 2)
    tot = np.sum((I_data - I_data.mean()) ** 2)
    return 1 - res / tot


def model_curve(density, dv, q_data, rho_bulk, drho, shell, padding, k):
    """Scattering curve of `density`, interpolated onto the data's q-values (NumPy)."""
    q, I = scattering_curve_gpu(density, dv, rho_bulk, drho, shell, padding, qmax=None, k=k)
    return np.interp(q_data, cp.asnumpy(q), cp.asnumpy(I))


def fit_dark_state(dark, dv, q_data, I_data, rho_bulk, padding, k, shells=(1.0, 2.0, 3.0, 4.0, 5.0), drho_percents=range(13)):
    """Scan shell thickness and contrast; for each, fit the absolute scale. Keep the best by r²."""
    if rho_bulk == 0:  # vacuum: no solvent parameters to fit
        shells, drho_percents = (0.0,), (0,)

    best = None
    for shell in shells:
        for pct in drho_percents:
            drho = pct * 0.01 * rho_bulk
            I_model = model_curve(dark, dv, q_data, rho_bulk, drho, shell, padding, k)
            scale = fit_scale(I_model, I_data)
            r2 = r_squared(scale * I_model, I_data)
            if best is None or r2 > best["r2"]:
                best = {"scale": scale, "drho": drho, "shell": shell, "r2": r2, "k": k}
    return best


# =========================================================
# Candidates
# =========================================================
def decode(vae, Z, xmin, xmax):
    """Latent vectors (n, latent_dim) -> densities in physical units, shape (n, 52, 52, 52)."""
    x = vae.decode(Z.astype(np.float32)).numpy()[..., 0]
    return np.clip(x * (xmax - xmin) + xmin, 0, None)


def candidate_dI(density, dark_fit, q_data, dv, rho_bulk, padding):
    """Model ΔI of one density for a yield of 1, using the dark state's solvent parameters."""
    I_light = model_curve(density, dv, q_data, rho_bulk, dark_fit["drho"], dark_fit["shell"], padding, dark_fit["k"])
    return dark_fit["scale"] * (I_light - dark_fit["I_dark"])


def score_candidate(density, dark_fit, q_data, dI_data, dv, rho_bulk, padding, target_yield, yield_weight, q_weight=True):
    """Fit one candidate to the measured ΔI. Returns (score, yield, r2)."""
    dI_model = candidate_dI(density, dark_fit, q_data, dv, rho_bulk, padding)
    if np.max(np.abs(dI_model)) < 1e-4 * np.max(np.abs(dark_fit["scale"] * dark_fit["I_dark"])):
        return -np.inf, 0.0, -np.inf  # no change from the dark state

    w = q_data if q_weight else 1.0  # weight higher q more
    yld = np.clip(fit_scale(w * dI_model, w * dI_data), 0.0, 1.0)
    r2 = r_squared(w * yld * dI_model, w * dI_data)
    score = r2 - yield_weight * (yld - target_yield) ** 2
    return score, yld, r2


def score_population(vae, Z, xmin, xmax, dark_fit, q_data, dI_data, dv, rho_bulk, padding, target_yield, yield_weight):
    """Decode and score all latent vectors in Z. Returns arrays (scores, yields, r2s)."""
    densities = decode(vae, Z, xmin, xmax)
    results = [score_candidate(d, dark_fit, q_data, dI_data, dv, rho_bulk, padding, target_yield, yield_weight) for d in densities]
    scores, yields, r2s = (np.array(v) for v in zip(*results))
    return scores, yields, r2s


# =========================================================
# Genetic algorithm
# =========================================================
def init_population(n, lower, upper, rng):
    """n random latent vectors, uniform within the training latents' range."""
    return rng.uniform(lower, upper, size=(n, len(lower)))


def select_parents(Z, scores, rng, k=3):
    """Tournament selection: for each slot, pick k random members and keep the best."""
    picks = rng.integers(0, len(Z), size=(len(Z), k))
    winners = picks[np.arange(len(Z)), np.argmax(scores[picks], axis=1)]
    return Z[winners]


def crossover(parents, rng):
    """Blend each parent with a random partner, per dimension."""
    partners = parents[rng.permutation(len(parents))]
    a = rng.uniform(size=parents.shape)
    return a * parents + (1 - a) * partners


def mutate(Z, step, rate, lower, upper, rng):
    """Add Gaussian noise of size `step` (per dimension) to a fraction `rate` of all entries."""
    mask = rng.random(Z.shape) < rate
    Z = Z + mask * rng.normal(0.0, 1.0, Z.shape) * step
    return np.clip(Z, lower, upper)


def run_ga(score_fn, lower, upper, std, pop_size, generations, n_elite, n_random, rate, seed):
    """Maximize score_fn over latent space. score_fn(Z) -> (scores, yields, r2s).

    Returns the final population sorted by score (best first), its scores, yields and r2s,
    and the best score per generation.
    """
    rng = np.random.default_rng(seed)
    Z = init_population(pop_size, lower, upper, rng)
    history = []

    for gen in range(generations):
        scores, yields, r2s = score_fn(Z)
        order = np.argsort(scores)[::-1]
        best = order[0]
        history.append(scores[best])
        print(f"Gen {gen:3d}  best {scores[best]:.4f}  r2 {r2s[best]:.4f}  yield {yields[best]:.3f}")

        if gen == generations - 1:
            break

        # mutation step shrinks from 1x to 0.1x the latent std over the run
        step = std * (1.0 - 0.9 * gen / max(generations - 1, 1))

        children = crossover(select_parents(Z, scores, rng), rng)
        children = mutate(children, step, rate, lower, upper, rng)
        children[:n_elite] = Z[order[:n_elite]]  # keep the best unchanged
        children[n_elite : n_elite + n_random] = init_population(n_random, lower, upper, rng)
        Z = children

    return Z[order], scores[order], yields[order], r2s[order], np.array(history)


def pick_distinct(Z, std, k, min_dist=0.1):
    """Indices of up to k candidates (Z sorted best first) at least `min_dist` latent std apart."""
    keep = []
    for i in range(len(Z)):
        if all(np.sqrt(np.mean(((Z[i] - Z[j]) / std) ** 2)) > min_dist for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return keep


# =========================================================
# Output
# =========================================================
def save_results(out, Z, scores, yields, r2s, densities, dark_fit, q_data, dI_data, dv, rho_bulk, padding):
    """Per candidate: latent, density, CCP4 map, ΔI curve. Plus one plot with all curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q_data, dI_data, "k.", label="data")

    for rank, (z, s, y, r2, d) in enumerate(zip(Z, scores, yields, r2s, densities)):
        tag = os.path.join(out, f"rank{rank}")
        np.save(f"{tag}_latent.npy", z)
        np.save(f"{tag}_density.npy", d)
        create_ccp4_map_(d, f"{tag}_density.ccp4", voxel_size=dv)

        dI_model = y * candidate_dI(d, dark_fit, q_data, dv, rho_bulk, padding)
        np.savetxt(f"{tag}_deltaI.csv", np.column_stack((q_data, dI_data, dI_model)), delimiter=",", header="q,data,model", comments="")
        ax.plot(q_data, dI_model, label=f"rank {rank}: score {s:.4f}, yield {y:.3f}")

    ax.set_xlabel("q (1/Å)")
    ax.set_ylabel("ΔI")
    ax.legend()
    fig.savefig(os.path.join(out, "deltaI_fits.png"), dpi=150)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="GA reconstruction of a light-state density from ΔI(q).")
    parser.add_argument("--model_path", required=True, help="Trained model directory (.vae), with latent_stats.txt")
    parser.add_argument("--meta", required=True, help="meta.json of the training data (normalization min/max)")
    parser.add_argument("--h5", required=True, help=".h5 with 'rho' (raw densities) and 'dv' (voxel size)")
    parser.add_argument("--dark_index", type=int, default=0, help="Frame used as the dark state (default: 0)")
    parser.add_argument("--dark_curve", required=True, help="Dark-state scattering, CSV (q, I)")
    parser.add_argument("--dI_curve", required=True, help="Measured ΔI, CSV (q, ΔI)")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--rho_bulk", type=float, default=0.334, help="Bulk solvent density in e/Å^3; 0 for vacuum (default: 0.334)")
    parser.add_argument("--target_yield", type=float, default=0.5, help="Expected photoactivation yield (default: 0.5)")
    parser.add_argument("--yield_weight", type=float, default=2.0, help="Weight of the yield penalty (default: 2)")
    parser.add_argument("--qmax", type=float, default=0.6, help="Largest q used in the fit (default: 0.6)")
    parser.add_argument("--padding", type=int, default=3, help="FFT padding factor, must match the data generation (default: 3)")
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--n_elite", type=int, default=5, help="Best candidates copied unchanged (default: 5)")
    parser.add_argument("--n_random", type=int, default=20, help="Random newcomers per generation (default: 20)")
    parser.add_argument("--mutation_rate", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=5, help="Number of distinct candidates to save (default: 5)")
    # no hydrogens -> 2.6 (1.21 Å^3/Da divided by 0.467 electrons/Da)
    # with hydrogens -> 2.26 (1.21 Å^3/Da divided by 0.535 electrons/Da)
    parser.add_argument("--volume_per_electron", type=float, default=2.6, help="Protein volume per electron in Å^3, sets the solvent envelope size (default: 2.6)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args


def main():
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    # Inputs
    with open(args.meta) as f:
        norm = json.load(f)["normalization"]
    xmin, xmax = float(norm["min"]), float(norm["max"])

    with h5py.File(args.h5, "r") as f:
        dark = f["rho"][args.dark_index].astype(np.float32)
        dv = float(f["dv"][()])

    n_electrons = float(dark.sum()) * dv**3
    k = envelope_voxels(n_electrons, dv, volume_per_electron=args.volume_per_electron)
    print(f"Envelope: {k} voxels ({k * dv**3:.0f} Å³) for {n_electrons:.0f} electrons")

    q_dark, I_dark_data = load_curve(args.dark_curve)
    q_d, dI_data = load_curve(args.dI_curve)
    sel = q_d <= args.qmax
    q_d, dI_data = q_d[sel], dI_data[sel]

    stats = np.loadtxt(os.path.join(args.model_path, "latent_stats.txt"))  # columns: mean std min max
    std, lower, upper = stats[:, 1], stats[:, 2], stats[:, 3]
    vae = load_vae(args.model_path)

    # 1. Dark state: fit on the dark curve, then evaluate on the ΔI q-grid
    dark_fit = fit_dark_state(dark, dv, q_dark, I_dark_data, args.rho_bulk, args.padding, k)
    dark_fit["I_dark"] = model_curve(dark, dv, q_d, args.rho_bulk, dark_fit["drho"], dark_fit["shell"], args.padding, k)
    print(f"Dark fit: scale {dark_fit['scale']:.4f}, drho {dark_fit['drho']:.4f}, " f"shell {dark_fit['shell']:.1f} Å, r2 {dark_fit['r2']:.5f}")
    create_ccp4_map_(dark, os.path.join(args.output_folder, "dark_state.ccp4"), voxel_size=dv)

    # 2. Genetic algorithm
    def score_fn(Z):
        return score_population(vae, Z, xmin, xmax, dark_fit, q_d, dI_data, dv, args.rho_bulk, args.padding, args.target_yield, args.yield_weight)

    Z, scores, yields, r2s, history = run_ga(score_fn, lower, upper, std, args.pop_size, args.generations, args.n_elite, args.n_random, args.mutation_rate, args.seed)

    # 3. Outputs: the best distinct candidates
    keep = pick_distinct(Z, std, args.top_k)
    densities = decode(vae, Z[keep], xmin, xmax)
    save_results(args.output_folder, Z[keep], scores[keep], yields[keep], r2s[keep], densities, dark_fit, q_d, dI_data, dv, args.rho_bulk, args.padding)

    np.savetxt(os.path.join(args.output_folder, "score_history.txt"), history, header="best score per generation")
    summary = {
        "settings": vars(args),
        "voxel_size": dv,
        "dark_fit": {k: float(v) for k, v in dark_fit.items() if k != "I_dark"},
        "candidates": [{"rank": r, "score": float(scores[i]), "yield": float(yields[i]), "r2": float(r2s[i])} for r, i in enumerate(keep)],
    }
    with open(os.path.join(args.output_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(keep)} candidates to {args.output_folder}")
    for c in summary["candidates"]:
        print(f"  rank {c['rank']}: score {c['score']:.4f}, yield {c['yield']:.3f}, r2 {c['r2']:.4f}")


if __name__ == "__main__":
    main()
