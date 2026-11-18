"""
Genetic-algorithm reconstruction of a light-state density from a deltaI(q) curve.
The search starts from, and draws newcomers from, a pool of training latents chosen to cover the
occupied latent space evenly so neither compact nor extended conformations
are favored. Each mutation draws its own step size, so local and large moves are always both tried.

Usage:
    python 4_genetic_algorithm.py \
        --model_path lov2/run_20261110_2037/lov2_61bb_2d4dv_5co.vae \
        --meta lov2/run_20261110_2037/lov2_61bb_2d4dv_5co.tfrecord/meta/meta.json \
        --h5 lov2/run_20261110_2037/lov2_61bb_2d4dv_5co.h5 \
        --dark_curve lov2/ground_abs.dat \
        --dI_curve lov2/deltaI_unfolded.dat \
        --output_folder lov2/run_20261110_2037/results/unfolded \
        --rho_bulk 0.334 --target_yield 0.15 --qmax 0.6
"""

import os

from xssdense import redirect_tf_cpp_logs

redirect_tf_cpp_logs.redirect()  # before TensorFlow is imported

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)
tf.config.experimental.enable_op_determinism()

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import h5py
import matplotlib.pyplot as plt
import numpy as np

from xssdense.density_manipulation import create_ccp4_map_
from xssdense import scattering
from xssdense.scattering import scattering_curve_gpu
from xssdense.vae import load_vae

N_CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()  # macOS has no sched_getaffinity
N_THREADS = max(1, N_CORES // 4)  # parallel candidates; each FFT gets N_CORES // N_THREADS threads
DECODE_BATCH = 64


# =========================================================
# Curves and fitting
# =========================================================
def load_curve(path):
    """Two-column CSV (q, I) without header."""
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0], data[:, 1]


def select_q(q, I, qmin, qmax):
    sel = (q >= qmin) & (q <= qmax)
    return q[sel], I[sel]


def fit_scale(I_model, I_data):
    """Least-squares factor minimizing sum (I_data - factor * I_model)^2."""
    return np.sum(I_data * I_model) / np.sum(I_model**2)


def r_squared(I_model, I_data):
    res = np.sum((I_data - I_model) ** 2)
    tot = np.sum((I_data - I_data.mean()) ** 2)
    return 1 - res / tot


def model_curve(density, dv, q_data, rho_bulk, drho, shell, padding, threshold):
    """Scattering curve of `density`, interpolated onto the data's q-values (NumPy)."""
    q, I = scattering_curve_gpu(density, dv, rho_bulk, drho, shell, padding, qmax=None, threshold=threshold)
    return np.interp(q_data, q, I)


def fit_dark_state(dark, dv, q_data, I_data, rho_bulk, padding, threshold):
    """Scan shell thickness (1-5 Å) and contrast (0-13% of bulk), as in the paper; fit the absolute scale for each. Keep the best by r²."""
    shells, drho_percents = ((0.0,), (0,)) if rho_bulk == 0 else ((1.0, 2.0, 3.0, 4.0, 5.0), range(0, 14))
    best = None
    for shell in shells:
        for pct in drho_percents:
            drho = pct * 0.01 * rho_bulk
            I_model = model_curve(dark, dv, q_data, rho_bulk, drho, shell, padding, threshold)
            scale = fit_scale(I_model, I_data)
            r2 = r_squared(scale * I_model, I_data)
            if best is None or r2 > best["r2"]:
                best = {"scale": scale, "drho": drho, "shell": shell, "r2": r2, "threshold": threshold}
    return best


# =========================================================
# Candidates
# =========================================================
def decode(vae, Z, xmin, xmax):
    """Latent vectors (n, latent_dim) -> densities in physical units, decoded DECODE_BATCH at a time."""
    Z = np.asarray(Z, dtype=np.float32)
    x = np.concatenate([vae.decode(Z[i : i + DECODE_BATCH]).numpy() for i in range(0, len(Z), DECODE_BATCH)])[..., 0]
    return np.clip(x * (xmax - xmin) + xmin, 0, None)


def encode_mean(vae, density, xmin, xmax):
    x = ((density - xmin) / (xmax - xmin))[None, ..., None].astype(np.float32)
    return vae.encode(x)[0].numpy()[0]


def radius_of_gyration(density, dv):
    coords = np.indices(density.shape).reshape(3, -1).T * dv
    w = density.ravel().astype(np.float64)
    center = w @ coords / w.sum()
    return float(np.sqrt(((coords - center) ** 2).sum(1) @ w / w.sum()))


def candidate_dI(density, dark_fit, q_data, dv, rho_bulk, padding):
    """Model ΔI of one density for a yield of 1; the light state uses the dark state's shell thickness and contrast."""
    I_light = model_curve(density, dv, q_data, rho_bulk, dark_fit["drho"], dark_fit["shell"], padding, dark_fit["threshold"])
    return dark_fit["scale"] * (I_light - dark_fit["I_dark"])


def score_candidate(density, dark_fit, q_data, dI_data, dv, rho_bulk, padding, target_yield, yield_weight):
    """q-weighted R² of the ΔI fit minus the yield penalty. Returns (score, yield, r2)."""
    w = q_data
    dI_model = candidate_dI(density, dark_fit, q_data, dv, rho_bulk, padding)
    if not np.any(dI_model):
        return -np.inf, 0.0, -np.inf  # identical to the dark state
    yld = np.clip(fit_scale(w * dI_model, w * dI_data), 0.0, 1.0)
    r2 = r_squared(w * yld * dI_model, w * dI_data)
    return r2 - yield_weight * (yld - target_yield) ** 2, yld, r2


def score_population(vae, Z, xmin, xmax, score_args):
    """Decode and score all latent vectors in Z, in parallel threads. Returns arrays (scores, yields, r2s)."""
    densities = decode(vae, Z, xmin, xmax)
    if N_THREADS == 1:
        results = [score_candidate(d, **score_args) for d in densities]
    else:
        with ThreadPoolExecutor(N_THREADS) as pool:
            results = list(pool.map(lambda d: score_candidate(d, **score_args), densities))
    scores, yields, r2s = (np.array(v) for v in zip(*results))
    return scores, yields, r2s


# =========================================================
# Search
# =========================================================
def diverse_pool(train_latents, std, n, rng):
    """n training latents that cover the occupied latent space evenly (farthest-point sampling, distances in latent std).
    Unlike random draws, this doesn't favor regions where many training frames lie."""
    X = train_latents / std
    idx = [int(rng.integers(len(X)))]
    dist = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(min(n, len(X)) - 1):
        i = int(np.argmax(dist))
        idx.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
    return train_latents[idx]


def select_parents(Z, scores, rng, k=3):
    """Tournament selection: for each slot, pick k random members and keep the best."""
    picks = rng.integers(0, len(Z), size=(len(Z), k))
    return Z[picks[np.arange(len(Z)), np.argmax(scores[picks], axis=1)]]


def crossover(parents, rng):
    """Blend each parent with a random partner, per dimension."""
    partners = parents[rng.permutation(len(parents))]
    a = rng.uniform(size=parents.shape)
    return a * parents + (1 - a) * partners


def mutate(Z, std, lower, upper, rate, step_min, step_max, rng):
    """Gaussian mutation of a fraction `rate` of the entries, with a step size drawn per child,
    log-uniformly between step_min and step_max (latent std): small local moves and large jumps alike."""
    steps = np.exp(rng.uniform(np.log(step_min), np.log(step_max), (len(Z), 1)))
    mask = rng.random(Z.shape) < rate
    return np.clip(Z + mask * rng.normal(0.0, 1.0, Z.shape) * steps * std, lower, upper)


def run_ga(score_fn, pool, lower, upper, std, args):
    """Maximize score_fn over latent space. Starts from, and draws newcomers from, the diverse pool;
    stops after `patience` generations without improvement. Returns the final population sorted by score
    (best first), its scores, yields and r2s, and the best score per generation."""
    rng = np.random.default_rng(args.seed)
    n_elite = max(1, int(args.elite_fraction * args.pop_size))
    n_new = int(args.newcomer_fraction * args.pop_size)
    Z = pool[rng.choice(len(pool), args.pop_size, replace=len(pool) < args.pop_size)]
    history, best_so_far, since_improvement = [], -np.inf, 0

    for gen in range(args.max_generations):
        scores, yields, r2s = score_fn(Z)
        order = np.argsort(scores)[::-1]
        best = order[0]
        history.append(scores[best])
        print(f"Gen {gen:3d}  best {scores[best]:.4f}  r2 {r2s[best]:.4f}  yield {yields[best]:.3f}")

        if scores[best] > best_so_far + 1e-4:
            best_so_far, since_improvement = scores[best], 0
        else:
            since_improvement += 1
        if since_improvement >= args.patience or gen == args.max_generations - 1:
            print(f"Stopped after {gen + 1} generations")
            break

        children = crossover(select_parents(Z, scores, rng), rng)
        children = mutate(children, std, lower, upper, args.mutation_rate, args.step_min, args.step_max, rng)
        children[:n_elite] = Z[order[:n_elite]]
        children[n_elite : n_elite + n_new] = pool[rng.integers(0, len(pool), n_new)]
        Z = children

    return Z[order], scores[order], yields[order], r2s[order], np.array(history)


# =========================================================
# Output
# =========================================================
def save_results(out, Z, densities, z_dark, std, dv, score_args):
    """Per candidate: latent, density, CCP4 map, ΔI curve. Plus one plot with all curves. Returns per-candidate info."""
    q_data, dI_data = score_args["q_data"], score_args["dI_data"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q_data, dI_data, "k.", label="data")
    info = []
    for rank, (z, d) in enumerate(zip(Z, densities)):
        s, y, r2 = score_candidate(d, **score_args)
        tag = os.path.join(out, f"rank{rank}")
        np.save(f"{tag}_latent.npy", z)
        np.save(f"{tag}_density.npy", d)
        create_ccp4_map_(d, f"{tag}_density.ccp4", voxel_size=dv)
        dI_model = y * candidate_dI(d, score_args["dark_fit"], q_data, dv, score_args["rho_bulk"], score_args["padding"])
        np.savetxt(f"{tag}_deltaI.csv", np.column_stack((q_data, dI_data, dI_model)), delimiter=",", header="q,data,model", comments="")
        ax.plot(q_data, dI_model, label=f"rank {rank}: score {s:.4f}, yield {y:.3f}")
        info.append({"rank": rank, "score": float(s), "yield": float(y), "r2": float(r2), "rg": radius_of_gyration(d, dv), "latent_dist_to_dark": float(np.sqrt(np.mean(((z - z_dark) / std) ** 2)))})
    ax.set_xlabel("q (1/Å)")
    ax.set_ylabel("ΔI")
    ax.legend()
    fig.savefig(os.path.join(out, "deltaI_fits.png"), dpi=150)
    plt.close(fig)
    return info


# =========================================================
# Main
# =========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="GA reconstruction of a light-state density from ΔI(q).")
    # Inputs
    parser.add_argument("--model_path", required=True, help="Trained model directory (.vae), with latent_stats.txt and latents_train.npy")
    parser.add_argument("--meta", required=True, help="meta.json of the training data (normalization min/max)")
    parser.add_argument("--h5", required=True, help=".h5 with 'rho' (raw densities) and 'dv' (voxel size)")
    parser.add_argument("--dark_index", type=int, default=0, help="Frame used as the dark state (default: 0)")
    parser.add_argument("--dark_curve", required=True, help="Dark-state scattering, CSV (q, I)")
    parser.add_argument("--dI_curve", required=True, help="Measured ΔI, CSV (q, ΔI)")
    parser.add_argument("--output_folder", required=True)
    # Physics and data
    parser.add_argument("--rho_bulk", type=float, default=0.334, help="Bulk solvent density in e/Å³; 0 for vacuum (default: 0.334)")
    parser.add_argument("--threshold", type=float, default=None, help="Density above which a voxel displaces bulk solvent, e/Å³ (default: rho_bulk)")
    parser.add_argument("--target_yield", type=float, default=0.5, help="Expected photoactivation yield (default: 0.5)")
    parser.add_argument("--yield_weight", type=float, default=2.0, help="Weight of the penalty on (yield - target)² (default: 2)")
    parser.add_argument("--qmin", type=float, default=0.0, help="Smallest q used in the fit (default: 0)")
    parser.add_argument("--qmax", type=float, default=0.6, help="Largest q used in the fit (default: 0.6)")
    parser.add_argument("--padding", type=int, default=3, help="FFT padding factor; for synthetic data it must match the data generation (default: 3)")
    # Search
    parser.add_argument("--pop_size", type=int, default=64, help="Population size (default: 64)")
    parser.add_argument("--max_generations", type=int, default=200, help="Upper limit on generations (default: 200)")
    parser.add_argument("--patience", type=int, default=25, help="Stop after this many generations without improvement (default: 25)")
    parser.add_argument("--elite_fraction", type=float, default=0.1, help="Fraction of best candidates copied unchanged (default: 0.1)")
    parser.add_argument("--newcomer_fraction", type=float, default=0.1, help="Fraction of each generation drawn fresh from the pool (default: 0.1)")
    parser.add_argument("--pool_size", type=int, default=1024, help="Training latents in the diverse pool (default: 1024)")
    parser.add_argument("--mutation_rate", type=float, default=0.5, help="Fraction of latent dimensions mutated per child (default: 0.5)")
    parser.add_argument("--step_min", type=float, default=0.03, help="Smallest mutation step, in latent std (default: 0.03)")
    parser.add_argument("--step_max", type=float, default=1.0, help="Largest mutation step, in latent std (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of best-scoring candidates to save (default: 5)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args


def main():
    args = parse_arguments()
    scattering.FFT_WORKERS = max(1, N_CORES // N_THREADS)
    threshold = args.rho_bulk if args.threshold is None else args.threshold
    print(f"{N_THREADS} candidate thread(s), {scattering.FFT_WORKERS} FFT thread(s) each; envelope: density > {threshold} e/Å³")
    os.makedirs(args.output_folder, exist_ok=True)

    with open(args.meta) as f:
        norm = json.load(f)["normalization"]
    xmin, xmax = float(norm["min"]), float(norm["max"])
    with h5py.File(args.h5, "r") as f:
        dark = f["rho"][args.dark_index].astype(np.float32)
        dv = float(f["dv"][()])

    q_dark, I_dark_data = select_q(*load_curve(args.dark_curve), args.qmin, args.qmax)
    q_d, dI_data = select_q(*load_curve(args.dI_curve), args.qmin, args.qmax)

    stats = np.loadtxt(os.path.join(args.model_path, "latent_stats.txt"))  # columns: mean std min max
    std, lower, upper = stats[:, 1], stats[:, 2], stats[:, 3]
    train_latents = np.load(os.path.join(args.model_path, "latents_train.npy"))
    vae = load_vae(args.model_path)

    # 1. Dark reference through the VAE, so the reconstruction error cancels in ΔI
    z_dark = encode_mean(vae, dark, xmin, xmax)
    dark_ref = decode(vae, z_dark[None], xmin, xmax)[0]
    dark_fit = fit_dark_state(dark_ref, dv, q_dark, I_dark_data, args.rho_bulk, args.padding, threshold)
    dark_fit["I_dark"] = model_curve(dark_ref, dv, q_d, args.rho_bulk, dark_fit["drho"], dark_fit["shell"], args.padding, threshold)
    print(f"Dark fit: scale {dark_fit['scale']:.3e}, drho {dark_fit['drho']:.4f}, shell {dark_fit['shell']:.1f} Å, r2 {dark_fit['r2']:.5f}, " f"Rg {radius_of_gyration(dark_ref, dv):.2f} Å")
    create_ccp4_map_(dark, os.path.join(args.output_folder, "dark_state.ccp4"), voxel_size=dv)
    create_ccp4_map_(dark_ref, os.path.join(args.output_folder, "dark_state_vae.ccp4"), voxel_size=dv)
    np.save(os.path.join(args.output_folder, "dark_state_vae_density.npy"), dark_ref)
    np.save(os.path.join(args.output_folder, "dark_latent.npy"), z_dark)

    # 2. Search
    score_args = {
        "dark_fit": dark_fit,
        "q_data": q_d,
        "dI_data": dI_data,
        "dv": dv,
        "rho_bulk": args.rho_bulk,
        "padding": args.padding,
        "target_yield": args.target_yield,
        "yield_weight": args.yield_weight,
    }

    def score_fn(Z):
        return score_population(vae, Z, xmin, xmax, score_args)

    pool = diverse_pool(train_latents, std, args.pool_size, np.random.default_rng(args.seed))
    Z, scores, yields, r2s, history = run_ga(score_fn, pool, lower, upper, std, args)

    # 3. Best candidates (Z is sorted best first)
    Z_out = Z[: args.top_k]

    densities = decode(vae, Z_out, xmin, xmax)
    candidates = save_results(args.output_folder, Z_out, densities, z_dark, std, dv, score_args)

    np.savetxt(os.path.join(args.output_folder, "score_history.txt"), history, header="best score per generation")
    summary = {
        "settings": vars(args),
        "voxel_size": dv,
        "dark_fit": {key: float(v) for key, v in dark_fit.items() if key != "I_dark"},
        "dark_rg": radius_of_gyration(dark_ref, dv),
        "candidates": candidates,
    }
    with open(os.path.join(args.output_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(candidates)} candidates to {args.output_folder}")
    for c in candidates:
        print(f"  rank {c['rank']}: score {c['score']:.4f}, yield {c['yield']:.3f}, r2 {c['r2']:.4f}, " f"Rg {c['rg']:.2f} Å, distance to dark {c['latent_dist_to_dark']:.2f} std")


if __name__ == "__main__":
    main()
