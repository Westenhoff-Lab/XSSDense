"""
Cluster per-frame scattering curves and write synthetic dark/light datasets per class
Usage:
    python crambin_saxs_class.py \
        --h5 data/1crn_41bb_1d61dv_5co.h5 \
        --out_dir data \
        --rho_bulk 0 \
        --padding 3
"""

import argparse
import json
import os

import h5py
import numpy
from matplotlib import pyplot
from sklearn import decomposition, cluster

from scattering import scattering_curve_gpu


def parse_args():
    p = argparse.ArgumentParser(description="XSS analysis: scattering curves, PCA/KMeans clustering and synthetic data.")
    p.add_argument("--h5", required=True, help="Input HDF5 file containing 'rho' and 'dv'.")
    p.add_argument("--out_dir", default="data", help="Output folder for figures, .h5 files and synthetic_class<N>_<vacuum|solvent>/ folders.")
    p.add_argument("--padding", type=int, default=3, help="FFT padding factor, must match the GA (default: 3)")
    p.add_argument("--n_pca", type=int, default=2, help="Number of PCA components used for clustering.")
    p.add_argument("--n_clusters", type=int, default=2, help="Number of KMeans clusters used for classification.")
    p.add_argument("--dark_index", type=int, default=0, help="Frame index used as the dark (ground) state.")
    p.add_argument("--rho_bulk", type=float, default=0.334, help="Bulk solvent density in e/Å³; 0 for vacuum (default: 0.334)")
    p.add_argument("--drho", type=float, default=None, help="Hydration shell contrast in e/Å³ (required if rho_bulk > 0)")
    p.add_argument("--shell", type=float, default=None, help="Hydration shell thickness in Å (required if rho_bulk > 0)")
    p.add_argument("--target_yield", dest="pa_yield", type=float, default=1.0, help="Photoactivation yield written to truth.json.")
    p.add_argument("--fig_dir", default=None, help="Folder for the plots (default: <out_dir>/figures)")
    args = p.parse_args()

    if args.rho_bulk == 0:
        if args.drho or args.shell:
            p.error("--drho and --shell have no effect in vacuum (--rho_bulk 0); leave them out")
        args.drho, args.shell = 0.0, 0.0
    elif args.drho is None or args.shell is None:
        p.error("--drho and --shell are required when --rho_bulk > 0")

    if args.fig_dir is None:
        args.fig_dir = os.path.join(args.out_dir, "figures")
    return args


# =========================================================
# I/O
# =========================================================
def load_density(path):
    with h5py.File(path, "r") as f:
        return f["rho"][()], float(f["dv"][()])


def save_h5(path, **datasets):
    with h5py.File(path, "w") as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)
    print("saved", path)


def save_fig(fig, out_dir, name):
    path = os.path.join(out_dir, name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    pyplot.close(fig)
    print("saved", path)


# =========================================================
# Scattering
# =========================================================
def scattering_curve(density, dv, args):
    """Full-q-range I(q) of one density with the solvent model, as NumPy arrays."""
    q, I = scattering_curve_gpu(density, dv, args.rho_bulk, args.drho, args.shell, args.padding, qmax=None)
    return q.get(), I.get()


def compute_scattering(rho, dv, args):
    qs, curves = None, []
    for i in range(rho.shape[0]):
        if (i + 1) % 50 == 0:
            print("frame %d/%d" % (i + 1, rho.shape[0]), end="\r")
        q, I = scattering_curve(rho[i], dv, args)
        if qs is None:
            qs = q
        curves.append(I)
    print()
    return qs, numpy.stack(curves, axis=0)


# =========================================================
# Clustering
# =========================================================
def normalise_curves(intensity):
    log_I = numpy.log10(intensity)
    return log_I - numpy.mean(log_I, axis=0)


def cluster_curves(curves_norm, n_pca, n_clusters):
    X = decomposition.PCA(n_components=n_pca).fit_transform(curves_norm)
    labels = cluster.KMeans(n_clusters, n_init=10, random_state=0).fit_predict(X)
    return X, labels


def class_mean_density(rho, labels, lid):
    return numpy.mean(rho[labels == lid], axis=0)


# =========================================================
# Plots
# =========================================================
def plot_normalised_curves(qs, curves_norm, out_dir):
    fig, ax = pyplot.subplots()
    ax.plot(qs, curves_norm.T)
    ax.set(ylabel="Log$_{10}$(I) - $\\langle$Log$_{10}$(I)$\\rangle_{t}$", xlabel="q (1/Å)")
    save_fig(fig, out_dir, "curves_norm")


def plot_clusters(X, labels, n_clusters, out_dir):
    fig, ax = pyplot.subplots()
    for lid in range(n_clusters):
        ax.plot(X[labels == lid, 0], X[labels == lid, 1], ".", label="class #%d" % (lid + 1))
    ax.set(xlabel="PC 1", ylabel="PC 2")
    ax.legend()
    save_fig(fig, out_dir, "pca_clusters")


def plot_class_intensities(qs, intensity, labels, n_clusters, out_dir):
    fig, ax = pyplot.subplots()
    for lid in range(n_clusters):
        mean_I = numpy.mean(intensity[labels == lid], axis=0)
        std_I = numpy.std(intensity[labels == lid], axis=0)
        ax.plot(qs, mean_I, label="class #%d" % (lid + 1))
        ax.fill_between(qs, mean_I - std_I, mean_I + std_I, alpha=0.5)
    ax.set(yscale="log", xlabel="q (1/Å)", ylabel="Intensity")
    ax.legend()
    save_fig(fig, out_dir, "class_intensity")


def plot_delta_curves(q, delta_curves, out_dir):
    fig, ax = pyplot.subplots()
    for lid, delta_I in enumerate(delta_curves):
        ax.plot(q, delta_I, label="class #%d" % (lid + 1))
    ax.set(xlabel="q (1/Å)", ylabel="$\\Delta$I")
    ax.legend()
    save_fig(fig, out_dir, "synthetic_deltaI")


# =========================================================
# Synthetic data
# =========================================================
def make_truth(dv, args):
    return {
        "pa_yield": args.pa_yield,
        "dark_index": args.dark_index,
        "rho_bulk": args.rho_bulk,
        "drho": args.drho,
        "shell": args.shell,
        "padding": args.padding,
        "voxel_size": dv,
    }


def save_synthetic(synth_dir, q, I_dark, delta_I, light, truth):
    os.makedirs(synth_dir, exist_ok=True)
    numpy.savetxt(os.path.join(synth_dir, "ground_abs.csv"), numpy.column_stack((q, I_dark)), delimiter=",")
    numpy.savetxt(os.path.join(synth_dir, "deltaI.csv"), numpy.column_stack((q, delta_I)), delimiter=",")
    numpy.save(os.path.join(synth_dir, "truth_light.npy"), light)
    with open(os.path.join(synth_dir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print("saved", synth_dir)


def make_synthetic_datasets(rho, dv, labels, args):
    q, I_dark = scattering_curve(rho[args.dark_index], dv, args)
    truth = make_truth(dv, args)
    tag = "vacuum" if args.rho_bulk == 0 else "solvent"

    delta_curves = []
    for lid in range(args.n_clusters):
        light = class_mean_density(rho, labels, lid)
        _, I_light = scattering_curve(light, dv, args)
        delta_I = I_light - I_dark
        delta_curves.append(delta_I)

        synth_dir = os.path.join(args.out_dir, "synthetic_class%d_%s" % (lid + 1, tag))
        save_synthetic(synth_dir, q, I_dark, delta_I, light, truth)

    plot_delta_curves(q, delta_curves, args.fig_dir)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.h5))[0]

    rho, dv = load_density(args.h5)

    qs, intensity = compute_scattering(rho, dv, args)
    save_h5(os.path.join(args.out_dir, stem + "_xss.h5"), q=qs, intensity=intensity)

    curves_norm = normalise_curves(intensity)
    X, labels = cluster_curves(curves_norm, args.n_pca, args.n_clusters)
    save_h5(os.path.join(args.out_dir, stem + "_xss_pca.h5"), pca_components=X, labels=labels)

    plot_normalised_curves(qs, curves_norm, args.fig_dir)
    plot_clusters(X, labels, args.n_clusters, args.fig_dir)
    plot_class_intensities(qs, intensity, labels, args.n_clusters, args.fig_dir)
    make_synthetic_datasets(rho, dv, labels, args)


if __name__ == "__main__":
    main()
