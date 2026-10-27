"""
From PDB ensemble or MD trajectory, computes the electron density on a 3D grid and save it to an HDF5 file
Usage:
    python 1_voxelize_pdb.py \
        --topology "data/1crn.psf" \
        --trajectory "data/1crn_eq_aligned.dcd" \
        --bbmax 41 \
        --save_path data \
        --system "1crn"
"""

import numpy
import MDAnalysis
import glob, re, os
import h5py
import argparse
from pathlib import Path


def natsort(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def sel_window(c, cutoff, origin, dv, npoints):
    lo = int(numpy.ceil((c - cutoff - origin) / dv))
    hi = int(numpy.floor((c + cutoff - origin) / dv)) + 1
    return max(lo, 0), min(hi, npoints)


def compute_rho(rs, element_parms, beta=0.0):
    a = element_parms["a"]
    b = element_parms["b"]
    rho = 0
    for ai, bi in zip(a, b):
        bi_modf = bi + beta
        prefactor = ai * ((numpy.pi * 4) / bi_modf) ** (3 / 2)
        exponent = -4 * numpy.pi**2 / bi_modf
        rho += prefactor * numpy.exp(exponent * rs**2)
    return rho


def load_sucoppens():
    data_file = Path(__file__).parent / "SuCoppens.dat"
    with open(data_file, "r") as f:
        content = [l for l in f.readlines() if l.strip() and not l.startswith("#")]
    asflib = {}
    for chunk in [content[i : i + 3] for i in range(0, len(content), 3)]:
        element = chunk[0].split()[1]
        a = [float(e) for e in chunk[1].split()]
        b = [float(e) for e in chunk[2].split()]
        asflib[element] = {"a": a, "b": b}
    return asflib


def estimate_beta(resolution):
    return 8 * numpy.pi**2 * (resolution / 2) ** 2


def compute_frame_density(elements, positions, l, dv, cutoff, asflib):
    true_atom_electrons = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
    beta = estimate_beta(dv)
    npoints = len(l)
    origin = l[0]
    rho = numpy.zeros((npoints, npoints, npoints), dtype=numpy.float32)
    for element, coord in zip(elements, positions):
        params = asflib[element]
        xmin, xmax = sel_window(coord[0], cutoff, origin, dv, npoints)
        ymin, ymax = sel_window(coord[1], cutoff, origin, dv, npoints)
        zmin, zmax = sel_window(coord[2], cutoff, origin, dv, npoints)
        dx = l[xmin:xmax] - coord[0]
        dy = l[ymin:ymax] - coord[1]
        dz = l[zmin:zmax] - coord[2]
        d = numpy.sqrt(dx[:, None, None] ** 2 + dy[None, :, None] ** 2 + dz[None, None, :] ** 2)
        atom_rho = compute_rho(d, params, beta)
        grid_atom_electrons = numpy.sum(atom_rho) * dv**3
        norm_factor = true_atom_electrons[element] / grid_atom_electrons
        rho[xmin:xmax, ymin:ymax, zmin:zmax] += norm_factor * atom_rho
    return rho


def create_universe(psf, dcd):
    print("creating universe...")
    u = MDAnalysis.Universe(psf, dcd, dt=0)
    u.guess_TopologyAttrs(context="default", to_guess=["elements"])
    print("done.")
    return u


def compute_rho_universe(elements, centered_positions, l, dv, cutoff):
    print("Computing density for each frame...")
    asflib = load_sucoppens()
    nframes = len(centered_positions)
    rho_universe = numpy.empty((nframes, len(l), len(l), len(l)), dtype=numpy.float32)
    for index, frame_positions in enumerate(centered_positions):
        print(f"{index + 1}/{nframes}", end="\r")
        rho_universe[index] = compute_frame_density(elements, frame_positions, l, dv, cutoff, asflib)
    print("done.")
    return rho_universe


def save_rho_universe(rho_universe, system, bbmax, dv, cutoff, save_path):
    print(f"Saving density to {save_path}...")
    bbmax_string = f"{bbmax:g}".replace(".", "d")
    dv_string = f"{dv:g}".replace(".", "d")
    cutoff_string = f"{cutoff:g}".replace(".", "d")
    outfile = os.path.join(save_path, f"{system}_{bbmax_string}bb_{dv_string}dv_{cutoff_string}co.h5")
    print(f"Saving density to {outfile}...")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("rho", data=rho_universe)
        f.create_dataset("dv", data=dv)
        f.create_dataset("bbmax", data=bbmax)
    print("done.")


def interpret_user_input(args, parser):
    if args.pdb_path and (args.topology or args.trajectory):
        parser.error("--pdb_path cannot be used together with " "--topology or --trajectory. " "Choose either a PDB ensemble or an MD trajectory.")
    if args.topology and args.trajectory:
        print("Using MD trajectory mode")
        psf = args.topology
        dcd = args.trajectory
        print(f"Using {psf} as reference PDB")
        print(f"Using {dcd} as trajectory")
    elif args.pdb_path:
        print("Using aligned PDB ensemble mode")
        models = sorted(glob.glob(os.path.join(args.pdb_path, "*.pdb")), key=natsort)
        # Remove reference structure if in the list
        models = [m for m in models if os.path.abspath(m) != os.path.abspath(args.reference_pdb)]
        if len(models) == 0:
            parser.error(f"No PDB files found in: {args.pdb_path}")
        # Put reference structure first
        dcd = [args.reference_pdb] + models
        psf = args.reference_pdb
        print(f"Using {psf} as reference PDB")
        print(f"Found {len(dcd)} PDB files")
    elif args.topology or args.trajectory:
        parser.error("--topology and --trajectory must be provided together.")
    else:
        parser.error("Either provide:\n" "  --topology AND --trajectory\n" "or\n" "  --pdb_path")
    return psf, dcd


def get_centered_positions(u, selection):
    """Return element symbols and per-frame centred coordinates, shape (nframes, natoms, 3)."""
    sel = u.select_atoms(selection)
    positions = numpy.array([sel.positions for _ in u.trajectory])
    centered = positions - positions.mean(axis=1, keepdims=True)
    return sel.elements, centered


def find_bbmax(centered_positions, margin=4):
    return numpy.ceil(numpy.abs(centered_positions).max()) + margin


def define_grid(bbmax, n_voxels):
    dv = numpy.ceil(2 * bbmax / (n_voxels - 1) * 100) / 100
    l = -bbmax + dv * numpy.arange(n_voxels)
    return l, dv


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbmax", help="Maximum dimension of the bounding box (Å)", default=None, type=float)
    parser.add_argument("--n_voxels", help="Number of voxels for density (must match autoencoder)", default=52, type=int)
    parser.add_argument("--cutoff", help="Cutoff radius around each atom (Å)", type=float, default=5.0)
    parser.add_argument("--pdb_path", help="Path to directory containing aligned PDBs", type=str)
    parser.add_argument("--trajectory", help="Path to MD trajectory file (e.g. DCD)", type=str)
    parser.add_argument("--reference_pdb", help="Reference structure (PDB) used as topology for PDB ensembles", type=str)
    parser.add_argument("--topology", help="Topology file for MD trajectory (PSF, GRO, PDB, etc.)", type=str)
    parser.add_argument("--save_path", help="Output HDF5 file", required=True, type=str)
    parser.add_argument("--system", help="Identifier for data set (PDB id or protein name for example)", required=True, type=str)
    parser.add_argument("--selection", help="Selection inside the PDB file", default="protein and not element H", type=str)
    args = parser.parse_args()
    psf, dcd = interpret_user_input(args, parser)
    return psf, dcd, args.selection, args.system, args.bbmax, args.n_voxels, args.cutoff, args.save_path


if __name__ == "__main__":
    psf, dcd, selection, system, bbmax, n_voxels, cutoff, save_path = parse_arguments()
    u = create_universe(psf, dcd)
    elements, centered = get_centered_positions(u, selection)
    if bbmax is None:
        bbmax = find_bbmax(centered)
    l, dv = define_grid(bbmax, n_voxels)
    print(f"bounding box: {bbmax} Å, num. of voxels: {n_voxels}, voxel size: {dv} Å")
    rho_universe = compute_rho_universe(elements, centered, l, dv, cutoff)
    save_rho_universe(rho_universe, system, bbmax, dv, cutoff, save_path)
