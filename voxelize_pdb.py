import numpy
import MDAnalysis
import glob, re, os
import h5py
import argparse

def natsort(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def sel_window(c, cutoff, bmax, dv, limit):
    min_idx = int(numpy.ceil((c - cutoff + bmax) / dv))
    max_idx = int(numpy.ceil((c + cutoff + bmax) / dv))
    return max(0, min_idx), min(max_idx, limit)


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
    with open("/home/monrroy/main/waxs-ml/data/SuCoppens.dat", "r") as f:
        content = [l for l in f.readlines() if l.strip() and not l.startswith("#")]
    asflib = {}
    for chunk in [content[i : i + 3] for i in numpy.arange(0, len(content), 3)]:
        element = chunk[0].split()[1]
        a = [float(e) for e in chunk[1].split()]
        b = [float(e) for e in chunk[2].split()]
        asflib[element] = {"a": a, "b": b}
    return asflib


def estimate_beta(resolution):
    return 8 * numpy.pi**2 * (resolution / 2) ** 2


def compute_frame_density(elements, positions, bbmax, dv, cutoff, asflib):
    true_atom_electrons = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
    beta = estimate_beta(dv)
    l = numpy.arange(-bbmax, bbmax + dv, dv)
    npoints = len(l)
    rho = numpy.zeros((npoints, npoints, npoints), dtype=numpy.float32)
    for element, coord in zip(elements, positions):
        params = asflib[element]
        xmin, xmax = sel_window(coord[0], cutoff, bbmax, dv, npoints)
        ymin, ymax = sel_window(coord[1], cutoff, bbmax, dv, npoints)
        zmin, zmax = sel_window(coord[2], cutoff, bbmax, dv, npoints)
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
    u_guesser = MDAnalysis.guesser.DefaultGuesser(u)
    guessed_elements = [u_guesser.guess_atom_element(name) for name in u.atoms.names]
    u.add_TopologyAttr("elements", guessed_elements)
    print("done.")
    return u


def compute_rho_universe(u, bbmax, dv, cutoff):
    asflib = load_sucoppens()
    rho_universe = []
    nframes = len(u.trajectory)
    sel = u.select_atoms("not element H")
    elements = sel.atoms.elements
    positions = numpy.array([sel.positions for _ in u.trajectory])
    means = numpy.mean(positions, axis=1, keepdims=True)
    centered_positions = positions - means
    for index in range(nframes):
        print("%d/%d" % (index + 1, nframes), end="\r")
        rho = compute_frame_density(elements, centered_positions[index], bbmax, dv, cutoff, asflib)
        rho_universe.append(rho)
    return rho_universe


def save_rho_universe(rho_universe, system, bbmax, dv, cutoff, save_path):
    print("saving density...")
    dv_string = str(dv).replace(".", "d")
    outfile = os.path.join(save_path,f"{system}_{bbmax}bb_{dv_string}dv_{cutoff}co.h5")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("rho", data=rho_universe)
    print("done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bbmax",
        help="Maximum dimension of the bounding box (Å)",
        required=True,
        type=float
    )

    parser.add_argument(
        "--dv",
        help="Grid spacing in Å",
        required=True,
        type=float
    )

    parser.add_argument(
        "--cutoff",
        help="Cutoff radius around each atom (Å)",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--pdb_path",
        help="Path to directory containing aligned PDBs",
        type=str
    )

    parser.add_argument(
        "--trajectory",
        help="Path to MD trajectory file (e.g. DCD)",
        type=str
    )

    parser.add_argument(
        "--reference_pdb",
        help="Reference structure (PDB) used as topology for PDB ensembles",
        type=str
    )

    parser.add_argument(
        "--topology",
        help="Topology file for MD trajectory (PSF, GRO, PDB, etc.)",
        type=str
    )

    parser.add_argument(
        "--save_path",
        help="Output HDF5 file",
        required=True,
        type=str
    )

    parser.add_argument(
        "--system",
        help="Identifier for data set (PDB id or protein name for example)",
        required=True,
        type=str
    )

    args = parser.parse_args()
    system = args.system
    bbmax = args.bbmax
    dv = args.dv
    cutoff = args.cutoff

    # ------------------------------------------------------------------
    # Validate input mode
    # ------------------------------------------------------------------

    if args.pdb_path and (args.topology or args.trajectory):
        parser.error(
            "--pdb_path cannot be used together with "
            "--topology or --trajectory. "
            "Choose either a PDB ensemble or an MD trajectory."
        )

    # MD trajectory mode
    if args.topology and args.trajectory:

        print("Using MD trajectory mode")

        psf = args.topology
        dcd = args.trajectory

    # PDB ensemble mode
    elif args.pdb_path:

        print("Using aligned PDB ensemble mode")
        models = sorted(glob.glob(os.path.join(args.pdb_path, "*.pdb")),key=natsort)        
        # Remove reference structure if it's already in the list
        models = [m for m in models if os.path.abspath(m) != os.path.abspath(args.reference_pdb)]
        # Put reference structure first
        dcd = [args.reference_pdb] + models
        if len(dcd) == 0:
            parser.error(
                f"No PDB files found in: {args.pdb_path}"
            )
        print(f"Found {len(dcd)} PDB files")
        psf = args.reference_pdb
        print(f"Using {psf} as reference pdb")

    # Incomplete MD specification
    elif args.topology or args.trajectory:

        parser.error(
            "--topology and --trajectory must be provided together."
        )

    # Nothing provided
    else:

        parser.error(
            "Either provide:\n"
            "  --topology AND --trajectory\n"
            "or\n"
            "  --pdb_path"
        )

    # ------------------------------------------------------------------
    # Compute densities
    # ------------------------------------------------------------------

    u = create_universe(psf, dcd)

    rho_universe = compute_rho_universe(
        u,
        bbmax,
        dv,
        cutoff
    )

    print(f"Saving density to {args.save_path}")
    save_rho_universe(rho_universe,system,bbmax,dv,cutoff,args.save_path)
