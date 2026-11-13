#!/usr/bin/env python3
"""
Create synthetic SAXS inputs for the GA from voxel densities in an .h5 file.

Uses the same scattering model as the GA (padding, solute threshold, hydration
shell, q-binning, radial average), so a known "light" frame can be recovered.

Writes to --out_dir:
  ground_abs.csv     q, I_dark(q)                       -> --ground_scattering
  deltaI.csv         q, yield * (I_light - I_dark)(q)   -> --iq_path
  truth_light.npy    the light density used (raw units), to compare with the GA result
  truth.json         all settings, so the run can be reproduced

Usage:
  python make_synthetic_saxs.py --h5 data/1crn_41bb_1d61dv_5co.h5 --light_index 1234 \
      --voxel_size 0.5 --out_dir data/synthetic_1234
"""

import argparse
import json
import os

import cupy as cp
import h5py
import numpy as np
from cupyx.scipy import fft as cpx_fft
from cupyx.scipy.ndimage import distance_transform_edt

import map2iq_GPU as map2iq

# Must match the GA script
THRESHOLD = 0.4
PAD_FRAC = 1.5


def scattering_curve(rho, voxel_size, rho_bulk, drho, shell):
    """I(q) of one density with excluded solvent and hydration shell, as in the GA."""
    v = cp.asarray(rho, dtype=cp.float32)
    pad = int(PAD_FRAC * v.shape[0])
    v = cp.pad(v, ((pad, pad), (pad, pad), (pad, pad)))
    scale = voxel_size**3

    q_rad, idx, N_sum, valid = map2iq._prepare_q_binning(v.shape[0], voxel_size)

    solute = v > THRESHOLD
    dist = distance_transform_edt(~solute) * voxel_size
    shell_mask = ((dist <= shell) & ~solute).astype(cp.float32)

    A = (cpx_fft.fftn(v * scale)
         - rho_bulk * cpx_fft.fftn(solute.astype(cp.float32) * scale)
         + drho * cpx_fft.fftn(shell_mask * scale))
    return q_rad, map2iq._radial_average_power(A, idx, N_sum, valid)


def parse_arguments():
    p = argparse.ArgumentParser(description="Create synthetic SAXS inputs for the GA.")
    p.add_argument("--h5", required=True, help=".h5 with dataset 'rho' (raw, unnormalized densities)")
    p.add_argument("--dark_index", type=int, default=0, help="Ground-state frame (default 0, as the GA reads rho[0])")
    p.add_argument("--light_index", type=int, required=True, help="Frame used as the synthetic light state")
    p.add_argument("--voxel_size", type=float, required=True, help="Voxel spacing in Å (the dv in the .h5)")
    p.add_argument("--rho_bulk", type=float, default=0.334, help="Bulk solvent density in e/Å³ (default: 0.334)")
    p.add_argument("--drho_percent", type=int, default=5, help="Shell contrast in %% of rho_bulk, 0-12 (default: 5)")
    p.add_argument("--shell", type=float, default=3.0, help="Hydration shell thickness in Å, 1-5 (default: 3)")
    p.add_argument("--yield_", dest="yield_", type=float, default=0.5, help="Photoactivation yield (default: 0.5)")
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian noise on ΔI, as a fraction of max|ΔI| (default: 0)")
    p.add_argument("--qmax", type=float, default=0.6, help="Largest q written (default: 0.6, as in the GA)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_arguments()

    # The GA only scans these values, so the ground-state fit can only recover them exactly
    if not 0 <= args.drho_percent <= 12:
        raise ValueError("--drho_percent must be 0-12 (the GA's scan range)")
    if args.shell not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError("--shell must be 1, 2, 3, 4 or 5 Å (the GA's scan values)")

    os.makedirs(args.out_dir, exist_ok=True)
    drho = args.drho_percent * 0.01 * args.rho_bulk

    with h5py.File(args.h5, "r") as f:
        dark = f["rho"][args.dark_index].astype(np.float32)
        light = f["rho"][args.light_index].astype(np.float32)

    q, I_dark = scattering_curve(dark, args.voxel_size, args.rho_bulk, drho, args.shell)
    _, I_light = scattering_curve(light, args.voxel_size, args.rho_bulk, drho, args.shell)
    delta_I = args.yield_ * (I_light - I_dark)

    if args.noise > 0:
        rng = cp.random.default_rng(args.seed)
        sigma = args.noise * float(cp.max(cp.abs(delta_I)))
        delta_I = delta_I + rng.normal(0.0, sigma, delta_I.shape, dtype=cp.float32)

    sel = q <= args.qmax
    q, I_dark, delta_I = (cp.asnumpy(a[sel]) for a in (q, I_dark, delta_I))

    np.savetxt(os.path.join(args.out_dir, "ground_abs.csv"), np.column_stack((q, I_dark)), delimiter=",")
    np.savetxt(os.path.join(args.out_dir, "deltaI.csv"), np.column_stack((q, delta_I)), delimiter=",")
    np.save(os.path.join(args.out_dir, "truth_light.npy"), light)
    with open(os.path.join(args.out_dir, "truth.json"), "w") as f:
        json.dump({**vars(args), "drho": drho}, f, indent=2)

    print(f"Wrote {len(q)} q-points (q <= {args.qmax}) to {args.out_dir}")
    print(f"  dark frame {args.dark_index}, light frame {args.light_index}, yield {args.yield_}, "
          f"drho {drho:.4f} e/Å³, shell {args.shell} Å, noise {args.noise}")


if __name__ == "__main__":
    main()