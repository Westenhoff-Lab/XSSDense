# -*- coding: utf-8 -*-
"""
map2iq_cupy.py - GPU-only version of map2iq using CuPy / cupyx (no CPU numpy/scipy).
All inputs and outputs are CuPy arrays (cupy.ndarray). No plotting.
"""

import cupy as cp
from cupyx.scipy import ndimage as cpx_ndimage
from cupyx.scipy import fft as cpx_fft
from cupyx.scipy import interpolate as cpx_interp  # cupy has cp.interp but keep names consistent
from cupyx.scipy.ndimage import binary_dilation
from cupyx.scipy.ndimage import distance_transform_edt




def r2_score_cp(y_true: cp.ndarray, y_pred: cp.ndarray) -> cp.float64:
    """
    Calcualte R² score between true and predicted values.
    """
    y_true = y_true.astype(cp.float64)
    y_pred = y_pred.astype(cp.float64)
    ss_res = cp.sum((y_true - y_pred) ** 2)
    ss_tot = cp.sum((y_true - cp.mean(y_true)) ** 2)
    if ss_tot == 0:
        # constant target
        return cp.float64(1.0) if bool(cp.allclose(y_true, y_pred)) else cp.float64(0.0)
    return cp.float64(1.0) - (ss_res / ss_tot)

def best_scaling_factor(I_model: cp.ndarray, target: cp.ndarray,target_yield, w) -> cp.float64:
    """
    Calculate best scaling factor based on I_model and target.
    """
    I_model = I_model.astype(cp.float64)
    target = target.astype(cp.float64)
    denom = cp.dot(I_model, I_model)
    if denom == 0:
        return cp.float64(1.0)
    alpha = cp.dot(target, I_model) / denom
    if alpha < 0:
        alpha = cp.float64(1.0)
    alpha_score = w * (alpha-target_yield) **2
    return cp.float64(alpha), alpha_score


def build_excess_density(
        rho_voxel: cp.ndarray,  # electron density [e/Å^3]
        voxel_size: float,  # Å
        rho_bulk: float = 0.334,  # e/Å^3 (bulk water)
        delta_rho_hyd: float = 0.03,  # hydration contrast (fit this)
        shell_thickness: float = 3.0,  # Å
        threshold: float = 0.01,  # solute mask threshold
        sV: float = 1.0  # excluded-volume scaling (fit this)
):
    """
    Construct EXCESS electron density:

        rho_excess = (rho_protein - sV*rho_bulk)*mask_solute
                      + delta_rho_hyd * mask_shell

    Using a SASA-like hydration shell:
        - Euclidean distance outside solute gives shell geometry
        - Hydration *contrast* only (NOT water density)
    """

    rho = rho_voxel.astype(cp.float64)

    # 1) Protein interior (solute region)
    solute = (rho > float(threshold))

    # 2) Distance outside solute (Euclidean)
    #    dist_outside = 0 at surface, increases outward
    dist_out = distance_transform_edt(~solute) * voxel_size

    # 3) Hydration shell mask = distance ≤ shell_thickness but NOT inside solute
    shell = (dist_out <= shell_thickness) & (~solute)

    # 4) Build EXCESS-density map
    rho_excess = rho.copy()

    # subtract displaced bulk solvent inside protein
    rho_excess = rho_excess - (sV * rho_bulk) * solute.astype(cp.float64)

    # add SMALL hydration contrast (not water density!)
    rho_excess = rho_excess + delta_rho_hyd * shell.astype(cp.float64)

    return rho_excess


def fft_intensity_excess(
        rho_voxel: cp.ndarray,
        voxel_size: float,
        rho_bulk: float = 0.334,
        delta_rho_hyd: float = 0.03,
        shell_thickness: float = 3.0,
        threshold: float = 0.01,
        sV: float = 1.0,
        sigma_q: float = 10,
        dq: float = 0.001
):
    """
    SAXS from EXCESS density using FFT:
        1) Build SASA-based hydration-shell contrast map
        2) FFT -> |F|^2
        3) Radial binning
    """

    # Build excess electron density (protein - water + hydration contrast)
    rho_excess = build_excess_density(
        rho_voxel,
        voxel_size,
        rho_bulk=rho_bulk,
        delta_rho_hyd=delta_rho_hyd,
        shell_thickness=shell_thickness,
        threshold=threshold,
        sV=sV
    )

    # Scale by voxel volume (convert to amplitude density)
    rho_scaled = rho_excess * (voxel_size ** 3)

    # 1) FFT
    Fq = cpx_fft.fftn(rho_scaled)

    # 2) Build q-grid
    N = rho_voxel.shape[0]
    freqs = cp.fft.fftfreq(N, d=float(voxel_size))  # cycles/Å
    kx, ky, kz = cp.meshgrid(freqs, freqs, freqs, indexing="ij")
    qgrid = 2.0 * cp.pi * cp.sqrt(kx * kx + ky * ky + kz * kz)

    # 3) Low-pass filter (optional)
    lowpass = cp.exp(-(qgrid ** 2) / (2.0 * sigma_q ** 2))

    # 4) Intensity
    I3d = (cp.abs(Fq) *lowpass )** 2

    # 5) Radial binning
    q_flat = qgrid.ravel()
    I_flat = I3d.ravel()

    qmax = float(q_flat.max())
    nbins = max(1, int(cp.ceil(qmax / dq).item()))
    bins = cp.linspace(0.0, qmax, nbins + 1, dtype=cp.float64)

    idx = cp.clip(cp.digitize(q_flat, bins) - 1, 0, nbins - 1).astype(cp.int64)

    I_sum = cp.bincount(idx, weights=I_flat, minlength=nbins)
    N_sum = cp.bincount(idx, minlength=nbins)
    valid = N_sum > 0

    q_mid = 0.5 * (bins[1:] + bins[:-1])
    q_rad = q_mid[valid]
    I_rad = I_sum[valid] / N_sum[valid]

    return q_rad.astype(cp.float64), I_rad.astype(cp.float64)

def _fftn_rho_scaled(vol: cp.ndarray, voxel_size: float) -> cp.ndarray:
    """FFT of electron density scaled to amplitude density units."""
    return cpx_fft.fftn(vol.astype(cp.float64) * (float(voxel_size) ** 3))

def _masks_ffts_from_rho(rho: cp.ndarray,
                         voxel_size: float,
                         threshold: float,
                         shell_thickness: float) -> tuple[cp.ndarray, cp.ndarray]:
    """
    From electron density 'rho' (padded to Z^3), build χ_protein and χ_shell at this threshold
    and return their FFTs (Fe, Fs) scaled consistently to amplitude density units.
    """
    solute = (rho > float(threshold))
    dist_out = distance_transform_edt(~solute) * float(voxel_size)  # Å
    shell = (dist_out <= float(shell_thickness)) & (~solute)
    scale = (float(voxel_size) ** 3)
    Fe = cpx_fft.fftn(solute.astype(cp.float64) * scale)  # FFT(χ_protein)*scale
    Fs = cpx_fft.fftn(shell.astype(cp.float64)  * scale)  # FFT(χ_shell)*scale
    return Fe, Fs

def _prepare_q_binning(Z: int, voxel_size: float, dq=0.001):
    """Precompute q-grid and radial-binning index once."""
    freqs = cp.fft.fftfreq(Z, d=float(voxel_size))  # cycles/Å
    kx, ky, kz = cp.meshgrid(freqs, freqs, freqs, indexing="ij")
    qgrid = 2.0 * cp.pi * cp.sqrt(kx*kx + ky*ky + kz*kz)

    q_flat = qgrid.ravel()
    qmax = float(q_flat.max())
    nbins = max(1, int(cp.ceil(qmax / float(dq)).item()))
    bins = cp.linspace(0.0, qmax, nbins + 1, dtype=cp.float64)

    idx = cp.clip(cp.digitize(q_flat, bins) - 1, 0, nbins - 1).astype(cp.int64)
    N_sum = cp.bincount(idx, minlength=nbins)
    valid = (N_sum > 0)
    q_mid = 0.5 * (bins[1:] + bins[:-1])
    q_rad = q_mid[valid]
    return q_rad.astype(cp.float64), idx, N_sum, valid

def _radial_average_power(A: cp.ndarray,
                          idx: cp.ndarray,
                          N_sum: cp.ndarray,
                          valid: cp.ndarray) -> cp.ndarray:
    I3d = cp.abs(A) ** 2
    I_flat = I3d.ravel()
    I_sum = cp.bincount(idx, weights=I_flat, minlength=N_sum.size)
    I_rad = I_sum[valid] / N_sum[valid]
    return I_rad.astype(cp.float64)

# --- Helpers for batched 1D building blocks ---------------------------------

def _radial_average_scalar(S3: cp.ndarray, idx: cp.ndarray, N_sum: cp.ndarray, valid: cp.ndarray) -> cp.ndarray:
    """Radial average of a real 3D scalar field already on the FFT grid."""
    S_flat = S3.ravel()
    S_sum  = cp.bincount(idx, weights=S_flat, minlength=N_sum.size)
    return (S_sum[valid] / N_sum[valid]).astype(cp.float64)

def _radial_cross_real(A: cp.ndarray, B: cp.ndarray,
                       idx: cp.ndarray, N_sum: cp.ndarray, valid: cp.ndarray) -> cp.ndarray:
    """Radial average of Re(A * conj(B)) on the FFT grid."""
    X = cp.real(A * cp.conj(B))
    return _radial_average_scalar(X, idx, N_sum, valid)


def batched_bincount(idx, weights, N_sum, valid, B):
    """
    Batched bincount for B volumes.
    idx: (Z³,)
    weights: (B, Z³)
    Returns (B, Nq_valid)
    """
    idx = idx.astype(cp.int64)
    Nq = N_sum.size

    offsets = cp.arange(B, dtype=cp.int64)[:, None] * Nq
    idx2 = idx[None, :] + offsets  # (B, Z³)

    bc = cp.bincount(
        idx2.ravel(),
        weights=weights.ravel(),
        minlength=B * Nq
    )

    bc = bc.reshape(B, Nq)
    return bc[:, valid] / N_sum[valid]


def build_shell_dilation(solute_4d, radius_voxels):
    """
    solute_4d: (Bb, Z, Z, Z) boolean
    radius_voxels: int
    Returns: (Bb, Z, Z, Z)
    CuPy only supports iterations=1 in 3D binary_dilation.
    So we implement multi-iteration dilation manually.
    """

    Bb = solute_4d.shape[0]

    if radius_voxels <= 0:
        return cp.zeros_like(solute_4d, dtype=cp.bool_)

    struct = cp.ones((3,3,3), dtype=cp.bool_)
    shells = []

    for i in range(Bb):
        sol = solute_4d[i].copy()

        # repeated 1-step dilations
        dil = sol
        for _ in range(radius_voxels):
            dil = binary_dilation(dil, structure=struct, iterations=1)

        shell = cp.logical_and(dil, ~sol)
        shells.append(shell)

    return cp.stack(shells, axis=0)



def run_batched(
        voxels_batch: cp.ndarray,
        iq_file,
        ground_state_voxel: cp.ndarray,
        absolute_scale: float,
        drho: float,
        dark_shell: float,
        voxel_size: float,
        target_yield: float,
        yield_weight: float,
        dq: float = 0.001,
        rho_bulk: float = 0.334,
        inner_batch: int = 4,
):


    # --------------------------
    # Constants
    # --------------------------
    threshold = 0.4
    pad_frac = 3
    scale = voxel_size**3
    sV = 1.0
    a = rho_bulk * sV

    # shell scan for LIGHT state
    shell_vals = cp.asarray([1., 2., 3., 4., 5.], dtype=cp.float64)

    # drho % scan for LIGHT state (0..12%)
    delta_vals = (cp.arange(0, 13) * 0.01 * rho_bulk).astype(cp.float64)

    # --------------------------
    # Experimental data
    # --------------------------
    q_all = cp.asarray(iq_file.q, cp.float64)
    sel = q_all <= 0.6
    exp_q = q_all[sel]
    exp_I = cp.asarray(iq_file.i, cp.float64)[sel]

    y_mean = exp_I.mean()
    ss_tot = cp.sum((exp_I - y_mean)**2)
    if float(ss_tot) == 0:
        ss_tot = cp.float64(1.0)

    # --------------------------
    # DARK STATE (single drho, single shell)
    # --------------------------
    G = ground_state_voxel.shape[0]
    pad_dark = int(pad_frac * G)

    dark = cp.pad(
        cp.asarray(ground_state_voxel, cp.float32),
        ((pad_dark,pad_dark),(pad_dark,pad_dark),(pad_dark,pad_dark))
    )

    Fp_dark = cpx_fft.fftn(dark * scale)

    Z = dark.shape[0]
    q_rad, idx, N_sum, valid = _prepare_q_binning(Z, voxel_size, dq)

    solute_dark = (dark > threshold)
    dist_dark = distance_transform_edt(~solute_dark) * float(voxel_size)

    shell_dark = ((dist_dark <= float(dark_shell)) &
                  (~solute_dark)).astype(cp.float32)

    Fe_dark = cpx_fft.fftn(solute_dark.astype(cp.float32) * scale)
    Fs_dark = cpx_fft.fftn(shell_dark * scale)

    A_dark = Fp_dark - rho_bulk*Fe_dark + drho*Fs_dark
    I_dark = _radial_average_power(A_dark, idx, N_sum, valid)

    # Interpolation preparation
    sel2 = exp_q <= q_rad[-1]
    exp_q2 = exp_q[sel2]
    exp_I2 = exp_I[sel2]

    left = cp.searchsorted(q_rad, exp_q2, side='right') - 1
    left = cp.clip(left, 0, q_rad.size - 1)
    right = cp.clip(left+1, 0, q_rad.size - 1)

    qL = q_rad[left]
    qR = q_rad[right]
    w = cp.where(qR > qL, (exp_q2 - qL)/(qR - qL), 0.0)

    # --------------------------
    # Output arrays
    # --------------------------
    B, D, _, _ = voxels_batch.shape
    pad = int(pad_frac * D)

    scores     = cp.full((B,), -cp.inf, dtype=cp.float32)
    best_delta = cp.zeros(B, dtype=cp.float32)
    best_shell = cp.zeros(B, dtype=cp.float32)
    best_alpha = cp.zeros(B, dtype=cp.float32)
    best_ascore= cp.zeros(B, dtype=cp.float32)

    # --------------------------
    # MAIN BATCH LOOP
    # --------------------------
    for b_start in range(0, B, inner_batch):
        b_end = min(B, b_start + inner_batch)
        Bb = b_end - b_start

        # Pad light voxel batch (in-place)
        vol_b = cp.pad(
            voxels_batch[b_start:b_end].astype(cp.float32),
            ((0,0),(pad,pad),(pad,pad),(pad,pad))
        )

        # FFT(Fp)
        Fp_all = cpx_fft.fftn(vol_b * scale, axes=(1,2,3))
        Ppp = batched_bincount(
            idx,
            (cp.abs(Fp_all)**2).reshape(Bb,-1),
            N_sum, valid, Bb
        )

        solute = (vol_b > threshold)

        # compute EDT for each structure (light) in place
        dist_out = cp.empty_like(vol_b, dtype=cp.float32)
        for i in range(Bb):
            dist_out[i] = distance_transform_edt(~solute[i]).astype(cp.float32) * float(voxel_size)

        # FFT(Fe)
        Fe = cpx_fft.fftn(solute.astype(cp.float32) * scale,
                          axes=(1,2,3))

        Pee = batched_bincount(
            idx,
            (cp.abs(Fe)**2).reshape(Bb,-1),
            N_sum, valid, Bb
        )

        Cpe = batched_bincount(
            idx,
            cp.real(Fp_all * cp.conj(Fe)).reshape(Bb,-1),
            N_sum, valid, Bb
        )

        # =========================
        # SHELL-BY-SHELL FFT
        # =========================
        for s_th in shell_vals:

            shell_b = ((dist_out <= s_th) &
                       (~solute)).astype(cp.float32)

            Fs = cpx_fft.fftn(shell_b * scale, axes=(1,2,3))

            Pss = batched_bincount(
                idx,
                (cp.abs(Fs)**2).reshape(Bb,-1),
                N_sum, valid, Bb
            )

            Cps = batched_bincount(
                idx,
                cp.real(Fp_all * cp.conj(Fs)).reshape(Bb,-1),
                N_sum, valid, Bb
            )

            Ces = batched_bincount(
                idx,
                cp.real(Fe * cp.conj(Fs)).reshape(Bb,-1),
                N_sum, valid, Bb
            )

            # -------------------------
            # DRHO LOOP (0–12%)
            # -------------------------
            for b_val in delta_vals:

                Iq = (Ppp
                      + (a*a)*Pee
                      + (b_val*b_val)*Pss
                      - 2*a*Cpe
                      + 2*b_val*Cps
                      - 2*(a*b_val)*Ces)

                Delta_q = (Iq - I_dark[None,:]) * absolute_scale

                D_left  = Delta_q[:, left]
                D_right = Delta_q[:, right]
                Dexp    = (1-w)*D_left + w*D_right

                den = cp.sum(Dexp*Dexp, axis=-1)
                num = cp.sum(exp_I2 * Dexp, axis=-1)

                alpha = cp.where(den > 0, num/den, 1.0)
                alpha = cp.where(alpha < 0, 1.0, alpha)

                penalty = yield_weight*(alpha - target_yield)**2
                res2    = cp.sum((exp_I2 - alpha[:,None]*Dexp)**2, axis=-1)
                r2      = 1.0 - res2/ss_tot

                score = r2 - penalty

                improved = score > scores[b_start:b_end]
                idx_imp = cp.where(improved)[0]

                for gi in idx_imp:
                    gi_abs = b_start + gi
                    scores[gi_abs]      = float(score[gi])
                    best_delta[gi_abs]  = float(b_val)
                    best_shell[gi_abs]  = float(s_th)
                    best_alpha[gi_abs]  = float(alpha[gi])
                    best_ascore[gi_abs] = float(penalty)

            # free shell FFTs
            del Fs, Pss, Cps, Ces
            cp._default_memory_pool.free_all_blocks()

        # free batch-level FFTs
        del vol_b, Fp_all, Fe, Pee, Cpe, dist_out
        cp._default_memory_pool.free_all_blocks()

    return (
        scores,
        cp.full_like(best_delta, 1.0),  # sV=1 fixed
        best_alpha,
        best_ascore,
        best_delta,
        cp.full_like(best_delta, threshold),
        best_shell,
    )


def run_batched_memory_safe(
    voxels_batch: cp.ndarray,
    iq_file,
    ground_state_voxel: cp.ndarray,
    absolute_scale: float,
    drho: float,
    dark_shell: float,
    voxel_size: float,
    target_yield: float,
    yield_weight: float,
    dq: float = 0.001,
    rho_bulk: float = 0.334,
    inner_batch: int = 2,   # reduce batch to save GPU memory
    shell_vals=cp.asarray([1.,2.,3.,4.,5.], cp.float32),
    delta_vals=None
):
    """
    Memory-safe batched scoring:
      - Pre-pad once for all
      - Reuse FFT buffers
      - Compute ΔI only for top voxels later
    """
    threshold = 0.4
    pad_frac = 1  # less padding
    scale = voxel_size**3
    sV = 1.0
    a = rho_bulk * sV

    if delta_vals is None:
        delta_vals = cp.arange(0, 13) * 0.01 * rho_bulk

    # --------------------------
    # Experimental data
    # --------------------------
    q_all = cp.asarray(iq_file.q, cp.float32)
    sel = q_all <= 0.6
    exp_q = q_all[sel]
    exp_I = cp.asarray(iq_file.i, cp.float32)[sel]

    y_mean = exp_I.mean()
    ss_tot = cp.sum((exp_I - y_mean)**2)
    if ss_tot == 0: ss_tot = cp.float32(1.0)

    # --------------------------
    # Dark state FFT (prepad)
    # --------------------------
    G = ground_state_voxel.shape[0]
    pad_dark = int(pad_frac * G)
    dark = cp.pad(ground_state_voxel.astype(cp.float32),
                  ((pad_dark,pad_dark),(pad_dark,pad_dark),(pad_dark,pad_dark)))

    Fp_dark = cpx_fft.fftn(dark * scale)

    Z = dark.shape[0]
    q_rad, idx, N_sum, valid = map2iq._prepare_q_binning(Z, voxel_size, dq)

    solute_dark = (dark > threshold)
    dist_dark = distance_transform_edt(~solute_dark) * float(voxel_size)
    shell_dark_mask = ((dist_dark <= float(dark_shell)) & (~solute_dark)).astype(cp.float32)
    Fe_dark = cpx_fft.fftn(solute_dark.astype(cp.float32)*scale)
    Fs_dark = cpx_fft.fftn(shell_dark_mask * scale)
    A_dark = Fp_dark - rho_bulk*Fe_dark + drho*Fs_dark
    I_dark = map2iq._radial_average_power(A_dark, idx, N_sum, valid)

    # --------------------------
    # Output arrays
    # --------------------------
    B, D, _, _ = voxels_batch.shape
    pad = int(pad_frac * D)

    scores     = cp.full((B,), -cp.inf, dtype=cp.float32)
    best_delta = cp.zeros(B, dtype=cp.float32)
    best_shell = cp.zeros(B, dtype=cp.float32)
    best_alpha = cp.zeros(B, dtype=cp.float32)
    best_ascore= cp.zeros(B, dtype=cp.float32)

    # Preallocate reusable buffers
    dist_out = cp.empty((inner_batch, Z, Z, Z), dtype=cp.float32)
    solute = cp.empty_like(dist_out)
    vol_b = cp.empty((inner_batch, Z, Z, Z), dtype=cp.float32)

    # --------------------------
    # MAIN BATCH LOOP
    # --------------------------
    for b_start in range(0, B, inner_batch):
        b_end = min(B, b_start + inner_batch)
        Bb = b_end - b_start

        # Light voxel padding (reuse buffer)
        vol_b[:Bb] = cp.pad(voxels_batch[b_start:b_end].astype(cp.float32),
                            ((0,0),(pad,pad),(pad,pad),(pad,pad)))

        # FFT of padded light
        Fp_all = cpx_fft.fftn(vol_b[:Bb]*scale, axes=(1,2,3))

        # Precompute solute + distance
        solute[:Bb] = (vol_b[:Bb] > threshold).astype(cp.float32)
        for i in range(Bb):
            dist_out[i] = distance_transform_edt(~solute[i]) * float(voxel_size)

        Fe = cpx_fft.fftn(solute[:Bb]*scale, axes=(1,2,3))

        Ppp = map2iq.batched_bincount(idx, (cp.abs(Fp_all)**2).reshape(Bb,-1),
                                      N_sum, valid, Bb)
        Pee = map2iq.batched_bincount(idx, (cp.abs(Fe)**2).reshape(Bb,-1),
                                      N_sum, valid, Bb)
        Cpe = map2iq.batched_bincount(idx, cp.real(Fp_all * cp.conj(Fe)).reshape(Bb,-1),
                                      N_sum, valid, Bb)

        # --------------------------
        # SHELL LOOP — memory safe
        # --------------------------
        for s_th in shell_vals:
            shell_b = ((dist_out[:Bb] <= s_th) & (~solute[:Bb])).astype(cp.float32)
            Fs = cpx_fft.fftn(shell_b * scale, axes=(1,2,3))

            Pss = map2iq.batched_bincount(idx, (cp.abs(Fs)**2).reshape(Bb,-1), N_sum, valid, Bb)
            Cps = map2iq.batched_bincount(idx, cp.real(Fp_all * cp.conj(Fs)).reshape(Bb,-1), N_sum, valid, Bb)
            Ces = map2iq.batched_bincount(idx, cp.real(Fe * cp.conj(Fs)).reshape(Bb,-1), N_sum, valid, Bb)

            # Δρ scan
            for b_val in delta_vals:
                Iq = Ppp + (a*a)*Pee + (b_val*b_val)*Pss - 2*a*Cpe + 2*b_val*Cps - 2*(a*b_val)*Ces
                Delta_q = (Iq - I_dark[None,:]) * absolute_scale

                # interpolation
                left  = cp.searchsorted(q_rad, exp_q, side='right') - 1
                left = cp.clip(left, 0, q_rad.size-1)
                right = cp.clip(left+1, 0, q_rad.size-1)
                qL = q_rad[left]
                qR = q_rad[right]
                w = cp.where(qR>qL,(exp_q - qL)/(qR - qL),0.0)

                D_left = Delta_q[:, left]
                D_right= Delta_q[:, right]
                Dexp   = (1-w)*D_left + w*D_right

                den = cp.sum(Dexp*Dexp, axis=-1)
                num = cp.sum(exp_I * Dexp, axis=-1)
                alpha = cp.where(den>0,num/den,1.0)
                alpha = cp.where(alpha<0,1.0,alpha)

                penalty = yield_weight*(alpha - target_yield)**2
                res2 = cp.sum((exp_I - alpha[:,None]*Dexp)**2, axis=-1)
                r2 = 1.0 - res2/ss_tot
                score = r2 - penalty

                improved = score > scores[b_start:b_end]
                idx_imp = cp.where(improved)[0]
                for gi in idx_imp:
                    gi_abs = b_start + gi
                    scores[gi_abs]      = float(score[gi])
                    best_delta[gi_abs]  = float(b_val)
                    best_shell[gi_abs]  = float(s_th)
                    best_alpha[gi_abs]  = float(alpha[gi])
                    best_ascore[gi_abs] = float(penalty)

            # free per-shell temp
            del Fs, Pss, Cps, Ces
            cp._default_memory_pool.free_all_blocks()

        # free per-batch
        del Fp_all, Fe
        cp._default_memory_pool.free_all_blocks()

    return scores, cp.full_like(best_delta, 1.0), best_alpha, best_ascore, best_delta, cp.full_like(best_delta, threshold), best_shell

