# !/usr/bin/env python3
# coding:utf-8
import os
import argparse
import json
import numpy as np
import cupy as cp
import tensorflow as tf
import h5py

import map2iq_GPU as map2iq
from VAE import VAE
from density_manipulation import create_ccp4_map_
from data_class_GPU import ScatterData

from cupyx.scipy import ndimage as cpx_nd
from cupyx.scipy.ndimage import distance_transform_edt
from cupyx.scipy import fft as cpx_fft
import matplotlib.pyplot as plt

def unnormalize_minmax(x_norm, xmin, xmax):
    return (x_norm * (xmax - xmin) + xmin).astype(np.float32)

def compute_exact_deltaI(
        voxel_light_cp, # cp.ndarray (D, D, D) — best voxel from SA
        voxel_dark_cp,           # cp.ndarray (D, D, D) — ground state voxel
        voxel_size: float,
        rho_bulk: float,
        absolute_scale: float,
        drho_light: float,       # delta_arr[0]  from run_batched
        shell_light: float,      # shell_arr[0]  from run_batched
        drho_dark: float,        # dark_drho     from get_absolute_scale
        shell_dark: float,       # dark_shell    from get_absolute_scale
        threshold: float = 0.4,
        pad_frac: int = 1.5
):
    """
    Reconstruct ΔI(q) curve evaluated inside run_batched(), using:
        - SAME padded FFT grids
        - SAME hydration shell construction
        - SAME Δρ and shell thickness chosen by run_batched
        - SAME q-binning
        - SAME radial averaging
        - SAME ΔI = (I_light - I_dark) * absolute_scale

    Returns:
        q_rad : cp.ndarray
        delta_I : cp.ndarray
    """

    scale = voxel_size ** 3
    sV = 1.0
    a = rho_bulk * sV

    # --------------------------
    # Pad light and dark densities
    # --------------------------
    D = voxel_light_cp.shape[0]
    pad = int(pad_frac * D)

    vL = cp.pad(voxel_light_cp.astype(cp.float32),
                ((pad, pad), (pad, pad), (pad, pad)))
    vD = cp.pad(voxel_dark_cp.astype(cp.float32),
                ((pad, pad), (pad, pad), (pad, pad)))

    # --------------------------
    # q-binning (same as run_batched)
    # --------------------------
    Z = vL.shape[0]
    q_rad, idx, N_sum, valid = map2iq._prepare_q_binning(Z, voxel_size)

    # --------------------------
    # Build DARK scattering exactly
    # --------------------------
    sol_dark = (vD > threshold)
    dist_dark = distance_transform_edt(~sol_dark) * voxel_size
    shell_dark_mask = ((dist_dark <= shell_dark) & (~sol_dark)).astype(cp.float32)

    Fp_dark = cpx_fft.fftn(vD * scale)
    Fe_dark = cpx_fft.fftn(sol_dark.astype(cp.float32) * scale)
    Fs_dark = cpx_fft.fftn(shell_dark_mask * scale)

    A_dark = Fp_dark - rho_bulk * Fe_dark + drho_dark * Fs_dark
    I_dark = map2iq._radial_average_power(A_dark, idx, N_sum, valid)

    # --------------------------
    # Build LIGHT scattering exactly
    # --------------------------
    sol_light = (vL > threshold)
    dist_light = distance_transform_edt(~sol_light) * voxel_size
    shell_light_mask = ((dist_light <= shell_light) & (~sol_light)).astype(cp.float32)

    Fp_light = cpx_fft.fftn(vL * scale)
    Fe_light = cpx_fft.fftn(sol_light.astype(cp.float32) * scale)
    Fs_light = cpx_fft.fftn(shell_light_mask * scale)

    A_light = Fp_light - rho_bulk * Fe_light + drho_light * Fs_light
    I_light = map2iq._radial_average_power(A_light, idx, N_sum, valid)

    # --------------------------
    # Final EXACT ΔI(q)
    # --------------------------
    delta_I = (I_light - I_dark) * absolute_scale

    return q_rad, delta_I

def get_absolute_scale(voxel_pdb_cp, voxel_size, rho_bulk, absolute_data):
    """
    Calculate absolute scale for given voxel and voxel size.
    voxel_pdb_cp: CuPy array
    voxel_size: float
    rho_bulk: float
    absolute_data: CuPy array of experimental absolute data
    """
    threshold = 0.4
    sV = 1.0
    a = rho_bulk * sV

    shell_vals = cp.array([1., 2., 3., 4., 5.], dtype=cp.float32)
    delta_percent = cp.arange(0, 13, dtype=cp.float32)
    b_vals = (delta_percent * 0.01 * rho_bulk).astype(cp.float32)

    # ------------------------------
    # Pad voxel (KEEP padding=3×)
    # ------------------------------
    N = int(voxel_pdb_cp.shape[0])
    pad_vox = int(1.5 * N)

    voxel = cp.pad(
        voxel_pdb_cp.astype(cp.float32),
        ((pad_vox, pad_vox), (pad_vox, pad_vox), (pad_vox, pad_vox)),
        mode="constant"
    )

    scale_voxel = voxel_size ** 3

    # ------------------------------
    # FFT(Fp) in complex64
    # ------------------------------
    Fp = map2iq._fftn_rho_scaled(voxel, float(voxel_size)).astype(cp.complex64)
    Z = voxel.shape[0]
    q_rad, idx, N_sum, valid = map2iq._prepare_q_binning(Z, float(voxel_size))

    # ------------------------------
    # Experimental data
    # ------------------------------
    q_exp = cp.asarray(absolute_data.q, dtype=cp.float32)
    i_exp = cp.asarray(absolute_data.i, dtype=cp.float32)

    sel = q_exp <= q_rad[-1]
    q_exp = q_exp[sel]
    i_exp = i_exp[sel]

    y_mean = cp.mean(i_exp)
    ss_tot = cp.sum((i_exp - y_mean) ** 2)
    if float(ss_tot) == 0.0:
        ss_tot = cp.float32(1.0)

    left_idx = cp.searchsorted(q_rad, q_exp, side='right') - 1
    left_idx = cp.clip(left_idx, 0, q_rad.size - 1)
    right_idx = cp.clip(left_idx + 1, 0, q_rad.size - 1)
    qL = q_rad[left_idx]
    qR = q_rad[right_idx]
    w = cp.where(qR > qL, (q_exp - qL) / (qR - qL), 0.0)

    # ------------------------------
    # Masks
    # ------------------------------
    solute = (voxel > threshold)
    dist_out = distance_transform_edt(~solute) * float(voxel_size)
    solute = solute.astype(cp.float32)

    # ------------------------------
    # FFT(Fe) ONCE
    # ------------------------------
    Fe = cpx_fft.fftn(solute * scale_voxel).astype(cp.complex64)
    Pee = map2iq._radial_average_power(Fe, idx, N_sum, valid).astype(cp.float32)

    # Cpe = <Fp, Fe>
    tmp = Fp * cp.conj(Fe)
    Cpe_raw = cp.real(tmp)
    del tmp
    Cpe = (cp.bincount(idx, Cpe_raw.ravel(), minlength=N_sum.size)[valid] /
           N_sum[valid]).astype(cp.float32)
    del Cpe_raw

    Ppp = map2iq._radial_average_power(Fp, idx, N_sum, valid).astype(cp.float32)

    # safety flush
    cp._default_memory_pool.free_all_blocks()

    # ------------------------------
    # Scan shells × deltas
    # ------------------------------
    best_r2 = -1e38
    best_scale = cp.float32(1.0)
    best_delta = cp.float32(0.0)
    best_shell = cp.float32(1.0)

    for s_th in shell_vals:

        # Build shell mask
        shell_mask = ((dist_out <= s_th) &
                      (~solute.astype(bool))).astype(cp.float32)

        # FFT(Fs)
        Fs = cpx_fft.fftn(shell_mask * scale_voxel).astype(cp.complex64)

        # Pss
        Pss = map2iq._radial_average_power(Fs, idx, N_sum, valid).astype(cp.float32)

        # Cps
        tmp = Fp * cp.conj(Fs)          # allocate ~Z^3 complex64
        Cps_raw = cp.real(tmp)
        del tmp
        Cps = (cp.bincount(idx, Cps_raw.ravel(), minlength=N_sum.size)[valid] /
               N_sum[valid]).astype(cp.float32)
        del Cps_raw

        # Ces
        tmp = Fe * cp.conj(Fs)          # allocate ~Z^3 complex64
        Ces_raw = cp.real(tmp)
        del tmp
        Ces = (cp.bincount(idx, Ces_raw.ravel(), minlength=N_sum.size)[valid] /
               N_sum[valid]).astype(cp.float32)
        del Ces_raw

        cp._default_memory_pool.free_all_blocks()

        # ------------------------------
        # Δρ loop
        # ------------------------------
        for b in b_vals:

            Iq = (
                Ppp
                + (a * a) * Pee
                + (b * b) * Pss
                - 2 * a * Cpe
                + 2 * b * Cps
                - 2 * (a * b) * Ces
            )

            # interpolate
            I_interp = (1 - w) * Iq[left_idx] + w * Iq[right_idx]

            den = cp.sum(I_interp * I_interp)
            num = cp.sum(i_exp * I_interp)

            alpha = num / den if den > 0 else cp.float32(1.0)
            if alpha < 0:
                alpha = cp.float32(1.0)

            res2 = cp.sum((i_exp - alpha * I_interp)**2)
            r2 = 1.0 - res2 / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_scale = alpha
                best_delta = b
                best_shell = s_th

        # free per-shell
        del Fs, Pss, Cps, Ces
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()

    return float(best_scale), float(best_delta), float(best_shell)

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
    inner_batch: int = 1,   # reduce batch to save GPU memory
    shell_vals=cp.asarray([1.,2.,3.,4.,5.], cp.float32),
    delta_vals=None
):
    """
    Memory-safe batched scoring with proper boolean masks for distance transforms.
    """
    threshold = 0.4
    pad_frac = 1.5# light padding
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
    ss_tot = cp.sum((exp_q*(exp_I - y_mean))**2)
    #y_log = cp.log(abs(exp_I))
    #y_mean_log = cp.mean(y_log)
    #ss_tot = cp.sum((y_log - y_mean_log) ** 2)
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

    solute_dark = (dark > threshold)  # bool mask
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
    solute = cp.empty((inner_batch, Z, Z, Z), dtype=bool)  # boolean mask
    vol_b = cp.empty((inner_batch, Z, Z, Z), dtype=cp.float32)
    ref_padded = cp.pad(ground_state_voxel.astype(cp.float32),
                        ((pad, pad), (pad, pad), (pad, pad)))

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

        # Precompute solute + distance (boolean)
        solute[:Bb] = vol_b[:Bb] > threshold
        for i in range(Bb):
            dist_out[i] = distance_transform_edt(~solute[i]) * float(voxel_size)

        Fe = cpx_fft.fftn(solute[:Bb].astype(cp.float32)*scale, axes=(1,2,3))

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
                exp_I_log = cp.log(abs(exp_I))
                Dexp_log = cp.log(abs(Dexp)*alpha[:, None])
                res2 = cp.sum((exp_q * (exp_I - alpha[:,None]*Dexp))**2, axis=-1)
                #res2 = cp.sum((exp_I_log - Dexp_log) ** 2, axis=-1)
                r2 = 1.0 - res2/ss_tot

                diff_norm = cp.max (abs(vol_b[:Bb] - ref_padded),axis=(1, 2, 3))  # or voxels before padding

                zero_penalty = cp.where(diff_norm > 0.4, 0, 1 - diff_norm)

                mean_voxel = cp.mean(vol_b[:Bb], axis=0, keepdims=True)
                diversity = cp.mean((vol_b[:Bb] - mean_voxel) ** 2, axis=(1, 2, 3))

                score = r2 - penalty - zero_penalty + 0.001 * diversity

                improved = score > scores[b_start:b_end]
                idx_imp = cp.where(improved)[0]
                for gi in idx_imp:
                    gi_abs = b_start + gi
                    scores[gi_abs]      = float(score[gi])
                    best_delta[gi_abs]  = float(b_val)
                    best_shell[gi_abs]  = float(s_th)
                    best_alpha[gi_abs]  = float(alpha[gi])
                    best_ascore[gi_abs] = float(penalty[gi])

            # free per-shell temp
            del Fs, Pss, Cps, Ces
            cp._default_memory_pool.free_all_blocks()

        # free per-batch
        del Fp_all, Fe
        cp._default_memory_pool.free_all_blocks()

    return scores, best_delta, best_shell, best_alpha, best_ascore, 0.4

    return scores, cp.full_like(best_delta, 1.0), best_alpha, best_ascore, best_delta, cp.full_like(best_delta, threshold), best_shell

def vae_decode_gpu(vae, latent_cp):
    """
    latent_cp: (B, latent_dim) CuPy array
    returns: (B, D, D, D, 1) CuPy
    """
    z_tf = tf_from_cp_dlpack(latent_cp)
    rec_tf = vae.decode(z_tf)      # TensorFlow model call (GPU)
    rec_cp = cp_from_tf_dlpack(rec_tf)
    return rec_cp


# ──────────────────────────────────────────────────────────────────────
# Conversion from TensorFlow to Cupy and vice versa
# ──────────────────────────────────────────────────────────────────────
def cp_from_tf_dlpack(t):
    """Zero-copy TF->CuPy conversion via DLPack (preferred)."""
    try:
        return cp.fromDlpack(tf.experimental.dlpack.to_dlpack(t))
    except Exception:
        return cp.asarray(t.numpy())  # fallback (copy via host)


def tf_from_cp_dlpack(a_cp: cp.ndarray):
    """Zero-copy CuPy->TF conversion via DLPack."""
    try:
        return tf.experimental.dlpack.from_dlpack(a_cp.toDlpack())
    except Exception:
        return tf.convert_to_tensor(cp.asnumpy(a_cp))  # fallback (copy via host)


# ──────────────────────────────────────────────────────────────────────
# Evolution class (GPU)
# ──────────────────────────────────────────────────────────────────────
class GeneticAlgorithmGPU:
    def __init__(
        self,
        output_folder,
        voxel_size,
        voxel_pdb_cpu,
        process_result,
        vae,
        latent_size,
        absolute_data,
        latent_stats,
        params,
        target_yield,
        yield_weight,
        rho_bulk,
        max_iter,
        population_size,
        batch_size
    ):

        self.output_folder = output_folder
        self.voxel_size = voxel_size
        self.process_result = process_result
        self.absolute_data = absolute_data
        self.vae = vae
        self.latent_size = latent_size
        self.target_yield = target_yield
        self.yield_weight = yield_weight
        self.rho_bulk = rho_bulk

        self.max_iter = max_iter
        self.population_size = population_size
        self.batch_size = batch_size

        self.voxel_pdb = cp.asarray(voxel_pdb_cpu, dtype=cp.float32)

        self.xmin, self.xmax = params

        ls = cp.asarray(latent_stats, dtype=cp.float32)
        self.latent_lower = ls[:, 2]
        self.latent_upper = ls[:, 3]
        self.latent_sigma = ls[:, 1]

        print("[GA] Computing absolute scale...")
        self.absolute_scale, self.dark_drho, self.dark_shell = get_absolute_scale(
            self.voxel_pdb,
            self.voxel_size,
            self.rho_bulk,
            self.absolute_data
        )

    # --------------------------------------------------
    def decode(self, Z):
        rec = vae_decode_gpu(self.vae, Z)
        return rec[..., 0]

    # --------------------------------------------------
    def evaluate(self, voxels):
        extrapolated = unnormalize_minmax(voxels, self.xmin, self.xmax)
        extrapolated = extrapolated.clip(0)

        scores, drho, best_shell, alpha, alpha_score, thr = run_batched(
            voxels_batch=extrapolated,
            iq_file=self.process_result,
            ground_state_voxel=self.voxel_pdb,
            absolute_scale=self.absolute_scale,
            drho=self.dark_drho,
            dark_shell=self.dark_shell,
            voxel_size=self.voxel_size,
            target_yield=self.target_yield,
            yield_weight=self.yield_weight,
            inner_batch=1
        )

        return scores, alpha, alpha_score, drho, thr, best_shell



    # --------------------------------------------------
    def initialize_population(self):
        z = cp.random.uniform(0, 1,
            size=(self.population_size, self.latent_size)).astype(cp.float32)
        return self.latent_lower + (self.latent_upper - self.latent_lower) * z

    # --------------------------------------------------
    def tournament_selection(self, Z, scores, k=3):
        selected = []
        for _ in range(self.population_size):
            idx = cp.random.randint(0, self.population_size, size=(k,))
            best = idx[cp.argmax(scores[idx])]
            selected.append(Z[best])
        return cp.stack(selected)

    # --------------------------------------------------
    def crossover(self, parents):
        idx = cp.random.permutation(len(parents))
        p1 = parents
        p2 = parents[idx]

        # Blend crossover
        alpha = cp.random.uniform(0, 1, size=p1.shape).astype(cp.float32)
        child = alpha * p1 + (1 - alpha) * p2
        mask = cp.random.rand(*p1.shape) < 0.3
        child = cp.where(mask, p1, child)

        return child

    # --------------------------------------------------
    def mutate(self, Z, rate=0.3):
        noise = cp.random.normal(0, self.latent_sigma, Z.shape)

        # stronger mutation for diversity
        mask = cp.random.rand(*Z.shape) < rate
        Z = Z + noise * mask

        # small global noise to avoid collapse
        Z = Z + 0.05 * cp.random.normal(0, 1, Z.shape)

        return cp.clip(Z, self.latent_lower, self.latent_upper)

    # --------------------------------------------------
    def select_diverse_topk(self,Z, scores, k, min_dist=0.5):
        idx_sorted = cp.argsort(scores)[::-1]
        selected = []

        for idx in idx_sorted:
            z = Z[idx]

            if len(selected) == 0:
                selected.append(idx)
                continue

            dists = cp.linalg.norm(Z[selected] - z, axis=1)
            if cp.all(dists > min_dist):
                selected.append(idx)

            if len(selected) == k:
                break

        return cp.asarray(selected, dtype=cp.int32)

    # --------------------------------------------------
    def run(self, logfile):
        top_k = 5

        top_voxels_all = []
        top_latents_all = []
        top_scores_all = []
        top_alpha_all = []
        top_alpha_scores_all = []
        top_drho_all = []
        top_shell_all = []

        Z = self.initialize_population()

        best_score = -1e30
        best_latent = None
        best_voxel = None

        score_history = []

        for gen in range(self.max_iter):

            # Decode + score
            voxels = self.decode(Z)
            scores, alpha, alpha_score, drho, thr, shell = self.evaluate(voxels)
            idx_sorted = self.select_diverse_topk(Z, scores, top_k)

            top_voxels = voxels[idx_sorted]
            top_latents = Z[idx_sorted]
            top_alpha = alpha[idx_sorted]
            top_alpha_scores = alpha_score[idx_sorted]
            top_scores = scores[idx_sorted]
            top_drho = drho[idx_sorted]
            top_shell = shell[idx_sorted]

            top_voxels_all.append(cp.asnumpy(top_voxels))
            top_latents_all.append(cp.asnumpy(top_latents))
            top_alpha_all.append(cp.asnumpy(top_alpha))
            top_alpha_scores_all.append(cp.asnumpy(top_alpha_scores))
            top_scores_all.append(cp.asnumpy(top_scores))
            top_drho_all.append(cp.asnumpy(top_drho))
            top_shell_all.append(cp.asnumpy(top_shell))

            # ---- ELITISM ----
            elite_k = 5
            elite_idx = cp.argsort(scores)[-elite_k:]
            elites = Z[elite_idx]

            # Track best
            idx = int(cp.argmax(scores))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_latent = Z[idx].copy()
                best_voxel = voxels[idx].copy()

            score_history.append(float(cp.max(scores)))

            print(f"[GA] Gen {gen}/{self.max_iter}  Best={best_score:.6f}")
            logfile.write(f"Gen {gen} Best={best_score:.6f}\n")

            # ---- GA pipeline ----
            parents = self.tournament_selection(Z, scores)
            children = self.crossover(parents)
            children = self.mutate(children)

            # ---- random injection ----
            n_rand = int(0.4 * self.population_size)
            rand = self.initialize_population()[:n_rand]
            children[-n_rand:] = rand

            # ---- insert elites ----
            children[:elite_k] = elites

            Z = children

            cp._default_memory_pool.free_all_blocks()

        return (
            best_voxel,
            best_latent,
            best_score,
            score_history,
            np.array(top_voxels_all),  # (T, K, D, D, D)
            np.array(top_latents_all),  # (T, K, latent)
            np.array(top_alpha_all),
            np.array(top_alpha_scores_all),
            np.array(top_scores_all),
            np.array(top_drho_all),
            np.array(top_shell_all)# (T, K)
        )


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.set_printoptions(precision=10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to Keras model/weights of the VAE", type=str, required=True)
    parser.add_argument("--iq_path", help="path to experimental data in absolute units", type=str, required=True)
    parser.add_argument("--output_folder", help="path where results will be saved", type=str, required=True)
    parser.add_argument("--voxel", help="path to ground state voxel .npy", type=str, required=True)
    parser.add_argument("--ground_scattering", help="Scattering of ground state in absolute units", type=str,
                        required=True)
    parser.add_argument("--params", help="Path to normalization params file (.npz) with key 'M'", type=str,
                        required=True)
    parser.add_argument("--meta_json", help="voxel size in Å)", type=float,
                        required=True)
    parser.add_argument("--batch_size", help="Batch size for GPU scoring ", type=int, default=8)
    parser.add_argument("--max_iter", help="maximum number of iteration", type=int, default=80)
    parser.add_argument("--target_yield", help="Estimated experimental photoactivation yield", type=float, default=0.5)
    parser.add_argument("--yield_weight", help="weight of yield", type=float, default=2)
    parser.add_argument("--shell_thickness", help="Thickness of hydration shell in Å", type=float, default=3)
    parser.add_argument("--rho_bulk", help="value of bulk solvent in e/Å³", type=float, default=0.334)
    parser.add_argument("--latent_size", help="latent variable size", type=int, default=8)

    args = parser.parse_args()

    model_path = args.model_path
    iq_path = args.iq_path
    output_folder = args.output_folder
    max_iter = args.max_iter
    absolute_scattering = args.ground_scattering
    params = args.params
    meta_json = args.meta_json
    latent_size = args.latent_size
    batch_size = args.batch_size
    target_yield = args.target_yield
    yield_weight = args.yield_weight
    rho_bulk = args.rho_bulk

    os.makedirs(output_folder, exist_ok=True)

    latent_stats = np.loadtxt(os.path.join(model_path, "latent_stats.txt"), delimiter=" ")

    # normalization M for I/O stage
    with open(params) as f:
        meta = json.load(f)

    xmin = float(meta["normalization"]["min"])
    xmax = float(meta["normalization"]["max"])

    #with open(meta_json, "r") as f:
    #    meta = json.load(f)
    voxel_size = meta_json

    # Ground voxel (CPU)
    with h5py.File(args.voxel, "r") as f:
        ground_state_voxel_cpu = f["rho"][0]

    # Save CCP4 map of the ground state
    create_ccp4_map_(ground_state_voxel_cpu, f"{output_folder}/ground_state.ccp4", voxel_size=voxel_size)

    # Load VAE
    print("Initiating Network...")
    if os.path.isdir(f"{args.model_path}/encoder_model") and os.path.isdir(f"{args.model_path}/decoder_model"):
        encoder = tf.saved_model.load(f"{args.model_path}/encoder_model")
        decoder = tf.saved_model.load(f"{args.model_path}/decoder_model")
    else:
        encoder = tf.keras.models.load_model(f"{args.model_path}/encoder_model.keras")
        decoder = tf.keras.models.load_model(f"{args.model_path}/decoder_model.keras")
    vae = VAE(latent_size, encoder, decoder)

    # Experimental SAXS (GPU)
    print("Processing Experimental scattering...")
    saxs_data = ScatterData(iq_path, ",")
    absolute_data = ScatterData(absolute_scattering, ",")

    logfile = open(f"{output_folder}/log.txt", "a")

    ga = GeneticAlgorithmGPU(
        output_folder=output_folder,
        voxel_size=voxel_size,
        voxel_pdb_cpu=ground_state_voxel_cpu,
        process_result=saxs_data,
        vae=vae,
        latent_size=latent_size,
        absolute_data=absolute_data,
        latent_stats=latent_stats,
        params=(xmin,xmax),
        target_yield=target_yield,
        yield_weight=yield_weight,
        rho_bulk=rho_bulk,
        max_iter=max_iter,
        population_size=64,  # try 64–256
        batch_size=batch_size
    )

    best_voxel, best_latent, best_score, history,top_voxels, top_latents,top_alphas, top_alpha_scores,top_scores,top_drhos, top_shells = ga.run(logfile)

    # ================================
    # SAVE RESULTS (LAST GENERATION ONLY)
    # ================================
    D = ground_state_voxel_cpu.shape[0]

    # Last generation only
    last_voxels = top_voxels[-1]  # (K, D, D, D)
    last_latents = top_latents[-1]
    last_alphas = top_alphas[-1]
    last_alpha_scores = top_alpha_scores[-1]
    last_scores = top_scores[-1]
    last_drhos = top_drhos[-1]
    last_shells = top_shells[-1]

    K = last_voxels.shape[0]

    for k in range(K):
        v = last_voxels[k]
        latent = last_latents[k]
        alpha = last_alphas[k]
        alpha_score = last_alpha_scores[k]
        score = last_scores[k]
        drho = last_drhos[k]
        shell = last_shells[k]

        tag = f"final_rank{k}"

        # -------------------------
        # Δρ (normalized)
        # -------------------------
        diff_norm = v[:D, :D, :D]

        #np.save(f"{output_folder}/{tag}_diff_norm.npy", diff_norm)

        #create_ccp4_map_(
        #    unnormalize_minmax(diff_norm ,xmin, xmax).clip(0),
        #    f"{output_folder}/{tag}_diff.ccp4",
        #    voxel_size=voxel_size
        #)

        # -------------------------
        # Metadata
        # -------------------------
        np.save(f"{output_folder}/{tag}_latent.npy", latent)
        np.save(f"{output_folder}/{tag}_alpha.npy", alpha)
        np.save(f"{output_folder}/{tag}_alpha_score.npy", alpha_score)
        np.save(f"{output_folder}/{tag}_score.npy", score)
        np.save(f"{output_folder}/{tag}_drho.npy", drho)
        np.save(f"{output_folder}/{tag}_shell_thickness.npy", shell)

        # -------------------------
        # Extrapolated density
        # -------------------------
        diff = unnormalize_minmax(diff_norm, xmin, xmax).clip(0)
        extrapolated = diff

        np.save(f"{output_folder}/{tag}_density.npy", extrapolated)

        create_ccp4_map_(
            extrapolated,
            f"{output_folder}/{tag}_density.ccp4",
            voxel_size=voxel_size
        )

        # -------------------------
        # EXACT SAXS curve (NEW ✅)
        # -------------------------
        q_model, delta_I = compute_exact_deltaI(
            voxel_light_cp=cp.asarray(extrapolated),
            voxel_dark_cp=cp.asarray(ground_state_voxel_cpu),
            voxel_size=voxel_size,
            rho_bulk=rho_bulk,
            absolute_scale=ga.absolute_scale,
            drho_light=drho,
            shell_light=shell,
            drho_dark=ga.dark_drho,
            shell_dark=ga.dark_shell
        )

        delta_I_interp = cp.interp(saxs_data.q, q_model, delta_I)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(saxs_data.q.get(), saxs_data.i.get(), label="Exp")
        ax.plot(saxs_data.q.get(), (delta_I_interp * alpha).get(), label="Model")
        ax.legend()

        fig.savefig(f"{output_folder}/{tag}_deltaI.png")
        plt.close(fig)

        for n,diff,drho,shell,alpha in zip(range(max_iter),top_voxels[:,0],top_drhos[:,0],top_shells[:,0],top_alphas[:,0]):
            extrapolated = unnormalize_minmax(diff,xmin, xmax).clip(0)
            q_model, delta_I = compute_exact_deltaI(
            voxel_light_cp=cp.asarray(extrapolated),
            voxel_dark_cp=cp.asarray(ground_state_voxel_cpu),
            voxel_size=voxel_size,
            rho_bulk=rho_bulk,
            absolute_scale=ga.absolute_scale,
            drho_light=drho,
            shell_light=shell,
            drho_dark=ga.dark_drho,
            shell_dark=ga.dark_shell
        )
            delta_I_interp = cp.interp(saxs_data.q, q_model, delta_I)
            data = np.vstack((saxs_data.q.get(),delta_I_interp.get()*alpha))
            np.savetxt(f"{output_folder}/iteration_{n}.dat", data, delimiter=",")
            # ================================
    # SAVE GLOBAL ARRAYS
    # ================================
    #np.save(f"{output_folder}/top_voxels_all.npy", top_voxels)
    #np.save(f"{output_folder}/top_latents_all.npy", top_latents)
    #np.save(f"{output_folder}/top_scores_all.npy", top_scores)
    #np.save(f"{output_folder}/top_alphas_all.npy", top_alphas)
    #np.save(f"{output_folder}/top_alpha_scores_all.npy", top_alpha_scores)
    #np.save(f"{output_folder}/top_drhos_all.npy", top_drhos)
    #np.save(f"{output_folder}/top_shells_all.npy", top_shells)

    #np.save(f"{output_folder}/score_history.npy", np.array(history))

    np.save(f"{output_folder}/fitted_absolute_alpha_scale.npy", ga.absolute_scale)
    np.save(f"{output_folder}/fitted_absolute_drho.npy", ga.dark_drho)
    np.save(f"{output_folder}/fitted_absolute_shell.npy", ga.dark_shell)
    logfile.close()


