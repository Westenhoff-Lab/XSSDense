"""
Scattering of voxel electron densities with a simple solvent model:
excluded bulk solvent plus a hydration shell (paper Eq. 4-7).
"""

import numpy
from scipy import ndimage, fft

_B = {"xp": numpy, "ndimage": ndimage, "fft": fft, "workers": -1}


def set_backend(device="auto", fft_workers=-1):
    """device 'cpu', 'gpu' or 'auto' (GPU if CuPy sees one); fft_workers = threads per CPU FFT. Returns the device used."""
    if device == "auto":
        try:
            import cupy

            device = "gpu" if cupy.cuda.runtime.getDeviceCount() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if device == "gpu":
        import cupy
        from cupyx.scipy import ndimage as gpu_ndimage, fft as gpu_fft

        _B.update(xp=cupy, ndimage=gpu_ndimage, fft=gpu_fft)
    else:
        _B.update(xp=numpy, ndimage=ndimage, fft=fft)
    _B["workers"] = fft_workers
    _radius_cache.clear()
    return device


def _to_numpy(a):
    return a if isinstance(a, numpy.ndarray) else a.get()


def envelope_and_shell(density, dv, shell, threshold):
    """Envelope = voxels with density > threshold; shell = voxels outside the envelope within `shell` Å of it."""
    envelope = density > threshold
    dist = _B["ndimage"].distance_transform_edt(~envelope, sampling=dv)
    return envelope.astype(density.dtype), ((dist <= shell) & ~envelope).astype(density.dtype)


def contrast_map(density, dv, rho_bulk=0.334, drho=0.0, shell=3.0, threshold=None):
    """Density minus the displaced bulk solvent, plus the hydration shell contrast.
    threshold defaults to rho_bulk. In vacuum (rho_bulk = 0, drho = 0) this is just the density itself."""
    density = _B["xp"].asarray(density, dtype=numpy.float32)
    if rho_bulk == 0 and drho == 0:
        return density
    envelope, shell_mask = envelope_and_shell(density, dv, shell, rho_bulk if threshold is None else threshold)
    return density - numpy.float32(rho_bulk) * envelope + numpy.float32(drho) * shell_mask


def amplitude_3d(grid, padding_factor=3):
    """Complex scattering amplitude of a zero-padded grid, zero frequency at the center."""
    xp, F = _B["xp"], _B["fft"]
    N = grid.shape[0]
    Npad = N * (padding_factor - 1) // 2
    padded = xp.pad(grid, Npad)
    return F.fftshift(F.fftn(padded, workers=_B["workers"]) if xp is numpy else F.fftn(padded))


def compute_3d_intensity(grid, padding_factor=3):
    """|F(grid)|² on a zero-padded grid."""
    A = amplitude_3d(grid, padding_factor)
    return A.real**2 + A.imag**2


_radius_cache = {}


def spherical_average(intensity_3d, voxel_size, qmin=0.0, qmax=None):
    """Average over shells one grid step wide; q at the shell centers, (k + 0.5)·Δq with Δq = 2π / (N·dv).
    Returns NumPy arrays."""
    xp = _B["xp"]
    N = intensity_3d.shape[0]
    center = N // 2
    if N not in _radius_cache:
        z, y, x = xp.indices((N, N, N))
        r = xp.sqrt(((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2).astype(xp.float64)).astype(xp.int64).ravel()
        _radius_cache[N] = (r, _to_numpy(xp.bincount(r, minlength=center)[:center]))
    r, counts = _radius_cache[N]

    sums = _to_numpy(xp.bincount(r, weights=intensity_3d.ravel().astype(xp.float64), minlength=center)[:center])
    intensity_1d = sums / counts
    q = (numpy.arange(center) + 0.5) * 2 * numpy.pi / (N * voxel_size)
    if qmax:
        mask = (qmin <= q) & (q <= qmax)
        return q[mask], intensity_1d[mask]
    return q, intensity_1d


def scattering_curve_gpu(density, dv, rho_bulk=0.334, drho=0.0, shell=3.0, padding=3, qmax=0.7, threshold=None):
    """I(q) of one density, including the solvent model, on the active backend (name kept). Returns NumPy (q, I)."""
    contrast = contrast_map(density, dv, rho_bulk, drho, shell, threshold)
    I3d = compute_3d_intensity(contrast * numpy.float32(dv**3), padding)
    return spherical_average(I3d, dv, 0.0, qmax)
