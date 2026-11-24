"""Scattering of voxel densities with a bulk-solvent and hydration-shell model."""

import numpy
from scipy import ndimage, fft

FFT_WORKERS = -1  # threads per FFT; -1 = all cores


def envelope_and_shell(density, dv, shell, threshold):
    """Masks of the envelope and its hydration shell."""
    envelope = density > threshold
    dist = ndimage.distance_transform_edt(~envelope, sampling=dv)
    return envelope.astype(density.dtype), ((dist <= shell) & ~envelope).astype(density.dtype)


def contrast_map(density, dv, rho_bulk=0.334, drho=0.0, shell=3.0, threshold=None):
    """Density minus displaced bulk solvent plus hydration-shell contrast."""
    density = numpy.asarray(density, dtype=numpy.float32)
    if rho_bulk == 0 and drho == 0:
        return density
    envelope, shell_mask = envelope_and_shell(density, dv, shell, rho_bulk if threshold is None else threshold)
    return density - numpy.float32(rho_bulk) * envelope + numpy.float32(drho) * shell_mask


def amplitude_3d(grid, padding_factor=3):
    """Scattering amplitude of the zero-padded grid."""
    Npad = grid.shape[0] * (padding_factor - 1) // 2
    padded = numpy.pad(grid, Npad)
    return fft.fftshift(fft.fftn(padded, workers=FFT_WORKERS))


def compute_3d_intensity(grid, padding_factor=3):
    A = amplitude_3d(grid, padding_factor)
    return A.real**2 + A.imag**2


_radius_cache = {}


def spherical_average(intensity_3d, voxel_size, qmin=0.0, qmax=None):
    """Spherical average of a 3D intensity, I(q)."""
    N = intensity_3d.shape[0]
    center = N // 2
    if N not in _radius_cache:
        z, y, x = numpy.indices((N, N, N))
        r = numpy.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2).astype(numpy.int64).ravel()
        _radius_cache[N] = (r, numpy.bincount(r, minlength=center)[:center])
    r, counts = _radius_cache[N]

    sums = numpy.bincount(r, weights=intensity_3d.ravel().astype(numpy.float64), minlength=center)[:center]
    intensity_1d = sums / counts
    q = (numpy.arange(center) + 0.5) * 2 * numpy.pi / (N * voxel_size)
    if qmax:
        mask = (qmin <= q) & (q <= qmax)
        return q[mask], intensity_1d[mask]
    return q, intensity_1d


def scattering_curve_gpu(density, dv, rho_bulk=0.334, drho=0.0, shell=3.0, padding=3, qmax=0.7, threshold=None):
    """I(q) of one density with the solvent model."""
    contrast = contrast_map(density, dv, rho_bulk, drho, shell, threshold)
    I3d = compute_3d_intensity(contrast * numpy.float32(dv**3), padding)
    return spherical_average(I3d, dv, 0.0, qmax)
