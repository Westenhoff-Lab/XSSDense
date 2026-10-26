import cupy
from cupyx.scipy import ndimage as cp_ndimage


def contrast_map_gpu(density, dv, rho_bulk=0.334, drho=0.0, shell=3.0, threshold=0.4):
    """Density minus the displaced bulk solvent, plus the hydration shell contrast.

    envelope = voxels with density > threshold (the protein)
    shell    = voxels outside the envelope within `shell` Å of it
    In vacuum (rho_bulk = 0, drho = 0) this is just the density itself.
    """
    density = cupy.asarray(density, dtype=cupy.float32)
    envelope = density > threshold
    dist = cp_ndimage.distance_transform_edt(~envelope, sampling=dv)
    shell_mask = (dist <= shell) & ~envelope
    return density - rho_bulk * envelope + drho * shell_mask


def scattering_curve_gpu(density, dv, rho_bulk=0.334, drho=0.0, shell=3.0, padding=3, qmax=0.7):
    """I(q) of one density, including the solvent model."""
    contrast = contrast_map_gpu(density, dv, rho_bulk, drho, shell)
    I3d = compute_3d_intensity_gpu(contrast * dv**3, padding)
    return spherical_average_gpu(I3d, voxel_size=dv, qmin=0.0, qmax=qmax)


def compute_3d_intensity_gpu(density_grid, padding_factor=3):
    N = density_grid.shape[0]
    Npad = N * (padding_factor - 1) // 2
    padded_density_grid = cupy.pad(density_grid, pad_width=Npad, mode="constant", constant_values=0)
    amplitude = cupy.fft.fftshift(cupy.fft.fftn(padded_density_grid))
    intensity_3d = cupy.abs(amplitude) ** 2
    return intensity_3d


def spherical_average_gpu(intensity_3d, voxel_size, qmin=0.0, qmax=None):
    N = intensity_3d.shape[0]
    center = N // 2
    z, y, x = cupy.indices(intensity_3d.shape)
    r = cupy.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
    r_int = r.astype(cupy.int32)
    radial_bins = cupy.arange(0, center)
    dq = 2 * cupy.pi / (N * voxel_size)
    q = (radial_bins + 0.5) * dq
    intensity_1d = cp_ndimage.mean(intensity_3d, labels=r_int, index=radial_bins)
    if qmax:
        mask = (qmin <= q) & (q <= qmax)
        return q[mask], intensity_1d[mask]
    return q, intensity_1d
