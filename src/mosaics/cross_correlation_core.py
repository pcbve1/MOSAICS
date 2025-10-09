"""Cross-correlation functions for the MOSAICS package."""

import torch

# from leopard_em.backend.utils import normalize_template_projections
from torch_fourier_slice import extract_central_slices_rfft_3d


# pylint: disable=too-many-locals
def cross_correlate_particle_stack(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (D, H, W)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, H, W)
    batch_size: int = 1024,
) -> torch.Tensor:  # (N, )
    """Cross-correlate a stack of particle images against a template.

    MOSAICS assumes that there is no dependence on the particle orientation, defocus, or
    (x, y) position for the cross-correlation operation; we are fixing these parameters
    and searching over a set of alternate templates. This means we don't need to compute
    the FFT-based cross-correlation and can instead use the Fourier slice directly to
    compute the cross-correlation (fewer FFTs)

    Parameters
    ----------
    particle_stack_images : torch.Tensor
        The stack of pre-filtered particle images with shape (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted with shape (D, H, W) in real-space (cubic).
    rotation_matrices : torch.Tensor
        The orientations of the particles to take the Fourier slices of, as a long
        list of rotation matrices. Shape of (N, 3, 3).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation values for each particle image. Shape of (N,).
    """
    # Helpful constants for later use
    device = particle_stack_images.device
    num_particles, image_h, image_w = particle_stack_images.shape
    _, template_h, template_w = template_dft.shape
    template_w = 2 * (template_w - 1)

    assert (
        image_h == template_h and image_w == template_w
    ), "Particle images and template must have the same height and width."

    if batch_size == -1:
        batch_size = num_particles

    out_correlation = torch.zeros(num_particles, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        batch_particles_images = particle_stack_images[batch_slice]
        batch_rotation_matrices = rotation_matrices[batch_slice]
        batch_projective_filters = projective_filters[batch_slice]

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(template_h,) * 3,
            rotation_matrices=batch_rotation_matrices,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast
        fourier_slice *= batch_projective_filters

        # Inverse Fourier transform
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))

        tmp = batch_particles_images * projections
        out_correlation[batch_slice] = torch.sum(tmp, dim=(-2, -1))

    return out_correlation
