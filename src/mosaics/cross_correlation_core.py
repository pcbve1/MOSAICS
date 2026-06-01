"""Cross-correlation functions for the MOSAICS package."""

import torch

# from leopard_em.backend.utils import normalize_template_projections
from torch_fourier_slice import extract_central_slices_rfft_3d


def _cross_correlate_different_size(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    fourier_slice: torch.Tensor,  # (N, h, w)
) -> torch.Tensor:  # (N, H-h+1, W-w+1)
    """Cross-correlate a stack of particle images against a template when the template is smaller than the images."""
    image_h, image_w = particle_stack_images.shape[-2:]
    template_h, template_w = fourier_slice.shape[-2], fourier_slice.shape[-1] * 2 - 2
    valid_slice = slice(0, image_h - template_h + 1), slice(0, image_w - template_w + 1)

    projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))

    particle_stack_images_fft = torch.fft.rfftn(particle_stack_images, dim=(-2, -1))
    fourier_slice = torch.fft.rfftn(projections, dim=(-2, -1), s=(image_h, image_w))

    cross_correlation_fft = particle_stack_images_fft * fourier_slice.conj()
    cross_correlation = torch.fft.irfftn(cross_correlation_fft, dim=(-2, -1))

    return cross_correlation[:, valid_slice[0], valid_slice[1]]


# pylint: disable=too-many-locals
def cross_correlate_particle_stack(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w)
    perfect_projection_images: torch.Tensor | None = None,  # (N, h, w)
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Cross-correlate a stack of particle images against a template.

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
    perfect_projection_images : torch.Tensor | None, optional
        Expected perfect projection images for each particle. If provided, used to
        calculate the overlap between teh new template and the perfect projection.
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

    # Can use faster approach if height and width are the same
    _is_same_size = (image_h == template_h) and (image_w == template_w)

    if batch_size == -1:
        batch_size = num_particles

    if _is_same_size:
        out_correlation = torch.zeros(num_particles, device=device)
        out_correlation_perfect = (
            torch.zeros(num_particles, device=device)
            if perfect_projection_images is not None
            else None
        )
    else:
        out_correlation = torch.zeros(
            (num_particles, image_h - template_h + 1, image_w - template_w + 1),
            device=device,
        )
        out_correlation_perfect = (
            torch.zeros(
                (num_particles, image_h - template_h + 1, image_w - template_w + 1),
                device=device,
            )
            if perfect_projection_images is not None
            else None
        )

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        batch_particles_images = particle_stack_images[batch_slice]
        batch_rotation_matrices = rotation_matrices[batch_slice]
        batch_projective_filters = projective_filters[batch_slice]

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            rotation_matrices=batch_rotation_matrices,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast
        fourier_slice *= batch_projective_filters

        if _is_same_size:
            projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
            projections = torch.fft.ifftshift(projections, dim=(-2, -1))
            tmp = torch.sum(batch_particles_images * projections, dim=(-2, -1))
            if perfect_projection_images is not None:
                perfect_projections = perfect_projection_images[batch_slice]
                tmp_perfect = torch.sum(perfect_projections * projections, dim=(-2, -1))
        else:
            tmp = _cross_correlate_different_size(batch_particles_images, fourier_slice)
            if perfect_projection_images is not None:
                perfect_projections = perfect_projection_images[batch_slice]
                tmp_perfect = _cross_correlate_different_size(
                    perfect_projections, fourier_slice
                )

        out_correlation[batch_slice] = tmp
        if perfect_projection_images is not None:
            out_correlation_perfect[batch_slice] = tmp_perfect

    return out_correlation, out_correlation_perfect
