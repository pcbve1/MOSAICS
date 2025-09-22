"""Cross-correlation functions for the MOSAICS package."""

from typing import Literal

import torch
from leopard_em.backend.utils import normalize_template_projection
from torch_fourier_slice import extract_central_slices_rfft_3d


def handle_correlation_mode(
    cross_correlation: torch.Tensor,
    out_shape: tuple[int, ...],
    mode: Literal["valid", "same"],
) -> torch.Tensor:
    """Handle cropping for cross correlation mode.

     NOTE: 'full' mode is not implemented.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        The cross correlation result.
    out_shape : tuple[int, ...]
        The desired shape of the output.
    mode : Literal["valid", "same"]
        The mode of the cross correlation. Either 'valid' or 'same'. See
        [numpy.correlate](https://numpy.org/doc/stable/reference/generated/
        numpy.convolve.html#numpy.convolve)
        for more details.
    """
    # Crop the result to the valid bounds
    if mode == "valid":
        slices = [slice(0, _out_s) for _out_s in out_shape]
        cross_correlation = cross_correlation[slices]
    elif mode == "same":
        pass
    elif mode == "full":
        raise NotImplementedError("Full mode not supported")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return cross_correlation


# pylint: disable=too-many-locals
def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against a template.

    This approach is more efficient than using the `core_refine_template` function from
    Leopard-EM because: 1) The orientation and Fourier filters do not change over the
    search space, 2) The reference template (from which projections are generated form)
    _is_ changing over the search space, and 3) The same reference template is being
    used for all particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    rotation_matrices : torch.Tensor
        The orientations of the particles to take the Fourier slices of, as a long
        list of rotation matrices. Shape of (N, 3, 3).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    mode : Literal["valid", "same"], optional
        Correlation mode to use, by default "valid". If "valid", the output will be
        the valid cross-correlation of the inputs. If "same", the output will be the
        same shape as the input particle stack.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used. If "valid", the output will be (N, H-h+1, W-w+1). If "same",
        the output will be (N, H, W).

    Raises
    ------
    ValueError
        If the mode is not "valid" or "same".
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, image_h, image_w = particle_stack_dft.shape
    _, template_h, template_w = template_dft.shape
    # account for RFFT
    image_w = 2 * (image_w - 1)
    template_w = 2 * (template_w - 1)

    if batch_size == -1:
        batch_size = num_particles

    if mode == "valid":
        output_shape = (
            num_particles,
            image_h - template_h + 1,
            image_w - template_w + 1,
        )
    elif mode == "same":
        output_shape = (num_particles, image_h, image_w)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'valid' or 'same'.")

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

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

        # Inverse Fourier transform and normalize the projection
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))

        # NOTE: Turning off template normalization for MOSAICS
        # projections = normalize_template_projection(
        #     projections, (template_h, template_w), (image_h, image_w)
        # )

        # Scale by the number of pixels
        projections *= (template_h * template_w) ** 0.5

        # Padded forward FFT and cross-correlate
        projections_dft = torch.fft.rfftn(
            projections, dim=(-2, -1), s=(image_h, image_w)
        )
        projections_dft = batch_particles_dft * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Handle the output shape
        cross_correlation = handle_correlation_mode(
            cross_correlation, output_shape, mode
        )

        out_correlation[i : i + batch_size] = cross_correlation

    return out_correlation
