"""Manager class for running MOSAICS."""

import warnings
from typing import Literal, Union
import mmdf
import numpy as np
import roma
import torch
import tqdm
import yaml  # type: ignore
from pydantic import BaseModel
from ttsim3d.models import Simulator
from torch_fourier_slice import project_3d_to_2d

from .cross_correlation_core import cross_correlate_particle_stack
from .mosaics_dataset import MosaicsDataset
from .mosaics_result import AlternateTemplateResult, MosaicsResult
from .template_iterator import BaseTemplateIterator, instantiate_template_iterator


SAME_BOX_SIZE_WARNING = (
    "The extracted box size is the as projection size, but requested particle shifts "
    "to be determined. Cannot find shifts with the same sizes. Please set "
    "'extracted_box_size' to be ~15-20 pct larger than 'original_template_size'."
)

volume_output_dir = "/home/brose/lab_projects/MOSAICS_paper_2026/eEF2_eEF1A_discrimination_test/data/mosaics_tests"


def _center_images_by_correlations(
    cross_correlation: torch.Tensor,  # (N, H - h, W - w)
    particle_images: torch.Tensor,  # (N, H, W)
    original_template_size: tuple[int, int],  # (h, w)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center and crop particle images by maximum cross-correlation peak."""
    H, W = particle_images.shape[-2], particle_images.shape[-1]
    h, w = original_template_size

    cc = cross_correlation.view(cross_correlation.shape[0], -1)
    max_cc_values, max_cc_indices = torch.max(cc, dim=1)
    max_xy_indices = torch.unravel_index(max_cc_indices, cross_correlation.shape[1:])

    # Crop to same shape as projection around the peak
    centered_images = torch.zeros(
        (particle_images.shape[0], h, w), device=particle_images.device
    )

    ccg_center_y = (H - h) // 2
    ccg_center_x = (W - w) // 2
    for i in range(particle_images.shape[0]):
        x_pos, y_pos = max_xy_indices[0][i], max_xy_indices[1][i]

        shift_y = y_pos - ccg_center_y
        shift_x = x_pos - ccg_center_x

        y_start = ccg_center_y + shift_y
        y_end = y_start + h
        x_start = ccg_center_x + shift_x
        x_end = x_start + w

        centered_images[i] = particle_images[i, x_start:x_end, y_start:y_end]

    # Renormalize the centered images and correlation values based on new shape
    factor = ((h * w) / (H * W)) ** 0.5
    centered_images *= factor
    max_cc_values *= factor

    return max_cc_values, centered_images


def _get_full_length_projections(
    default_volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projective_filters: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Calculate the full-length projections for the default template."""
    full_length_projections = torch.zeros(
        (
            rotation_matrices.shape[0],
            default_volume.shape[-2],
            default_volume.shape[-1],
        ),
        device=default_volume.device,
    )

    for i in range(0, rotation_matrices.shape[0], batch_size):
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        batch_projections = project_3d_to_2d(
            volume=default_volume, rotation_matrices=batch_rotation_matrices
        )
        batch_projections_dft = torch.fft.rfftn(batch_projections, dim=(-2, -1))
        batch_projections_dft *= batch_projective_filters
        batch_projections_dft *= -1
        batch_projections = torch.fft.irfftn(batch_projections_dft, dim=(-2, -1))

        full_length_projections[i : i + batch_size] = batch_projections

    return full_length_projections


class MosaicsManager(BaseModel):
    """Class for importing, running, and exporting MOSAICS program data.

    Attributes
    ----------
    simulator : Simulator
        Instance of Simulator model from ttsim3d package. Holds the pdb file and
        associated atom positions, bfactors, etc. for simulating a 3D volume.
    template_iterator : BaseTemplateIterator
        Iteration configuration model for describing how to iterate over the reference
        structure. Should be an instance of a subclass of BaseTemplateIterator.
    dataset : MosaicsDataset
        Dataset object that handles processing, loading, and caching of input data.
    sim_removed_atoms_only : bool
        When True, only re-simulate the removed atoms from the alternate template and
        subtract the alternate volume from the default volume. When False, simulate the
        entire alternate template and subtract the alternate volume from the default.
        Simulating only the removed atoms is generally faster. Default is True.
    particle_positions_are_centered : bool
        If True, then assume that the extracted boxes of particle images coming from
        'dataset.particle_images' will produce correlogram peak at central pixel.
        If False (typical), then determine the (x, y) position of each correlogram peak
        before proceeding. Default is False. NOTE: When this is set to False,
        the extracted box size should be larger than the maximum expected particle
        shift
    """

    simulator: Simulator
    template_iterator: BaseTemplateIterator
    dataset: MosaicsDataset
    sim_removed_atoms_only: bool = True
    particle_positions_are_centered: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MosaicsManager":
        """Create a MosaicsManager instance from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML file containing the configuration for the MosaicsManager.

        Returns
        -------
        MosaicsManager
            Instance of MosaicsManager created from the YAML file.
        """
        with open(yaml_path) as yaml_f:
            data = yaml.load(yaml_f, Loader=yaml.SafeLoader)

        # Load the pdb file from the Simulator into a DataFrame
        pdb_df = mmdf.read(data["simulator"]["pdb_filepath"])
        # removes hydrogens from pdb_df if option is enabled
        if data['simulator'].get('remove_hydrogens', True):
            pdb_df = pdb_df[pdb_df['element'] != 'H'].reset_index()
        data["template_iterator"]["structure_df"] = pdb_df
        # Create the template iterator using the factory method
        template_iterator = instantiate_template_iterator(data["template_iterator"])
        data["template_iterator"] = template_iterator
        return cls(**data)

    def _mosaics_inner_loop(
        self,
        particle_images: torch.Tensor,
        full_length_projections: torch.Tensor,
        rotation_matrices: torch.Tensor,
        projective_filters: torch.Tensor,
        default_volume: torch.Tensor,
        atom_indices: torch.Tensor,
        device: torch.device,
        counter: int, 
        batch_size: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner loop function for running the MOSAICS program.

        Parameters
        ----------
        particle_images : torch.Tensor
            Pre-processed and normalized particle images *in real space*.
        full_length_projections : torch.Tensor
            The full-length projection images for comparison.
        rotation_matrices : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        default_volume : torch.Tensor
            The default (full-length) simulated structure volume to compare against.
        atom_indices : torch.Tensor
            Which atoms should be removed from the template for the alternate model.
        device : torch.device
            The device to use for the computation. Should be either a CPU or GPU device.
        batch_size : int, optional
            The batch size to use for the cross-correlation calculations. Default is
            2048.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - First element is cross-correlation values of the alternate model with
            particle images.
            - Second element is the correlation between the default model
            projections and alternate model projections (expected decrease).
        """
        alternate_volume = self.simulator.run(
            device=str(device), atom_indices=atom_indices
        )

        # Subtract the alternate_volume from the default_volume if
        # self.sim_removed_atoms_only is set.
        # This is because, when inverted, only the atoms which should be
        # removed get simulated rather than the atoms which should be kept.
        if self.sim_removed_atoms_only:
            alternate_volume = default_volume - alternate_volume

        alternate_volume = torch.fft.fftshift(alternate_volume)
        alternate_volume_dft = torch.fft.rfftn(alternate_volume, dim=(-3, -2, -1))
        alternate_volume_dft = torch.fft.fftshift(alternate_volume_dft, dim=(-3, -2))

        # Recalculate the cross-correlation with the alternate model
        # and take the maximum value over space
        alternate_cc, overlap_cc = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            perfect_projection_images=full_length_projections,
            template_dft=alternate_volume_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )
        

        return alternate_cc, overlap_cc

    def run_mosaics(
        self,
        gpu_id: Union[Literal["cpu"], int],
        batch_size: int = 2048,
        cache_path: Union[None, str] = None,
    ) -> MosaicsResult:
        """Run the MOSAICS program.

        Parameters
        ----------
        gpu_id : Union[Literal["cpu"], int]
            The GPU ID to use for the computation. Can either be the string "cpu" to
            use the CPU, or an integer specifying the GPU ID. All other values are
            invalid.
        batch_size : int, optional
            The batch size -- number of particle images to process at once -- to use
            for the cross-correlation calculations. The default is 2048.
        cache_path : Union[None, str], optional
            If a string is provided, then the path is assumed to contain a .npz file
            which contains pre-computed dataset variables. Useful when many experiments
            are being run in sequence. Default is None, which means no cache file is
            used.
        """
        if gpu_id == "cpu":
            device = torch.device("cpu")
        elif isinstance(gpu_id, int) and gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. Must be 'cpu' or a non-negative integer."
            )
        print("loading dataset...")
        print("Rpl38_version")

        # Load or process the dataset
        if cache_path is not None:
            self.dataset.load_from_cache(cache_path, device)
        else:
            self.dataset.process_inputs(device, self.simulator)
        print(self.simulator.pixel_spacing)

        # Extract processed variables from dataset
        rotation_matrices = self.dataset.rotation_matrices
        projective_filters = self.dataset.projective_filters
        particle_images = self.dataset.particle_images
        default_template = self.dataset.template_volume
        default_template_dft = self.dataset.template_volume_dft

        #####################################################################
        ### 1. Calculate default (full length) cross corr and projections ###
        #####################################################################
        print("calculating full-length projections...")
        full_length_projections = _get_full_length_projections(
            default_volume=default_template,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )
        print('Cross-correlating particle stack...')
        default_cc, _ = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            perfect_projection_images=None,  # Don't need auto-correlation
            template_dft=default_template_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        default_scattering_potential = (
            self.template_iterator.get_template_scattering_potential(None)
        )

        ######################################################
        ### 2. Determine the position of correlogram peaks ###
        ######################################################

        if not self.particle_positions_are_centered:
            _box_size_is_same = (
                self.dataset.particle_stack.extracted_box_size
                == self.dataset.particle_stack.original_template_size
            )

            if _box_size_is_same:
                warnings.warn(
                    warning_message=SAME_BOX_SIZE_WARNING,
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            else:
                orig_template_size = self.dataset.particle_stack.original_template_size
                _, particle_images = _center_images_by_correlations(
                    cross_correlation=default_cc,
                    particle_images=particle_images,
                    original_template_size=orig_template_size,
                )
        print("calculating overlap...")
        default_cc, default_overlap = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            perfect_projection_images=full_length_projections,
            template_dft=default_template_dft,
            rotation_matrices=rotation_matrices,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )

        ######################################################
        ### 3. Iteration over alternate (truncated) models ###
        ######################################################

        # NOTE: When the inverted flag is set to True, the iterator will return the
        # indices of the atoms that should NOT be removed. This is opposite of the
        # the 'sim_removed_atoms_only' flag.
        invert_iterator = not self.sim_removed_atoms_only
        alternate_template_iterator = self.template_iterator.alternate_template_iter(
            invert_iterator
        )
        num_alternate_structures = self.template_iterator.num_alternate_structures

        counter = 0
        alternate_template_results = []
        for chains, residues, atom_indices in tqdm.tqdm(
            alternate_template_iterator,
            total=num_alternate_structures,
            desc="Iterating over alternate models",
        ):
            if len(atom_indices) == 0:
                warnings.warn(
                    "No atoms to remove for this iteration. Skipping calculation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alternate_cc = default_cc
                alternate_overlap = torch.ones_like(default_cc)
                alternate_scattering_potential = default_scattering_potential

            else:
                alternate_cc, alternate_overlap = self._mosaics_inner_loop(
                    particle_images=particle_images,
                    full_length_projections=full_length_projections,
                    rotation_matrices=rotation_matrices,
                    projective_filters=projective_filters,
                    default_volume=default_template,
                    atom_indices=atom_indices,
                    device=device,
                    counter=counter,
                )

            alternate_cc = alternate_cc.cpu().numpy()
            alternate_overlap = alternate_overlap.cpu().numpy()
            alternate_scattering_potential = (
                self.template_iterator.get_template_scattering_potential(atom_indices)
            )

            this_result = AlternateTemplateResult(
                cross_correlation=alternate_cc,
                chain_ids=chains,
                residue_ids=residues,
                removed_atom_indices=atom_indices.cpu().numpy(),
                sim_removed_atoms_only=self.sim_removed_atoms_only,
                alternate_overlap=alternate_overlap,
                scattering_potential_full_length=default_scattering_potential,
                scattering_potential_alternate=alternate_scattering_potential,
            )
            alternate_template_results.append(this_result)
            counter += 1

        return MosaicsResult(
            default_cross_correlation=default_cc.cpu().numpy(),
            default_overlap=default_overlap.cpu().numpy(),
            template_iterator_config=self.template_iterator.model_dump(),
            sim_removed_atoms_only=self.sim_removed_atoms_only,
            alternate_template_results=alternate_template_results,
        )
