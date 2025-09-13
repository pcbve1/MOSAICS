"""Manager class for running MOSAICS."""

from typing import Literal, Union

import mmdf
import numpy as np
import roma
import torch
import tqdm
import yaml  # type: ignore
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    calculate_ctf_filter_stack_full_args,
    setup_images_filters_particle_stack,
)
from pydantic import BaseModel
from ttsim3d.models import Simulator

from .cross_correlation_core import cross_correlate_particle_stack
from .mosaics_result import MosaicsResult
from .template_iterator import BaseTemplateIterator, instantiate_template_iterator


class MosaicsManager(BaseModel):
    """Class for importing, running, and exporting MOSAICS program data.

    Attributes
    ----------
    particle_stack : ParticleStack
        Stack of particle images with associated metadata (orientation, position,
        defocus) necessary for template matching.
    simulator : Simulator
        Instance of Simulator model from ttsim3d package. Holds the pdb file and
        associated atom positions, bfactors, etc. for simulating a 3D volume.
    template_iterator : BaseTemplateIterator
        Iteration configuration model for describing how to iterate over the reference
        structure. Should be an instance of a subclass of BaseTemplateIterator.
    preprocessing_filters : PreprocessingFilters
        Configuration for the pre-processing filters to apply to the particle images.
    sim_removed_atoms_only : bool
        When True, only re-simulate the removed atoms from the alternate template and
        subtract the alternate volume from the default volume. When False, simulate the
        entire alternate template and subtract the alternate volume from the default.
        Simulating only the removed atoms is generally faster. Default is True.
    """

    particle_stack: ParticleStack  # comes from Leopard-EM
    simulator: Simulator  # comes from ttsim3d
    template_iterator: BaseTemplateIterator
    preprocessing_filters: PreprocessingFilters  # comes from Leopard-EM
    sim_removed_atoms_only: bool = True

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
        data["template_iterator"]["structure_df"] = pdb_df

        # Create the template iterator using the factory method
        template_iterator = instantiate_template_iterator(data["template_iterator"])
        data["template_iterator"] = template_iterator

        return cls(**data)

    def _mosaics_inner_loop(
        self,
        particle_images_dft: torch.Tensor,
        rot_mat: torch.Tensor,
        projective_filters: torch.Tensor,
        default_volume: torch.Tensor,
        atom_indices: torch.Tensor,
        gpu_id: Union[Literal["cpu"], int],
        batch_size: int = 2048,
    ) -> torch.Tensor:
        """Inner loop function for running the MOSAICS program.

        Parameters
        ----------
        particle_images_dft : torch.Tensor
            The DFT of the particle images. Pre-processed and normalized.
        rot_mat : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        default_volume : torch.Tensor
            The default (full-length) volume to compare against.
        atom_indices : torch.Tensor
            Which atoms should be removed from the template for the alternate model.
        gpu_id : int
            The GPU ID to use for the calculations. Should either be "cpu" or a
            non-negative integer.
        batch_size : int, optional
            The batch size to use for the cross-correlation calculations. Default is
            2048.
        """
        alternate_volume = self.simulator.run(device=gpu_id, atom_indices=atom_indices)

        # Subtract the alternate_volume from the default_volume if
        # self.sim_only_removed_atoms is set.
        # This is because when inverted, then only the atoms which should be
        # removed get simulated rather than the atoms which should be kept.
        if self.sim_removed_atoms_only:
            alternate_volume = default_volume - alternate_volume

        alternate_volume = torch.fft.fftshift(alternate_volume)
        alternate_volume_dft = torch.fft.rfftn(alternate_volume, dim=(-3, -2, -1))
        alternate_volume_dft = torch.fft.fftshift(alternate_volume_dft, dim=(-3, -2))

        # Recalculate the cross-correlation with the alternate model
        # and take the maximum value over space
        alt_cc = cross_correlate_particle_stack(
            particle_stack_dft=particle_images_dft,
            template_dft=alternate_volume_dft,
            rotation_matrices=rot_mat,
            projective_filters=projective_filters,
            mode="valid",
            batch_size=batch_size,
        )
        alt_cc = torch.max(alt_cc.view(particle_images_dft.shape[0], -1), dim=-1).values

        return alt_cc

    def run_mosaics(
        self, gpu_id: Union[Literal["cpu"], int], batch_size: int = 2048
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
        """
        if gpu_id == "cpu":
            device = torch.device("cpu")
        elif isinstance(gpu_id, int) and gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. Must be 'cpu' or a non-negative integer."
            )

        ################################################################
        ### 0. Do necessary data extraction and pre-processing steps ###
        ################################################################

        # Simulate the default (full-length) template volume
        default_template = self.simulator.run(device=gpu_id)

        # Use the built-in processing functionality from Leopard-EM to compute
        # the filtered particle images (in Fourier space) and the projective filters.
        particle_images_dft, default_template_dft, projective_filters = (
            setup_images_filters_particle_stack(
                self.particle_stack, self.preprocessing_filters, default_template
            )
        )

        # Calculate the per-particle CTF array and combine it with projective filters
        defocus_u, defocus_v = self.particle_stack.get_absolute_defocus()
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])
        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            self.particle_stack,
            (default_template.shape[-2], default_template.shape[-1]),
        )
        ctf_filters = calculate_ctf_filter_stack_full_args(
            defocus_u=defocus_u,  # in Angstrom
            defocus_v=defocus_v,  # in Angstrom
            astigmatism_angle=defocus_angle,  # in degrees
            defocus_offsets=torch.Tensor([0.0]),
            pixel_size_offsets=torch.Tensor([0.0]),
            **ctf_kwargs,
        )
        ctf_filters.squeeze_(0)  # remove defocus offset dim
        projective_filters = projective_filters * ctf_filters

        # Grab the per-particle orientations preferring refined angles (if they exist)
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles=True)
        rot_mat = roma.euler_to_rotmat("ZYZ", euler_angles, degrees=True)
        rot_mat = rot_mat.float()

        # Pass tensors to device
        rot_mat = rot_mat.to(device)
        projective_filters = projective_filters.to(device)
        particle_images_dft = particle_images_dft.to(device)
        default_template_dft = default_template_dft.to(device)

        #####################################################
        ### 1. Calculate default (full length) cross corr ###
        #####################################################

        default_cc = cross_correlate_particle_stack(
            particle_stack_dft=particle_images_dft,
            template_dft=default_template_dft,
            rotation_matrices=rot_mat,
            projective_filters=projective_filters,
            mode="valid",
            batch_size=batch_size,
        )
        default_cc = torch.max(default_cc.view(default_cc.shape[0], -1), dim=-1).values

        ######################################################
        ### 2. Iteration over alternate (truncated) models ###
        ######################################################

        # The chains and residues removed for each alternate mode (for metadata)
        # Also used to infer the number of iterations for the tqdm object.
        if self.template_iterator.type != "random":
            chain_residue_iterator = self.template_iterator.chain_residue_iter()
            alternate_chain_residue_pairs = [
                list(zip(*pair)) for pair in list(chain_residue_iterator)
            ]
            num_iters = len(alternate_chain_residue_pairs)
        else:
            alternate_chain_residue_pairs = []
            num_iters = self.template_iterator.num_alternate_structures

        # NOTE: When the inverted flag is set to True, the iterator will return the
        # indices of the atoms that should NOT be removed. This is opposite of the
        # the 'sim_removed_atoms_only' flag.
        inverted = not self.sim_removed_atoms_only
        atom_idx_iterator = self.template_iterator.atom_idx_iter(inverted=inverted)

        alternate_ccs = []
        for atom_indices in tqdm.tqdm(
            atom_idx_iterator,
            total=num_iters,
            desc="Iterating over alternate models",
        ):
            alt_cc = self._mosaics_inner_loop(
                particle_images_dft=particle_images_dft,
                rot_mat=rot_mat,
                projective_filters=projective_filters,
                default_volume=default_template,
                atom_indices=atom_indices,
                gpu_id=gpu_id,
            )
            alternate_ccs.append(alt_cc)

        # Post-hoc calculate the relative removed from each alternate model
        # for the mass adjustment factor
        default_mass = self.template_iterator.get_default_template_mass()
        alternate_masses = self.template_iterator.get_alternate_template_mass()
        alternate_masses = np.array(alternate_masses)

        mass_adjustment_factors = (default_mass - alternate_masses) / default_mass  # type: ignore

        # Stack the alternate cross-correlation values into a single tensor
        alternate_ccs = torch.stack(alternate_ccs, dim=0)

        # Create the metadata for the alternate chain residues
        alternate_chain_residue_metadata = {
            f"alt_cc_{i}": chain_residue_pairs
            for i, chain_residue_pairs in enumerate(alternate_chain_residue_pairs)
        }

        return MosaicsResult(
            default_cross_correlation=default_cc.cpu().numpy(),
            alternate_cross_correlations=alternate_ccs.cpu().numpy(),  # type: ignore
            mass_adjustment_factors=mass_adjustment_factors,
            template_iterator_type=self.template_iterator.type,
            alternate_chain_residue_metadata=alternate_chain_residue_metadata,
            sim_removed_atoms_only=self.sim_removed_atoms_only,
        )
