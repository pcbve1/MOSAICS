"""Manager class for running MOSAICS."""

import warnings
from typing import Literal, Union
import sys
import mmdf
import mrcfile
import pandas as pd
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
from .mosaics_result import AlternateTemplateResult, MosaicsResult
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
        particle_images: torch.Tensor,
        rot_mat: torch.Tensor,
        projective_filters: torch.Tensor,
        default_volume: torch.Tensor,
        atom_indices: torch.Tensor,
        device: torch.device,
        batch_size: int = 2048,
    ) -> torch.Tensor:
        """Inner loop function for running the MOSAICS program.

        Parameters
        ----------
        particle_images : torch.Tensor
            Pre-processed and normalized particle images *in real space*.
        rot_mat : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        default_volume : torch.Tensor
            The default (full-length) volume to compare against.
        atom_indices : torch.Tensor
            Which atoms should be removed from the template for the alternate model.
        device : torch.device
            The device to use for the computation. Should be either a CPU or GPU device.
        batch_size : int, optional
            The batch size to use for the cross-correlation calculations. Default is
            2048.
        """
        alternate_volume = self.simulator.run(
            device=str(device), atom_indices=atom_indices
        )

        # Subtract the alternate_volume from the default_volume if
        # self.sim_removed_atoms_only is set.
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
            particle_stack_images=particle_images,
            template_dft=alternate_volume_dft,
            rotation_matrices=rot_mat,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )
        

        return alt_cc

    def _create_data_cache(self, device: torch.device, cache_file: str) -> None:
        """Sets up the MOSAICS variables and saves them to a cached .npz file."""
        (
            rot_mat,
            projective_filters,
            particle_images,
            default_template,
            default_template_dft,
        ) = self._setup_mosaics_variables(device=device)

        # Pass all tensors onto the CPU and then to numpy arrays
        rot_mat = rot_mat.cpu().numpy()
        projective_filters = projective_filters.cpu().numpy()
        particle_images = particle_images.cpu().numpy()
        default_template = default_template.cpu().numpy()
        default_template_dft = default_template_dft.cpu().numpy()

        np.savez(
            cache_file,
            rot_mat=rot_mat,
            projective_filters=projective_filters,
            particle_images=particle_images,
            default_template=default_template,
            default_template_dft=default_template_dft,
        )

    def _load_mosaics_variables_from_cache(
        self, device: torch.device, cache_file: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loads the MOSAICS variables from a cached .npz file."""
        cache_data = np.load(cache_file)

        rot_mat = torch.from_numpy(cache_data["rot_mat"]).to(device)
        projective_filters = torch.from_numpy(cache_data["projective_filters"]).to(
            device
        )
        particle_images = torch.from_numpy(cache_data["particle_images"]).to(device)
        default_template = torch.from_numpy(cache_data["default_template"]).to(device)
        default_template_dft = torch.from_numpy(cache_data["default_template_dft"]).to(
            device
        )

        return (
            rot_mat,
            projective_filters,
            particle_images,
            default_template,
            default_template_dft,
        )

    def _setup_mosaics_variables(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up necessary variables for running MOSAICS.

        Parameters
        ----------
        device : torch.device
            The device to use for the computation. Can only be a single device.

        Returns
        -------
        rot_mat : torch.Tensor
            The rotation matrices for the orientations of each particle.
        projective_filters : torch.Tensor
            The projection filters for each particle image.
        particle_images : torch.Tensor
            Pre-filtered images of particles in real-space.
        default_template : torch.Tensor
            The default (full-length) volume to compare against.
        default_template_dft : torch.Tensor
            The DFT of the default (full-length) volume.
        """
        # Simulate the default (full-length) template volume
        default_template = self.simulator.run(
            atom_indices=self.template_iterator.get_default_template_idxs(), device=str(device))  # will need to replace this!
        
        # Use the built-in processing functionality from Leopard-EM to compute
        # the filtered particle images (in Fourier space) and the projective filters.
        particle_images_dft, default_template_dft, projective_filters = (
            setup_images_filters_particle_stack(
                self.particle_stack, self.preprocessing_filters, default_template
            )
        )
        particle_images = torch.fft.irfftn(particle_images_dft, dim=(-2, -1))
        particle_images *= torch.numel(particle_images[0]) ** 0.5  # norm again

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
        particle_images = particle_images.to(device)
        default_template_dft = default_template_dft.to(device)

        return (
            rot_mat,
            projective_filters,
            particle_images,
            default_template,
            default_template_dft,
        )

    def run_mosaics(
        self,
        gpu_id: Union[Literal["cpu"], int],
        batch_size: int = 2048,
        use_cache_file: Union[None, str] = None,
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
        use_cache_file : Union[None, str], optional
            If a string is provided, then the path is assumed to contain a .npz file
            named 'mosaics_cache_{timestamp}.npz' which contains pre-computed mosaics
            variables from the function _setup_mosaics_variables(). Useful when many
            experiments are being run in sequence. Default is None, which means no cache
            file is used.
        """
        if gpu_id == "cpu":
            device = torch.device("cpu")
        elif isinstance(gpu_id, int) and gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. Must be 'cpu' or a non-negative integer."
            )
        print("simulating default template...")
        if use_cache_file is not None:
            (
                rot_mat,
                projective_filters,
                particle_images,
                default_template,
                default_template_dft,
            ) = self._load_mosaics_variables_from_cache(device, use_cache_file)
        else:
            (
                rot_mat,
                projective_filters,
                particle_images,
                default_template,
                default_template_dft,
            ) = self._setup_mosaics_variables(device=device)
        #####################################################
        ### 1. Calculate default (full length) cross corr ###
        #####################################################
        print("cross-correlating default template")

        default_cc = cross_correlate_particle_stack(
            particle_stack_images=particle_images,
            template_dft=default_template_dft,
            rotation_matrices=rot_mat,
            projective_filters=projective_filters,
            batch_size=batch_size,
        )
        default_sc_pot = self.template_iterator.get_template_scattering_potential(None)  

        ######################################################
        ### 2. Iteration over alternate (truncated) models ###
        ######################################################

        # NOTE: When the inverted flag is set to True, the iterator will return the
        # indices of the atoms that should NOT be removed. This is opposite of the
        # the 'sim_removed_atoms_only' flag.
        inverted = not self.sim_removed_atoms_only
        print('Working on alternate templates')
        num_iters = self.template_iterator.num_alternate_structures
        alt_template_iter = self.template_iterator.alternate_template_iter(inverted)

        alternate_template_results = []
        for chains, residues, atom_indices in tqdm.tqdm(
            alt_template_iter,
            total=num_iters,
            desc="Iterating over alternate models",
        ):
            if len(atom_indices) == 0:
                warnings.warn(
                    "No atoms to remove for this iteration. Skipping calculation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alt_cc = default_cc
                alt_sc_pot = default_sc_pot

            else:
                # be able to change the pixel size here in the inner loop. 
                alt_cc = self._mosaics_inner_loop(
                    particle_images=particle_images,
                    rot_mat=rot_mat,
                    projective_filters=projective_filters,
                    default_volume=default_template,
                    atom_indices=atom_indices,
                    device=device,
                )
            alt_cc = alt_cc.cpu().numpy()
            alt_sc_pot = self.template_iterator.get_template_scattering_potential(
                atom_indices
            )

            res = AlternateTemplateResult(
                cross_correlation=alt_cc,
                chain_ids=chains,
                residue_ids=residues,
                removed_atom_indices=atom_indices.cpu().numpy(),
                sim_removed_atoms_only=self.sim_removed_atoms_only,
                scattering_potential_full_length=default_sc_pot,
                scattering_potential_alternate=alt_sc_pot,
            )
            alternate_template_results.append(res)

        return MosaicsResult(
            default_cross_correlation=default_cc.cpu().numpy(),
            template_iterator_config=self.template_iterator.model_dump(),
            sim_removed_atoms_only=self.sim_removed_atoms_only,
            alternate_template_results=alternate_template_results,
        )
