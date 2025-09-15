"""Class for storing the results from a MOSAICS run."""

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class AlternateTemplateResult(BaseModel):
    """Class for storing the result for a single alternate template run in MOSAICS.

    Attributes
    ----------
    cross_correlation : np.ndarray
        The cross-correlation values for the alternate (truncated) model for each
        particle in the dataset.
    mass_adjustment_factor : float
        The mass adjustment factor, which is calculated as the ratio of the mass of the
        truncated model to the mass of the default model.
    chain_residue_pairs : list[tuple[str, int]]
        The metadata for the chain residues that were removed to create the alternate
        model. Each entry in the list is a tuple containing the (chain, residue_id)
        pair that was removed
    removed_atom_indices : np.ndarray
        The indices of the atoms that were removed to create the alternate model.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cross_correlation: list[float]
    alternate_structure_mass: float
    mass_adjustment_factor: float
    chain_ids: list[Optional[str]]
    residue_ids: list[Optional[int]]
    removed_atom_indices: list[int]


class MosaicsResult(BaseModel):
    """Class for storing the results from a MOSAICS run.

    Attributes
    ----------
    num_particles : int
        The number of particles in the dataset.
    template_iterator_type : str
        The type of template iterator used for the alternate models.
    sim_removed_atoms_only : bool
        Whether only the removed residues were used for the cross-correlation
        calculation. Default is False which means the entire model (minus the removed
        atoms from the corresponding residues) were used in the calculation. If True,
        only the removed atoms were used in the calculation.
    default_cross_correlation : list[float]
        The default (non-truncated model) cross-correlation values.
    alternate_template_results : list[AlternateTemplateResult]
        The results for each of the alternate template runs.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    num_particles: int
    template_iterator_type: str
    sim_removed_atoms_only: bool = False
    default_cross_correlation: list[float]
    alternate_template_results: list[AlternateTemplateResult]

    def to_df(self, extra_columns: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """Containerizes the experiment results into a Pandas DataFrame.

        The returned DataFrame will have following columns:
        - particle_id: The particle ID, defaults to ['part_0', 'part_1', ...]
        - default_cc: The default cross-correlation value
        - alt_cc_0: The cross-correlation value for the first alternate model
        - alt_cc_1: The cross-correlation value for the second alternate model
        ...

        The 'extra_columns' parameter can be used to add additional columns to the
        DataFrame.

        Parameters
        ----------
        extra_columns : dict[str, Any], optional
            The extra columns to add to the DataFrame. Default is None and no extra
            columns will be added.

        Returns
        -------
            pd.DataFrame: The DataFrame containing the held data.
        """
        if extra_columns is None:
            extra_columns = {}

        # Figure out the dimensions of the data
        num_parts = self.num_particles
        num_alts = len(self.alternate_template_results)

        # Create the base dataframe with particle IDs and default cross-correlations
        particle_id = [f"part_{i}" for i in range(num_parts)]
        default_cc = self.default_cross_correlation
        df = pd.DataFrame(
            {"particle_id": particle_id, **extra_columns, "default_cc": default_cc}
        )

        # Add the alternate cross-correlations values to the dataframe
        for i, alt_result in enumerate(self.alternate_template_results):
            df[f"alt_cc_{i}"] = alt_result.cross_correlation

        return df

    def as_ndarrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the default and alternate cross-correlation values as numpy arrays.

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: The tuple containing the default and
            alternate cross-correlation values as numpy arrays. The firs array is of
            shape (num_particles,) and the second array is of shape
            (num_particles, num_alternate_models).
        """
        default_cc = np.array(self.default_cross_correlation)
        alt_cc = np.array(
            [
                alt_result.cross_correlation
                for alt_result in self.alternate_template_results
            ]
        ).T
        return default_cc, alt_cc
    
    def mosaics_scores(self) -> np.ndarray:
        """Calculates the MOSAICS scores for each particle and alternate model.

        The MOSAICS score is calculated as the ratio between the alternate model
        cross-correlations (adjusted by mass) and the default model cross-correlations.

        Returns
        -------
            np.ndarray: The MOSAICS scores of shape
            (num_particles, num_alternate_models).
        """
        default_cc, alt_cc = self.as_ndarrays()
        mass_adjustment_factors = np.array(
            [
                alt_result.mass_adjustment_factor
                for alt_result in self.alternate_template_results
            ]
        )

        scores = (alt_cc / mass_adjustment_factors) / default_cc[:, None]
        return scores
