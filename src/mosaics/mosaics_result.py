"""Class for storing the results from a MOSAICS run."""

from typing import Annotated, Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


def nd_array_before_validator(v: Any) -> np.ndarray:
    """Validator to ensure the value is a numpy array."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def nd_array_serializer(v: np.ndarray) -> list:
    """Serializer to convert numpy array to list."""
    return v.tolist()  # type: ignore


NDArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_before_validator),
    PlainSerializer(nd_array_serializer, return_type=list),
]


class AlternateTemplateResult(BaseModel):
    """Class for storing the result for a single alternate template run in MOSAICS.

    Attributes
    ----------
    cross_correlation : np.ndarray
        The cross-correlation values for the alternate (truncated) model for each
        particle in the dataset.
    chain_ids : list[Optional[str]]
        The chain IDs for each residue in the model. This list is index aligned with
        'residue_ids'.
    residue_ids : list[Optional[int]]
        The residue IDs for each residue in the model. This list is index aligned with
        'chain_ids'.
    removed_atom_indices : list[int]
        The atom indices that were removed from the full model to create the alternate
        (truncated) model.
    sim_removed_atoms_only : bool
        Whether only the removed residues were used for the cross-correlation
        calculation. Default is False which means the entire model (minus the removed
        atoms from the corresponding residues) were used in the calculation. If True,
        only the removed atoms were used in the calculation.
    scattering_potential_full_length : float
        The total scattering potential of the full-length model.
    scattering_potential_alternate : float
        The total scattering potential of the alternate (truncated) model. If
        'sim_removed_atoms_only' is True, this will be the scattering potential
        of the removed atoms only. Otherwise, it will be the scattering potential of the
        full model minus the removed atoms.
    added_chains : bool
        Whether added chains are being simulated or removed. 

    Methods
    -------
    expected_correlation_decrease() -> float
        Returns the expected relative decrease in cross-correlation due to the change
        in scattering potential.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cross_correlation: NDArray
    chain_ids: list[Optional[str]]  # index aligned with residue_ids
    residue_ids: list[Optional[int]]  # index aligned with chain_ids
    removed_atom_indices: NDArray
    sim_removed_atoms_only: bool
    scattering_potential_full_length: float
    scattering_potential_alternate: float
    added_chains: bool

    def expected_correlation_decrease(self) -> float:
        """Relative decrease in cross-corr due to change in scattering potential."""
        if self.sim_removed_atoms_only:
            delta = self.scattering_potential_alternate
        else:
            delta = (
                self.scattering_potential_full_length 
                - self.scattering_potential_alternate
            )
        return delta / self.scattering_potential_full_length


class MosaicsResult(BaseModel):
    """Class for storing the results from a MOSAICS run.

    Attributes
    ----------
    default_cross_correlation : np.ndarray
        The default (non-truncated model) cross-correlation values.
    template_iterator_config : dict[str, Any]
        The configuration of the template iterator used for the alternate models.
    sim_removed_atoms_only : bool
        Whether only the removed residues were used for the cross-correlation
        calculation.
    alternate_template_results : list[AlternateTemplateResult]
        The results for each of the alternate template runs.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    default_cross_correlation: NDArray
    template_iterator_config: dict[str, Any]
    sim_removed_atoms_only: bool
    alternate_template_results: list[AlternateTemplateResult]

    @property
    def num_particles(self) -> int:
        """Returns the number of particles in the dataset."""
        return len(self.default_cross_correlation)

    @property
    def num_alternate_models(self) -> int:
        """Returns the number of alternate (truncated) models used."""
        return len(self.alternate_template_results)

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

    def expected_correlation_decreases(self) -> np.ndarray:
        """Returns the expected corr decrease for each alternate model."""
        res = []
        for alt_result in self.alternate_template_results:
            res.append(alt_result.expected_correlation_decrease())

        return np.array(res)

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
        adjustment_factors = self.expected_correlation_decreases()

        
        if self.alternate_template_results[0].added_chains:
            scores= (alt_cc / adjustment_factors) / default_cc[:, None]
        else:
            scores = (alt_cc / adjustment_factors) / default_cc[:, None]

        return scores
