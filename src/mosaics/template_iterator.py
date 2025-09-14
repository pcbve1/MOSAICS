"""Module for different ways of generating alternate templates for comparison."""

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

#########################################################
### Data for residue/atom identification in PDB files ###
#########################################################

# These are the default atoms to remove (under "atom" column in df loaded from PDB)
DEFAULT_AMINO_ACID_ATOMS = ["N", "CA", "C", "O"]
DEFAULT_RNA_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O4'"]
DEFAULT_DNA_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O4'"]

# Names in the "residue" column of the PDB file which correspond to the residue types
AMINO_ACID_RESIDUES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # Unknown residue
]
RNA_RESIDUES = ["A", "C", "G", "U", "N"]
DNA_RESIDUES = ["A", "C", "G", "T", "N"]

# All atoms for each corresponding residue type. These lists can be used to pass
# a string "all" to the atom_types parameter in the template iterator config to remove
# all atoms from the residue type.
ALL_AMINO_ACID_ATOMS = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CD",
    "CE",
    "NZ",
    "OG",
    "CD1",
    "CD2",
    "CE1",
    "CE2",
    "CZ",
    "OH",
    "NE",
    "NH1",
    "NH2",
    "OE1",
    "NE2",
    "OG1",
    "CG2",
    "OE2",
    "OD1",
    "OD2",
    "CG1",
    "ND1",
    "ND2",
    "SG",
    "NE1",
    "CE3",
    "CZ2",
    "CZ3",
    "CH2",
    "SD",
    "OXT",
]
ALL_RNA_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N1",
    "C2",
    "O2",
    "N3",
    "C4",
    "O4",
    "C5",
    "C6",
    "N9",
    "C8",
    "N7",
    "O6",
    "N2",
    "N6",
    "N4",
]
ALL_DNA_ATOMS = [
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
    "N1",
    "C2",
    "O2",
    "N3",
    "C4",
    "N4",
    "C5",
    "C6",
    "P",
    "OP1",
    "OP2",
    "N9",
    "C8",
    "N7",
    "O6",
    "N2",
    "N6",
    "O4",
    "C7",
]

# TODO: Make this a lookup table, move to a separate file, or use external package
MASS_DICT = {"H": 1.01, "C": 12.01, "N": 14.01, "O": 16.00, "S": 32.06, "P": 30.97}


###############################################
### Helper functions for template iterators ###
###############################################


def sliding_window_iterator(
    length: int, window_width: int, step_size: int
) -> Iterator[np.ndarray]:
    """Generator for a 1-dimensional sliding window of indexes.

    Each iteration yields a long tensor of indexes with the same length as the window
    width. The starting index is incremented by the step size each iteration.
    Note that the last window may be shorter than the window width.

    Parameters
    ----------
    length : int
        Length of the sequence to iterate over.
    window_width : int
        Width of the sliding window.
    step_size : int
        Step size for incrementing the window.

    Yields
    ------
    torch.Tensor
        Tensor of indexes for the current window.

    Example
    -------
    >>> for window in sliding_window_iterator(10, 3, 2):
    ...     print(window)
    [0 1 2]
    [2 3 4]
    [4 5 6]
    [6 7 8]
    [8 9]
    """
    for i in range(0, length - window_width + step_size, step_size):
        yield np.arange(i, min(i + window_width, length))


def instantiate_template_iterator(data: dict) -> "BaseTemplateIterator":
    """Factory function for instantiating a template iterator object."""
    iterator_type = data.pop("type", None)
    if iterator_type == "random":
        return RandomAtomTemplateIterator(**data)
    elif iterator_type == "random_residue":
        return RandomResidueTemplateIterator(**data)
    elif iterator_type == "chain":
        return ChainTemplateIterator(**data)
    elif iterator_type == "residue":
        return ResidueTemplateIterator(**data)
    else:
        raise ValueError(f"Invalid template iterator type: {iterator_type}")


class BaseTemplateIterator(BaseModel):
    """Base class for defining template iterator configurations.

    Attributes
    ----------
    num_residues_removed : int
        Number of residues to remove from the structure at each step.
    residue_increment : int
        Number of residues to increment the removal by each iteration.
    residue_types : list[str]
        Types of residues to target for removal. Options are 'amino_acid', 'rna', 'dna'.
    amino_acid_atoms : list[str], optional
        List of atom type labels (in the PDB file) to remove from amino acid residues.
        Default is ['N', 'CA', 'C', 'O'].
    rna_atoms : list[str], optional
        List of atom type labels (in the PDB file) to remove from RNA residues.
        Default is ['C1', 'C2', 'C3', 'C4', 'O4'].
    dna_atoms : list[str], optional
        List of atom type labels (in the PDB file) to remove from DNA residues.
        Default is ['C1', 'C2', 'C3', 'C4', 'O4'].
    structure_df : pd.DataFrame
        Underlying Pandas DataFrame for the PDB model.

    Methods
    -------
    chain_residue_iter()
        Iterator over the chain, residue pairs removed in each alternate template.
        Must be implemented by subclasses.
    atom_idx_iter(inverted=False)
        Generator for iterating over atom indexes to keep in each structure.
        Must be implemented by subclasses.
    get_default_template_mass()
        Get the mass (in amu) of the default template structure.
    get_alternate_template_mass()
        Get the mass (in amu) of all the alternate template structures as a list.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    type: ClassVar[Literal["random", "chain", "residue"]]

    residue_types: list[Literal["amino_acid", "rna", "dna"]]
    amino_acid_atoms: list[str] = DEFAULT_AMINO_ACID_ATOMS
    rna_atoms: list[str] = DEFAULT_RNA_ATOMS
    dna_atoms: list[str] = DEFAULT_DNA_ATOMS

    structure_df: pd.DataFrame  # NOTE: Comes from Simulator object

    @field_validator("residue_types")  # type: ignore
    def _validate_residue_types(cls, v):
        if not v:
            raise ValueError("At least one residue type must be specified.")
        return v

    @field_validator("amino_acid_atoms", mode="before")  # type: ignore
    def _validate_amino_acid_atoms(cls, v):
        """Validator to convert "all" to the full list of amino acid atoms."""
        if v == ["all"]:
            return ALL_AMINO_ACID_ATOMS
        return v

    @field_validator("rna_atoms", mode="before")  # type: ignore
    def _validate_rna_atoms(cls, v):
        """Validator to convert "all" to the full list of RNA atoms."""
        if v == ["all"]:
            return ALL_RNA_ATOMS
        return v

    @field_validator("dna_atoms", mode="before")  # type: ignore
    def _validate_dna_atoms(cls, v):
        """Validator to convert "all" to the full list of DNA atoms."""
        if v == ["all"]:
            return ALL_DNA_ATOMS
        return v

    @abstractmethod
    def chain_residue_iter(self) -> Iterator[tuple[list[str], list[int]]]:
        """Iterator over the chain, residue pairs removed in each alternate template.

        Yields
        ------
        tuple[list[str], list[int]]
            List of chains and list of residue ids removed at each iteration.
            The ordering of these lists correspond to each other, example
            (['A', 'A', 'B'], [1, 2, 3]) means residues 1 & 2 were removed from
            chain 'A' and the 3rd was removed from chain 'B'.
        """

    @abstractmethod
    def atom_idx_iter(self, inverted: bool = False) -> Iterator[torch.Tensor]:
        """Generator for iterating over atom indexes to keep in each structure.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        torch.Tensor
            Tensor of indexes for the atoms to remove.
        """
        raise NotImplementedError

    def get_default_template_mass(self) -> float:
        """Get the mass (in amu) of the default template structure."""
        total_mass = 0

        atom_counts = self.structure_df["element"].value_counts()
        for atom, count in atom_counts.items():
            if atom not in MASS_DICT:
                raise ValueError(f"Unknown atom type: {atom}")

            mass = MASS_DICT[atom]
            total_mass += mass * count

        return total_mass

    def get_alternate_template_mass(
        self, atom_idxs: np.ndarray | torch.Tensor
    ) -> float:
        """Get the mass (in amu) of all the alternate template structures."""
        atom_counts = self.structure_df.iloc[atom_idxs]["element"].value_counts()
        total_mass = 0
        for atom, count in atom_counts.items():
            if atom not in MASS_DICT:
                raise ValueError(f"Unknown atom type: {atom}")

            mass = MASS_DICT[atom]
            total_mass += mass * count

        return total_mass


class RandomAtomTemplateIterator(BaseTemplateIterator):
    """Template iterator for removing random atoms from a pdb structure.

    Attributes
    ----------
    type : Literal["random"]
        Discriminator field for differentiating between template iterator types.
    coherent_removal : bool
        If 'True', remove atoms from residues in order. For example, would remove atoms
        from residue [i, i+1, i+2, ...] rather than random indices. Default is 'True'.
    num_atoms_removed : int
        Number of atoms to remove from the structure at each iteration.
        Must be greater than 0.
    num_alternate_structures : int
        Number of alternate structures to generate by removing random atoms. Must be
        greater than 0.
    """

    type: ClassVar[Literal["random"]] = "random"
    coherent_removal: bool = True
    num_atoms_removed: Annotated[int, Field(gt=0)]
    num_alternate_structures: Annotated[int, Field(gt=0)]

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Add dummy column to keep track of the original indexes
        self.structure_df["original_index"] = self.structure_df.index

    def chain_residue_iter(self) -> Iterator[tuple[list[str], list[int]]]:
        """Generator for iteration of the chain, residue pairs in the structure."""
        # NOTE: This method returns empty lists since this class does not iterate
        # over chains or residues. It is only implemented to satisfy the abstract
        # base class requirement.
        for _ in range(self.num_alternate_structures):
            yield None

    def atom_idx_iter(self, inverted: bool = False) -> Iterator[torch.Tensor]:
        """Generator for iterating over atom indexes to keep in each structure.

        Since this class iterates over atoms in the structure (only the specified atom
        types) randomly, each iteration will yield atoms from multiple different chains.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        torch.Tensor
            Tensor of indexes for the atoms to remove.
        """
        keep_residues = []
        keep_atoms = []
        if "amino_acid" in self.residue_types:
            keep_residues.extend(AMINO_ACID_RESIDUES)
            keep_atoms.extend(self.amino_acid_atoms)
        if "rna" in self.residue_types:
            keep_residues.extend(RNA_RESIDUES)
            keep_atoms.extend(self.rna_atoms)
        if "dna" in self.residue_types:
            keep_residues.extend(DNA_RESIDUES)
            keep_atoms.extend(self.dna_atoms)

        # Subset the df to only residues that match the desired residue types
        # AND to keep only the desired atom types
        subset_df = self.structure_df[self.structure_df["residue"].isin(keep_residues)]
        subset_df = subset_df[subset_df["atom"].isin(keep_atoms)]

        for _ in range(self.num_alternate_structures):
            # If coherent removal, then choose a random starting index and remove a chunk
            if self.coherent_removal:
                start_idx = np.random.randint(
                    0, len(subset_df) - self.num_atoms_removed + 1
                )
                df_window = subset_df.iloc[
                    start_idx : start_idx + self.num_atoms_removed
                ]
            else:  # If not coherent removal, then choose random indexes to remove
                random_idxs = np.random.choice(
                    subset_df.index,
                    size=min(self.num_atoms_removed, len(subset_df)),
                    replace=False,
                )
                df_window = subset_df.loc[random_idxs]

            # Get the original indexes of the structure (rather than the window indexes)
            df_window = df_window.set_index("original_index")
            atom_idxs = df_window.index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield torch.tensor(atom_idxs)


class ChainTemplateIterator(BaseTemplateIterator):
    """Iterates over each chain in the structure and remove it in its entirety.

    NOTE: Parent class attributes from BaseTemplateIterator are *not* used besides the
    structure_df attribute. All atoms from a chain are completely removed.

    Attributes
    ----------
    type : Literal["chain"]
        Discriminator field for differentiating between template iterator types.
    """

    type: ClassVar[Literal["chain"]] = "chain"

    # Override the default atoms to include all atoms
    amino_acid_atoms: Annotated[list[str], Field(validate_default=True)] = ["all"]
    rna_atoms: Annotated[list[str], Field(validate_default=True)] = ["all"]
    dna_atoms: Annotated[list[str], Field(validate_default=True)] = ["all"]

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Add dummy column to keep track of the original indexes
        self.structure_df["original_index"] = self.structure_df.index

    @property
    def num_alternate_structures(self) -> int:
        """Get the number of alternate structures (i.e. number of chains)."""
        return len(self.structure_df["chain"].unique())

    def chain_residue_iter(self) -> Iterator[tuple[list[str], list[int]]]:
        """Generator for iterating over the chain, residue pairs in the structure.

        Since this class iterates over each chain individually, each iteration will
        yield a single chain with all of its residue ids.

        Yields
        ------
        tuple[str, int]
            Tuple of (chain, residue_id) pairs in the structure.
        """
        keep_residues = []
        if "amino_acid" in self.residue_types:
            keep_residues.extend(AMINO_ACID_RESIDUES)
        if "rna" in self.residue_types:
            keep_residues.extend(RNA_RESIDUES)
        if "dna" in self.residue_types:
            keep_residues.extend(DNA_RESIDUES)

        unique_chain_ids = self.structure_df["chain"].unique()
        for chain_id in unique_chain_ids:
            residue_ids = self.structure_df[self.structure_df["chain"] == chain_id][
                "residue_id"
            ].unique()

            # Subselect residue ids that match the desired residue types
            residue_ids = self.structure_df[
                (self.structure_df["chain"] == chain_id)
                & (self.structure_df["residue"].isin(keep_residues))
            ]["residue_id"].unique()

            residue_ids = residue_ids.tolist()

            # If no residues match the desired residue types, skip the chain
            if not residue_ids:
                continue

            yield [chain_id] * len(residue_ids), residue_ids

    def atom_idx_iter(self, inverted: bool = False) -> Iterator[torch.Tensor]:
        """Generator for iterating over atom indexes to keep in each structure.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        torch.Tensor
            Tensor of indexes for the atoms to remove.
        """
        remove_atoms = []
        if "amino_acid" in self.residue_types:
            remove_atoms.extend(self.amino_acid_atoms)
        if "rna" in self.residue_types:
            remove_atoms.extend(self.rna_atoms)
        if "dna" in self.residue_types:
            remove_atoms.extend(self.dna_atoms)

        cr_iter = self.chain_residue_iter()

        # Iterate over the chain, residue pairs and yield the atom indexes
        for chains_window, residues_window in cr_iter:
            # Merge the DataFrame to keep only positions where the chain and residue
            # pairs match the current window
            # NOTE: When the dataframe is merged, the row indexes are overwritten...
            # We use a dummy column to keep track of the original indexes by re-indexing
            # the dataframe after the merge.
            merge_df = pd.DataFrame(
                {"chain": chains_window, "residue_id": residues_window}
            )
            df_window = self.structure_df.merge(merge_df)
            df_window = df_window.set_index("original_index")

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window[df_window["atom"].isin(remove_atoms)].index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield torch.tensor(atom_idxs)


class ResidueTemplateIterator(BaseTemplateIterator):
    """Template iterator for removing chunks of atoms from residues from a structure.

    NOTE: If you want to set a fixed chain order (e.g. for reproducibility), you can
    use the built-in function (TODO).

    Attributes
    ----------
    type : Literal["residue"]
        Discriminator field for differentiating between template iterator types.
    randomize_chain_order : bool
        If 'True', randomize the order of chains in the structure. Default is 'False'.
    """

    type: ClassVar[Literal["residue"]] = "residue"
    num_residues_removed: Annotated[int, Field(gt=0)]
    residue_increment: Annotated[int, Field(gt=0)]

    randomize_chain_order: bool = False

    _chain_order: list[str]

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Add dummy column to keep track of the original indexes
        self.structure_df["original_index"] = self.structure_df.index

        # The unique method should retain default order
        self._chain_order = self.structure_df["chain"].unique()
        if self.randomize_chain_order:
            np.random.shuffle(self._chain_order)

    def set_chain_order(self, chain_order: list[str]) -> None:
        """Set the order of chains to iterate over.

        Parameters
        ----------
        chain_order : list[str]
            List of chain identifiers, in desired order, to use when iterating
            over the structure.

        Raises
        ------
        ValueError
            If the chain order does not contain all chains in the structure.

        Returns
        -------
        None
        """
        # Check that all the chains are present in the chain_order list
        if set(chain_order) != set(self.structure_df["chain"].unique()):
            raise ValueError("Chain order must contain all chains in the structure.")

        self._chain_order = chain_order

    @property
    def num_alternate_structures(self) -> int:
        """Get the number of alternate structures (i.e. number of chains)."""
        # Find the number of unique (chain, residue_id) pairs
        cr_pairs = self.chain_residue_pairs()
        num_pairs = len(cr_pairs)
        if num_pairs == 0:
            return 0

        return (num_pairs - 1) // self.residue_increment + 1

    def chain_residue_pairs(self) -> list[tuple[str, int]]:
        """Get the (chain, residue) pairs for the structure.

        NOTE: The returned pairs are unique and in the desired chain order. If you
        want to randomize or otherwise set the the order, must be done before calling
        this method.

        # This method is necessary because a PDB file may be malformed and have
        # duplicate chain identifiers. This method should handle the case where there
        # might be duplicate chain identifiers.

        Returns
        -------
        list[tuple[str, int]]
            List of (chain, residue_id) pairs in the structure.
        """
        # Determine which residues to keep based on the residue types
        keep_residues = []
        if "amino_acid" in self.residue_types:
            keep_residues.extend(AMINO_ACID_RESIDUES)
        if "rna" in self.residue_types:
            keep_residues.extend(RNA_RESIDUES)
        if "dna" in self.residue_types:
            keep_residues.extend(DNA_RESIDUES)

        # Subset the df to only residues that match the desired residue types
        subset_df = self.structure_df[self.structure_df["residue"].isin(keep_residues)]

        # Chunk the df into groups based on chain and re-stich together in order
        # This will ensure that the chain order is respected
        df_list = []
        for chain in self._chain_order:
            df_list.append(subset_df[subset_df["chain"] == chain])
        ordered_df = pd.concat(df_list)

        # Find unique (chain, residue_id) pairs, in order
        unique_chain_res_id = ordered_df[["chain", "residue_id"]].drop_duplicates()

        return unique_chain_res_id.to_records(index=False).tolist()  # type: ignore

    def chain_residue_iter(self) -> Iterator[tuple[list[str], list[int]]]:
        """Generator for iterating over the chain, residue pairs in the structure.

        Yields
        ------
        tuple[str, int]
            Tuple of (chain, residue_id) pairs in the structure.
        """
        # Get the unique chain, residue pairs in order
        chain_res_pairs = self.chain_residue_pairs()
        chains = np.array([chain for chain, _ in chain_res_pairs])
        residues = np.array([residue for _, residue in chain_res_pairs])

        window_iter = sliding_window_iterator(
            length=len(chain_res_pairs),
            window_width=self.num_residues_removed,
            step_size=self.residue_increment,
        )

        # Iterate over the chain, residue pairs and yield the pairs
        for window in window_iter:
            chains_window = chains[window]
            residues_window = residues[window]
            yield chains_window, residues_window

    def atom_idx_iter(self, inverted: bool = False) -> Iterator[torch.Tensor]:
        """Generator for iterating over atom indexes to keep in each structure.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        torch.Tensor
            Tensor of indexes for the atoms to remove.
        """
        remove_atoms = []
        if "amino_acid" in self.residue_types:
            remove_atoms.extend(self.amino_acid_atoms)
        if "rna" in self.residue_types:
            remove_atoms.extend(self.rna_atoms)
        if "dna" in self.residue_types:
            remove_atoms.extend(self.dna_atoms)

        cr_iter = self.chain_residue_iter()

        # Iterate over the chain, residue pairs and yield the atom indexes
        for chains_window, residues_window in cr_iter:
            # Merge the DataFrame to keep only positions where the chain and residue
            # pairs match the current window
            # NOTE: When the dataframe is merged, the row indexes are overwritten...
            # We use a dummy column to keep track of the original indexes by re-indexing
            # the dataframe after the merge.
            merge_df = pd.DataFrame(
                {"chain": chains_window, "residue_id": residues_window}
            )
            df_window = self.structure_df.merge(merge_df)
            df_window = df_window.set_index("original_index")

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window[df_window["atom"].isin(remove_atoms)].index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield torch.tensor(atom_idxs)


class RandomResidueTemplateIterator(ResidueTemplateIterator):
    """Template iterator for removing random residues from a pdb structure.

    Attributes
    ----------
    type : Literal["random_residue"]
        Discriminator field for differentiating between template iterator types.
    coherent_removal : bool
        If 'True', remove residues in order. For example, would remove residues
        [i, i+1, i+2, ...] rather than random indices. Default is 'True'.
    num_residues_removed : int
        Number of residues to remove from the structure at each iteration.
        Must be greater than 0.
    num_alternate_structures : int
        Number of alternate structures to generate by removing random residues. Must be
        greater than 0.
    """

    type: ClassVar[Literal["random_residue"]] = "random_residue"
    coherent_removal: bool = True
    _num_alternate_structures: int

    def __init__(self, num_alternate_structures: int, **data: Any):
        super().__init__(**data)

        self._num_alternate_structures = num_alternate_structures
        # Add dummy column to keep track of the original indexes
        self.structure_df["original_index"] = self.structure_df.index
        
    @property
    def num_alternate_structures(self) -> int:
        """Get the number of alternate structures (i.e. number of chains)."""
        return self._num_alternate_structures

    def chain_residue_iter(self) -> Iterator[tuple[list[str], list[int]]]:
        """Generator for iteration of the chain, residue pairs in the structure."""
        chain_res_pairs = self.chain_residue_pairs()

        if self.coherent_removal:
            start_indexes = np.random.randint(
                0,
                len(chain_res_pairs) - self.num_residues_removed + 1,
                size=self.num_alternate_structures,
            )
            for start_idx in start_indexes:
                window = np.arange(start_idx, start_idx + self.num_residues_removed)
                chains_window = [chain_res_pairs[i][0] for i in window]
                residues_window = [chain_res_pairs[i][1] for i in window]
                yield chains_window, residues_window
        else:
            for _ in range(self.num_alternate_structures):
                random_idxs = np.random.choice(
                    len(chain_res_pairs),
                    size=min(self.num_residues_removed, len(chain_res_pairs)),
                    replace=False,
                )
                chains_window = [chain_res_pairs[i][0] for i in random_idxs]
                residues_window = [chain_res_pairs[i][1] for i in random_idxs]
                yield chains_window, residues_window


# class PrecalculatedVolumesTemplateIterator(BaseTemplateIterator):
#     """Template iterator for iterating over a set of precalculated volumes."""

#     pass
#     _type: Literal["precalculated-list"] = "precalculated-list"
