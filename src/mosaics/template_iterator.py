"""Module for different ways of generating alternate templates for comparison."""

from abc import abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
import mmdf
from ttsim3d.models import Simulator
from pydantic import BaseModel, ConfigDict, Field, field_validator
from teamtomo_basemodel import ExcludedDataFrame
from ttsim3d.scattering_potential import get_a_param

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
    elif iterator_type == "added_template":
        return AddedTemplateIterator(**data)
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
    structure_df : ExcludedDataFrame
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
    get_default_template_idxs()
        Get the atom indices of the structure_df corresponding to the default template
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    type: ClassVar[Literal["random", "random_residue", "chain", "residue", "added_template"]]

    residue_types: list[Literal["amino_acid", "rna", "dna"]]
    amino_acid_atoms: list[str] = DEFAULT_AMINO_ACID_ATOMS
    rna_atoms: list[str] = DEFAULT_RNA_ATOMS
    dna_atoms: list[str] = DEFAULT_DNA_ATOMS

    structure_df: ExcludedDataFrame  # NOTE: Comes from Simulator object

    @field_validator("structure_df", mode="after")  # type: ignore
    def _validate_structure_df(cls, v):
        v["original_index"] = v.index
        return v

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
    def alternate_template_iter(
        self, inverted: bool = True
    ) -> Iterator[tuple[list[str | None], list[int | None], torch.Tensor]]:
        """Iterator over chain, reside, and atom indices removed for alternates.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        tuple[list[str], list[int], torch.Tensor]
            Tuple of (chains, residue_ids, atom_idxs) removed at each iteration.
            The ordering of the chains and residue_ids lists correspond to each other,
            example (['A', 'A', 'B'], [1, 2, 3]) means residues 1 & 2 were removed from
            chain 'A' and the 3rd was removed from chain 'B'.
            The atom_idxs is a tensor of the atom indexes that were removed and depends
            on the atom types specified in the configuration.
        """

    def subset_df_on_residues(self) -> pd.DataFrame:
        """Returns a df subset selecting only valid residues to consider."""
        keep_residues: list[str] = []
        if "amino_acid" in self.residue_types:
            keep_residues.extend(AMINO_ACID_RESIDUES)
        if "rna" in self.residue_types:
            keep_residues.extend(RNA_RESIDUES)
        if "dna" in self.residue_types:
            keep_residues.extend(DNA_RESIDUES)
        subset_df = self.structure_df[self.structure_df["residue"].isin(keep_residues)]
        subset_df.set_index("original_index", inplace=True, drop=False)

        return subset_df

    def subset_df_on_residues_and_atoms(self) -> pd.DataFrame:
        """Returns a df subset selecting only valid residues and atoms to consider."""
        subset_df = self.subset_df_on_residues()

        keep_atoms: list[str] = []
        if "amino_acid" in self.residue_types:
            keep_atoms.extend(self.amino_acid_atoms)
        if "rna" in self.residue_types:
            keep_atoms.extend(self.rna_atoms)
        if "dna" in self.residue_types:
            keep_atoms.extend(self.dna_atoms)

        subset_df = subset_df[subset_df["atom"].isin(keep_atoms)]
        subset_df.set_index("original_index", inplace=True, drop=False)

        return subset_df

    def get_template_scattering_potential(
        self, atom_idxs: torch.Tensor | np.ndarray = None
    ) -> float:
        """Get the mass (in amu) of a template structure given atom indexes."""
        if atom_idxs is None:
            atom_idxs = self.get_default_template_idxs()

        if isinstance(atom_idxs, torch.Tensor):
            atom_idxs = atom_idxs.numpy()

        total_scattering_potential = 0
        atom_counts = self.structure_df.iloc[atom_idxs]["element"].value_counts()
        for atom, count in atom_counts.items():
            atom = atom.upper()
            potentials = get_a_param([atom]) # pass in list of atoms, since for 2 letters, the param thinks you're passing in a list
            potentials = torch.sum(potentials).item()
            total_scattering_potential += potentials * count

        

        return total_scattering_potential
    
    def get_default_template_idxs(self) -> torch.Tensor:
        """Return the atom indices corresponding to the default template"""
        default_df = self.structure_df.copy()
        default_df.set_index("original_index")
        return torch.tensor(default_df.index)


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

    def alternate_template_iter(
        self, inverted: bool = True
    ) -> Iterator[tuple[list[str | None], list[int | None], torch.Tensor]]:
        """Randomly removes atoms (of specified types) from the structure.

        If coherent_removal is True, then a random starting index is chosen and a chunk
        of num_atoms_removed atoms is removed. If coherent_removal is False, then atoms
        are randomly sampled from the structure, without replacement.

        """
        subset_df = self.subset_df_on_residues_and_atoms()

        # Choose which atom indices to remove based on the coherent_removal flag
        if self.coherent_removal:
            start_indexes = np.random.randint(
                0,
                len(subset_df) - self.num_atoms_removed + 1,
                size=self.num_alternate_structures,
            )
            indices = np.arange(self.num_atoms_removed)
            removed_indices = np.stack(
                [indices + start_idx for start_idx in start_indexes]
            )
        else:
            removed_indices = np.stack(
                [
                    np.random.choice(
                        subset_df.index,
                        size=min(self.num_atoms_removed, len(subset_df)),
                        replace=False,
                    )
                    for _ in range(self.num_alternate_structures)
                ]
            )

        for i in range(self.num_alternate_structures):
            atom_idxs = removed_indices[i]

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield [None], [None], torch.tensor(atom_idxs)


class ChainTemplateIterator(BaseTemplateIterator):
    """Iterates over each chain in the structure and removes specified atoms from it.

    Attributes
    ----------
    type : Literal["chain"]
        Discriminator field for differentiating between template iterator types.
    """

    type: ClassVar[Literal["chain"]] = "chain"

    @property
    def num_alternate_structures(self) -> int:
        """Get the number of alternate structures (i.e. number of chains)."""
        return len(self.structure_df["chain"].unique())

    def alternate_template_iter(
        self, inverted: bool = True
    ) -> Iterator[tuple[list[str | None], list[int | None], torch.Tensor]]:
        """Iterator over chain, residue, and atom indices removed for alternates.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.
        """
        subset_df = self.subset_df_on_residues_and_atoms()
        unique_chain_ids = subset_df["chain"].unique()

        for chain_id in unique_chain_ids:
            print("deleting chain", chain_id)
            residue_ids = subset_df[subset_df["chain"] == chain_id]["residue_id"]
            residue_ids = residue_ids.unique().tolist()

            chain_ids = [chain_id] * len(residue_ids)

            # Merge the DataFrame to keep only positions where the chain and residue
            # pairs match the current window
            # NOTE: When the dataframe is merged, the row indexes are overwritten...
            # We use a dummy column to keep track of the original indexes by re-indexing
            # the dataframe after the merge.
            merge_df = pd.DataFrame({"chain": chain_ids, "residue_id": residue_ids})
            df_window = subset_df.merge(merge_df)
            df_window = df_window.set_index("original_index")

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window.index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield chain_ids, residue_ids, torch.tensor(atom_idxs)


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
        subset_df = self.subset_df_on_residues()

        # Chunk the df into groups based on chain and re-stich together in order
        # This will ensure that the chain order is respected
        df_list = []
        for chain in self._chain_order:
            df_list.append(subset_df[subset_df["chain"] == chain])
        ordered_df = pd.concat(df_list)

        # Find unique (chain, residue_id) pairs, in order
        unique_chain_res_id = ordered_df[["chain", "residue_id"]].drop_duplicates()

        return unique_chain_res_id.to_records(index=False).tolist()  # type: ignore

    def alternate_template_iter(
        self, inverted: bool = True
    ) -> Iterator[tuple[list[str | None], list[int | None], torch.Tensor]]:
        """Iterator over chain, residue, and atom indices removed for alternates.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.
        """
        chain_res_pairs = self.chain_residue_pairs()
        chains = np.array([chain for chain, _ in chain_res_pairs])
        residues = np.array([residue for _, residue in chain_res_pairs])

        window_iter = sliding_window_iterator(
            length=len(chain_res_pairs),
            window_width=self.num_residues_removed,
            step_size=self.residue_increment,
        )

        subset_df = self.subset_df_on_residues_and_atoms()

        for idx in window_iter:
            chain_ids = chains[idx]
            residue_ids = residues[idx]

            # Merge the DataFrame to keep only positions where the chain and residue
            # pairs match the current window
            # NOTE: When the dataframe is merged, the row indexes are overwritten...
            # We use a dummy column to keep track of the original indexes by re-indexing
            # the dataframe after the merge.
            merge_df = pd.DataFrame({"chain": chain_ids, "residue_id": residue_ids})
            df_window = subset_df.merge(merge_df)
            df_window = df_window.set_index("original_index")

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window.index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield chain_ids.tolist(), residue_ids.tolist(), torch.tensor(atom_idxs)


class RandomResidueTemplateIterator(BaseTemplateIterator):
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
    num_residues_removed: Annotated[int, Field(gt=0)]
    num_alternate_structures: Annotated[int, Field(gt=0)]

    def alternate_template_iter(
        self, inverted: bool = True
    ) -> Iterator[tuple[list[str | None], list[int | None], torch.Tensor]]:
        """Randomly removes residues (of specified types) from the structure.

        If coherent_removal is True, then a random starting index is chosen and a chunk
        of num_residues_removed residues is removed. If coherent_removal is False, then
        residues are randomly sampled from the structure, without replacement.

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.

        Yields
        ------
        tuple[list[str], list[int], torch.Tensor]
            Tuple of (chains, residue_ids, atom_idxs) removed at each iteration.
            The ordering of the chains and residue_ids lists correspond to each other,
            example (['A', 'A', 'B'], [1, 2, 3]) means residues 1 & 2 were removed from
            chain 'A' and the 3rd was removed from chain 'B'.
            The atom_idxs is a tensor of the atom indexes that were removed and depends
            on the atom types specified in the configuration.
        """
        subset_df = self.subset_df_on_residues_and_atoms()

        # Get the unique chain, residue pairs in order
        chain_res_pairs = subset_df[["chain", "residue_id"]].drop_duplicates()
        chain_res_pairs = chain_res_pairs.to_records(index=False).tolist()
        chains = np.array([chain for chain, _ in chain_res_pairs])
        residues = np.array([residue for _, residue in chain_res_pairs])

        # Choose which residue indices to remove based on the coherent_removal flag
        if self.coherent_removal:
            start_indexes = np.random.randint(
                0,
                len(chain_res_pairs) - self.num_residues_removed + 1,
                size=self.num_alternate_structures,
            )
            indices = np.arange(self.num_residues_removed)
            removed_indices = np.stack(
                [indices + start_idx for start_idx in start_indexes]
            )
        else:
            removed_indices = np.stack(
                [
                    np.random.choice(
                        len(chain_res_pairs),
                        size=min(self.num_residues_removed, len(chain_res_pairs)),
                        replace=False,
                    )
                    for _ in range(self.num_alternate_structures)
                ]
            )

        for i in range(self.num_alternate_structures):
            res_indices = removed_indices[i]
            chain_ids = chains[res_indices].tolist()
            residue_ids = residues[res_indices].tolist()

            # Merge the DataFrame to keep only positions where the chain and residue
            # pairs match the current window
            # NOTE: When the dataframe is merged, the row indexes are overwritten...
            # We use a dummy column to keep track of the original indexes by re-indexing
            # the dataframe after the merge.
            merge_df = pd.DataFrame({"chain": chain_ids, "residue_id": residue_ids})
            df_window = subset_df.merge(merge_df)
            df_window = df_window.set_index("original_index")

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window.index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield chain_ids, residue_ids, torch.tensor(atom_idxs)

class AddedTemplateIterator(BaseTemplateIterator):
    """Template iterator for using an alternative list of PDB files. 

    Attributes
    ----------
    type : Literal["added_template"]
        Discriminator field for differentiating between template iterator types.
    added_startswith : list[str]
        specified characters that all added chain names start with.
        Default is ["_"]
    """

    type: ClassVar[Literal["added_template"]] = "added_template"
    added_startswith: list[str] = Field(default_factory=lambda: ["_"])
    

    def __init__(self, added_startswith: list[str] | None = None, **data):  # need to initialize the default start character
        super().__init__(**data)
        self.added_startswith = added_startswith or ["_"]
    
    @property
    def num_alternate_structures(self) -> int:
        """returns the number of alternate structures (number of alternate PDB files)"""
        unique_chain_ids = [chain for chain in self.structure_df["chain"].unique() if chain[0] in self.added_startswith]
        return len(unique_chain_ids)
    
    def get_default_template_idxs(self) -> torch.Tensor:
        """Removes all added chain indices and simulates the default 
        
        Added chains must start with something in self.added_startswith

        Returns
        -------
        torch.Tensor
        tensor of the atom indices to simulate to produce the default template
        """
        default_chains = [chain for chain in self.structure_df["chain"].unique() if not chain[0] in self.added_startswith]
        default_df = self.structure_df[self.structure_df['chain'].isin(default_chains)].copy()
        default_df = default_df.set_index("original_index")
        return torch.tensor(default_df.index)
    
    def get_template_scattering_potential(
        self, atom_idxs: torch.Tensor | np.ndarray = None
    ) -> float:
        """Get the mass (in amu) of a template structure given atom indexes. Default template does not contain alt chains"""
        if atom_idxs is None:
            atom_idxs = self.get_default_template_idxs().numpy()

        if isinstance(atom_idxs, torch.Tensor):
            #I'll want all the idxs from the structure_df
            df = self.structure_df
            added_chain_ids = [chain for chain in df["chain"].unique() if chain[0] in self.added_startswith]
            added_chain_idxs = df[df["chain"].isin(added_chain_ids)].copy()
            atom_idxs = added_chain_idxs[~added_chain_idxs.isin(atom_idxs.numpy())]
            

        total_scattering_potential = 0
        atom_counts = self.structure_df.iloc[atom_idxs]["element"].value_counts()
        for atom, count in atom_counts.items():
            atom = atom.upper()
            potentials = get_a_param([atom]) # pass in list of atoms, since for 2 letters, the param thinks you're passing in a list
            potentials = torch.sum(potentials).item()
            total_scattering_potential += potentials * count
        

        return total_scattering_potential
    
    def alternate_template_iter(
            self, inverted: bool=True
    ) -> Iterator[tuple[list[str | None], list[int | None], torch.Tensor]]:
        """Iterate over alternate templates by removing all but one alternate chain at a time.

        Added chains must start with something in the self.added_startswith list

        Parameters
        ----------
        inverted : bool
            If 'True', return the indexes of atoms to remove rather than keep.
        """
        
        subset_df = self.subset_df_on_residues_and_atoms()
        subset_df = subset_df.reset_index(drop=True)
        # set unique chains as those without the 
        unique_chain_ids = [chain for chain in subset_df["chain"].unique() if chain[0] in self.added_startswith]
        for chain_id in unique_chain_ids:
            chains_to_remove = [chain for chain in unique_chain_ids if chain != chain_id]
            print(f"Removing chains {chains_to_remove}")
            removed_residues = []
            removed_chains = []
            for chain in chains_to_remove:
                residue_ids = subset_df[subset_df["chain"]==chain]["residue_id"]
                residue_ids = residue_ids.unique().tolist()
                chain_ids = [chain_id] * len(residue_ids)
                removed_chains.extend(chain_ids)
                removed_residues.extend(residue_ids)
            merge_df = pd.DataFrame({"chain": removed_chains, "residue_id": removed_residues})
            df_window = subset_df.merge(merge_df)
            df_window = df_window.set_index("original_index")

            # Get the atom indexes for the current window (atoms that should be removed)
            atom_idxs = df_window.index

            if inverted:
                atom_idxs = np.setdiff1d(np.arange(len(self.structure_df)), atom_idxs)

            yield chain_ids, residue_ids, torch.tensor(atom_idxs)