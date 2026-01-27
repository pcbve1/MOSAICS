import mmdf 
import pandas as pd 

def create_added_pdb_df(added_pdb_list: list[str]) -> pd.DataFrame:
    """Creates a dataframe from added pdb files
    
    Parameters
    ----------
    added_pdb_list : list[str]
        List of pdb filepaths that you want added to the default pdb
    
    Returns
    -------
    pd.DataFrame :
        DataFrame of all added pdbs
    """
    assert len(added_pdb_list) > 0, "Please input a list with pdb filepaths in it"
    dfs = []

    for i, pdb in enumerate(added_pdb_list):
        j = i // 9
        df = mmdf.read(pdb)
        if j == 0:
            df["chain"] = f"_{i}"  # does not keep original chains
        elif j == 1:
            df["chain"] = f"+{i - 9}"  # does not keep original chains
        else:
            raise ValueError("Too many chains passed! 20 is the max!")
        dfs.append(df)
        if i // 9 == 1: 
            j = 1
        if i // 9 == 2:
            j = 2
    return pd.concat(dfs, ignore_index=True)

def produce_concat_df(
        default_pdb: str, 
        added_pdbs: list[str]
        ) -> pd.DataFrame:
    """Creates a dataframe from added pdb files
    
    Parameters
    ----------
    default_pdb : str
        Filename of the default pdb that will have chains added to it
    added_pdb_list : list[str]
        List of pdb filepaths whose chains will be added to the pdb file
    
    Returns
    -------
    pd.DataFrame:
        a dataframe containing all atoms, b-factors, and atomic coordinates 
        of the added pdb chains + the default
    """
    default_pdb_df = mmdf.read(default_pdb)
    for chain in set(default_pdb_df["chain"]):
        if len(chain) > 2: 
            default_pdb_df.replace(chain, chain[0:2], inplace=True)
            print(chain[0:2])
    added_pdb_df = create_added_pdb_df(added_pdbs)
    return pd.concat([default_pdb_df, added_pdb_df], ignore_index=True)

def concat_chains_to_default_pdb(
        default_pdb_path: str, 
        added_pdbs_list: list[str], 
        output_path: str
        ) -> None:
    """Produces a hybrid pdb file
    
    Parameters
    ----------
    default_pdb : str
        Filename of the default pdb that will have chains added to it
    added_pdb_list : list[str]
        List of pdb filepaths whose chains will be added to the pdb file
    output_path : str
        Path to the final pdb file to be created
    """
    all_chains_df = produce_concat_df(default_pdb_path, added_pdbs_list)
    mmdf.write(output_path, all_chains_df)
    