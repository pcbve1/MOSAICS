import mmdf
import numpy as np
import os

def change_pixel_size(
        pdb_file: str, 
        original_pixel_size: float, 
        final_pixel_size: float,
        out_dir: str|None = None
        ) -> None:
    """
    Transforms a pdb file's pixel size using mmdf to a proper size by adjusting all distances by the ratio of pixel sizes
    
    Parameters: 
    -----------
    pdb_file (str):
        path to a .pdb or .cif file specifying atomic coordinates
    original_pixel_size (float): 
        pixel size that is optimized for the template in question (in Angstroms)
    final_pixel_size (float):
        final pixel size (in Angstroms) that you want your template to have
    out_dir (str, optional):
        directory where you want to save the pdb file. If None, will default to the directory of the pdb file
    
    Returns:
    --------
    None
    """
    if not out_dir:
        out_dir = os.path.dirname(pdb_file)
    df = mmdf.read(pdb_file)
    structure_coords = df[["x", "y", "z"]].values
    center = np.mean(structure_coords, axis=0)
    ratio = final_pixel_size/original_pixel_size
    adjusted_coords = (structure_coords - center) * ratio
    df2 = df.copy()
    updated_chains = []
    for chain in df["chain"]:
        updated_chains.append(chain[:2])
    df2["chain"] = updated_chains
    df2[["x", "y", "z"]] = adjusted_coords
    fname = f"{os.path.basename(pdb_file).split('_pix')[0]}_pix_{final_pixel_size:.3F}.pdb"
    mmdf.write(f"{out_dir}/{fname}", df2)