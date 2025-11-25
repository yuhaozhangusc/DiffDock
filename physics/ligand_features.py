# ligand_features.py
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


# A tiny VDW radii table (Å); extend if needed.
VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}


def compute_ligand_features_from_rdkit(
    mol: Chem.Mol,
    conf_id: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """
    Build per-ligand features.

    IMPORTANT:
        - This function assumes the RDKit atom ordering in `mol` matches the
          ligand atom ordering in your DiffDock `complex_graph['ligand']`.
        - Therefore you should call this AFTER any atom-matching / reordering
          (e.g. after `get_lig_graph_with_matching`), on the *final* RDKit
          ligand whose order matches the graph.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule with a valid conformer.
    conf_id : int
        Conformer index to use for reference geometry.
    device : torch.device, optional
        Device for returned tensors (default: CPU).

    Returns
    -------
    Dict[str, Tensor]
        - "vdw_radius": (N,)  per-atom VDW radius
        - "bond_index": (M, 2)  bond pairs (i, j)
        - "bond_ref_length": (M,)  reference bond lengths
        - "angle_index": (K, 3)  angle triples (i, j, k)
        - "angle_ref": (K,)  reference angle values (radians)
    """
    if device is None:
        device = torch.device("cpu")

    # We MUST NOT add/remove atoms here: that would break the
    # correspondence with DiffDock's ligand atom ordering.
    assert mol.GetNumConformers() > 0, (
        "Ligand RDKit mol must have a conformer before computing features. "
        "Generate a conformer earlier in your preprocessing pipeline."
    )
    conf = mol.GetConformer(conf_id)

    num_atoms = mol.GetNumAtoms()

    # VDW radii
    vdw = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        vdw.append(VDW_RADII.get(z, 1.7))  # default ~C for unknown types
    vdw = torch.tensor(vdw, dtype=torch.float32, device=device)

    # Bonds
    bond_indices = []
    bond_ref_lengths = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_indices.append((i, j))
        d = rdMolTransforms.GetBondLength(conf, i, j)
        bond_ref_lengths.append(d)

    if bond_indices:
        bond_index = torch.tensor(bond_indices, dtype=torch.long, device=device)      # (M, 2)
        bond_ref_length = torch.tensor(bond_ref_lengths, dtype=torch.float32, device=device)  # (M,)
    else:
        bond_index = torch.zeros((0, 2), dtype=torch.long, device=device)
        bond_ref_length = torch.zeros((0,), dtype=torch.float32, device=device)

    # Simple angle list: for each atom j with at least 2 neighbors, form (i, j, k)
    angle_triples = []
    angle_refs = []
    for j in range(num_atoms):
        nbrs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(j).GetNeighbors()]
        if len(nbrs) < 2:
            continue
        for a_i in range(len(nbrs)):
            for a_k in range(a_i + 1, len(nbrs)):
                i = nbrs[a_i]
                k = nbrs[a_k]
                angle_triples.append((i, j, k))
                theta = rdMolTransforms.GetAngleRad(conf, i, j, k)
                angle_refs.append(theta)

    if angle_triples:
        angle_index = torch.tensor(angle_triples, dtype=torch.long, device=device)  # (K, 3)
        angle_ref = torch.tensor(angle_refs, dtype=torch.float32, device=device)    # (K,)
    else:
        angle_index = torch.zeros((0, 3), dtype=torch.long, device=device)
        angle_ref = torch.zeros((0,), dtype=torch.float32, device=device)

    feats = {
        "vdw_radius": vdw,                # (N,)
        "bond_index": bond_index,         # (M, 2)
        "bond_ref_length": bond_ref_length,  # (M,)
        "angle_index": angle_index,       # (K, 3)
        "angle_ref": angle_ref,           # (K,)
    }
    return feats