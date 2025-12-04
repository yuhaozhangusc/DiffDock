from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdchem

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

    # Planar (improper) dihedrals for double/conjugated bonds with sp2 ends
    planar_quads = []
    for bond in mol.GetBonds():
        if not (bond.GetBondType() == rdchem.BondType.DOUBLE or bond.GetIsConjugated()):
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetHybridization() != rdchem.HybridizationType.SP2:
            continue
        if mol.GetAtomWithIdx(j).GetHybridization() != rdchem.HybridizationType.SP2:
            continue
        nbr_i = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
        nbr_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
        if not nbr_i or not nbr_j:
            continue
        for a in nbr_i:
            for b in nbr_j:
                planar_quads.append((a, i, j, b))

    if planar_quads:
        planar_improper_index = torch.tensor(planar_quads, dtype=torch.long, device=device)  # (P, 4)
        planar_ref = torch.zeros((len(planar_quads),), dtype=torch.float32, device=device)   # target angle = 0
    else:
        planar_improper_index = torch.zeros((0, 4), dtype=torch.long, device=device)
        planar_ref = torch.zeros((0,), dtype=torch.float32, device=device)

    # Chiral centers (R/S)
    chiral_rows, chiral_orients = [], []
    for idx, label in Chem.FindMolChiralCenters(
        mol, includeUnassigned=False, useLegacyImplementation=False
    ):
        nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
        if len(nbrs) < 3:
            continue
        chiral_rows.append([idx, nbrs[0], nbrs[1], nbrs[2]])
        chiral_orients.append(label == "R")  # True=R, False=S
    chiral_atom_index = (
        torch.tensor(chiral_rows, dtype=torch.long, device=device).T
        if chiral_rows
        else torch.empty((4, 0), dtype=torch.long, device=device)
    )
    chiral_atom_orientations = (
        torch.tensor(chiral_orients, dtype=torch.bool, device=device)
        if chiral_rows
        else torch.empty((0,), dtype=torch.bool, device=device)
    )

    # E/Z stereo bonds
    stereo_rows, stereo_orients = [], []
    for bond in mol.GetBonds():
        if bond.GetBondType() != rdchem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo not in (rdchem.BondStereo.STEREOE, rdchem.BondStereo.STEREOZ):
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atoms = bond.GetStereoAtoms()  # two substituents, one on each side
        if len(atoms) != 2:
            continue
        stereo_rows.append([atoms[0], i, j, atoms[1]])
        stereo_orients.append(stereo == rdchem.BondStereo.STEREOE)  # True=E, False=Z
    stereo_bond_index = (
        torch.tensor(stereo_rows, dtype=torch.long, device=device).T
        if stereo_rows
        else torch.empty((4, 0), dtype=torch.long, device=device)
    )
    stereo_bond_orientations = (
        torch.tensor(stereo_orients, dtype=torch.bool, device=device)
        if stereo_rows
        else torch.empty((0,), dtype=torch.bool, device=device)
    )

    feats = {
        "vdw_radius": vdw,                # (N,)
        "bond_index": bond_index,         # (M, 2)
        "bond_ref_length": bond_ref_length,  # (M,)
        "angle_index": angle_index,       # (K, 3)
        "angle_ref": angle_ref,           # (K,)
        "planar_improper_index": planar_improper_index,  # (P, 4)
        "planar_ref": planar_ref,                         # (P,)
        "chiral_atom_index": chiral_atom_index,                    # (4, C)
        "chiral_atom_orientations": chiral_atom_orientations,      # (C,)
        "stereo_bond_index": stereo_bond_index,                    # (4, S)
        "stereo_bond_orientations": stereo_bond_orientations,      # (S,)
    }
    return feats
