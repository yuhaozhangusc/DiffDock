# potentials.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor


@dataclass
class Potential:
    """
    Base class for simple ligand potentials.
    Each potential carries a scalar `weight` that scales its contribution.
    """
    weight: float = 1.0

    def energy(self, coords: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        """
        Compute the (unweighted) energy given coordinates.

        Parameters
        ----------
        coords : Tensor
            Shape (N, 3): ligand atom coordinates.
        feats : Dict[str, Tensor]
            Per-ligand features (bond indices, reference distances, etc.)

        Returns
        -------
        Tensor
            Scalar tensor E (unweighted).
        """
        raise NotImplementedError

    def total_energy(self, coords: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        """
        Convenience: returns weight * energy(coords, feats).
        """
        return self.weight * self.energy(coords, feats)

    def gradient(self, coords: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        """
        Compute d(weight * E)/d(coords) with autograd.

        Parameters
        ----------
        coords : Tensor
            Shape (N, 3), typically atom_coords_denoised[b] for one sample.
        feats : Dict[str, Tensor]

        Returns
        -------
        Tensor
            Shape (N, 3): gradient w.r.t coords.
        """
        with torch.enable_grad():
            coords_req = coords.detach().clone().requires_grad_(True)
            E = self.total_energy(coords_req, feats)
            (grad,) = torch.autograd.grad(
                E,
                coords_req,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grad


class BondLengthPotential(Potential):
    """
    Quadratic penalty on deviation from reference bond lengths.

    Requires feats:
      - "bond_index": (M, 2) int64
      - "bond_ref_length": (M,) float32
      - optional "bond_weight": (M,)
    """

    def energy(self, coords: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        bond_index = feats["bond_index"]        # (M, 2)
        ref_len = feats["bond_ref_length"]      # (M,)
        bond_weight = feats.get("bond_weight", None)

        if bond_index.numel() == 0:
            return coords.new_tensor(0.0)

        pos_i = coords[bond_index[:, 0]]
        pos_j = coords[bond_index[:, 1]]
        dist = torch.linalg.norm(pos_i - pos_j, dim=-1)  # (M,)

        diff = dist - ref_len
        if bond_weight is not None:
            diff = diff * bond_weight

        # Unweighted energy; Potential.total_energy() multiplies by self.weight
        return (diff ** 2).sum()


class VDWClashPotential(Potential):
    """
    Simple van der Waals clash penalty.

    Requires feats:
      - "vdw_radius": (N,) float32
    """

    def __init__(self, overlap_frac: float = 0.8, weight: float = 1.0):
        """
        Parameters
        ----------
        overlap_frac : float
            Fraction of (r_i + r_j) allowed before penalty.
            e.g. 0.8 means penalty if dist < 0.8 * (r_i + r_j)
        weight : float
            Overall scale factor for this potential.
        """
        super().__init__(weight=weight)
        self.overlap_frac = overlap_frac

    def energy(self, coords: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        vdw = feats["vdw_radius"]  # (N,)
        N = coords.shape[0]

        if N <= 1:
            return coords.new_tensor(0.0)

        # Pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)
        dist = torch.linalg.norm(diff + 1e-9, dim=-1)     # (N, N)

        # Symmetric VDW sum
        r_sum = vdw.unsqueeze(1) + vdw.unsqueeze(0)       # (N, N)
        cutoff = self.overlap_frac * r_sum

        # Only consider i < j
        mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=coords.device), 1
        )
        clash = torch.relu(cutoff - dist)  # (N, N)
        clash = clash[mask]

        if clash.numel() == 0:
            return coords.new_tensor(0.0)

        # Unweighted energy; Potential.total_energy() multiplies by self.weight
        return (clash ** 2).sum()


class AnglePotential(Potential):
    """
    Quadratic penalty for deviations from reference bond angles.

    Requires feats:
      - "angle_index": (K, 3) int64 (i, j, k)
      - "angle_ref": (K,) float32 [radians]
      - optional "angle_weight": (K,)
    """

    def energy(self, coords: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        if "angle_index" not in feats or feats["angle_index"].numel() == 0:
            return coords.new_tensor(0.0)

        idx = feats["angle_index"]  # (K, 3)
        ref = feats["angle_ref"]    # (K,)
        w = feats.get("angle_weight", None)

        p_i = coords[idx[:, 0]]
        p_j = coords[idx[:, 1]]
        p_k = coords[idx[:, 2]]

        v1 = p_i - p_j
        v2 = p_k - p_j
        v1 = torch.nn.functional.normalize(v1, dim=-1)
        v2 = torch.nn.functional.normalize(v2, dim=-1)

        cos_theta = (v1 * v2).sum(-1).clamp(-1.0, 1.0)
        theta = torch.arccos(cos_theta)  # (K,)

        diff = theta - ref
        if w is not None:
            diff = diff * w

        # Unweighted energy; Potential.total_energy() multiplies by self.weight
        return (diff ** 2).sum()


def build_default_potentials() -> List[Potential]:
    """
    Factory for Option A: simple ligand geometry + clash control.

    You can tune the weights here to control the strength of each term.
    """
    return [
        BondLengthPotential(weight=1.0),
        AnglePotential(weight=0.2),
        VDWClashPotential(overlap_frac=0.8, weight=0.5),
    ]