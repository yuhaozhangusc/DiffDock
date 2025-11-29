import copy
import random

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time, modify_conformer_batch
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R
from utils.utils import crop_beyond
from utils.logging_utils import get_logger

from physics.ligand_features import compute_ligand_features_from_rdkit
from physics.potentials import build_default_potentials 


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, pocket_knowledge=False, pocket_cutoff=7,
                       initial_noise_std_proportion=-1.0, choose_residue=False):
    # in place modification of the list
    center_pocket = data_list[0]['receptor'].pos.mean(dim=0)
    if pocket_knowledge:
        complex = data_list[0]
        d = torch.cdist(complex['receptor'].pos, torch.from_numpy(complex['ligand'].orig_pos[0]).float() - complex.original_center)
        label = torch.any(d < pocket_cutoff, dim=1)

        if torch.any(label):
            center_pocket = complex['receptor'].pos[label].mean(dim=0)
        else:
            print("No pocket residue below minimum distance ", pocket_cutoff, "taking closest at", torch.min(d))
            center_pocket = complex['receptor'].pos[torch.argmin(torch.min(d, dim=1)[0])]

    if not no_torsion and not no_random:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        if no_random:
            # Deterministic placement at the pocket center with original orientation.
            complex_graph['ligand'].pos = complex_graph['ligand'].pos - molecule_center + center_pocket
            continue

        # randomize position
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T + center_pocket
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            if choose_residue:
                idx = random.randint(0, len(complex_graph['receptor'].pos)-1)
                tr_update = torch.normal(mean=complex_graph['receptor'].pos[idx:idx+1], std=0.01)
            elif initial_noise_std_proportion >= 0.0:
                std_rec = torch.sqrt(torch.mean(torch.sum(complex_graph['receptor'].pos ** 2, dim=1)))
                tr_update = torch.normal(mean=0, std=std_rec * initial_noise_std_proportion / 1.73, size=(1, 3))
            else:
                # if initial_noise_std_proportion < 0.0, we use the tr_sigma_max multiplied by -initial_noise_std_proportion
                tr_update = torch.normal(mean=0, std=-initial_noise_std_proportion * tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update


def is_iterable(arr):
    try:
        some_object_iterator = iter(arr)
        return True
    except TypeError as te:
        return False


def sampling(
    data_list,
    model,
    inference_steps,
    tr_schedule,
    rot_schedule,
    tor_schedule,
    device,
    t_to_sigma,
    model_args,
    no_random=False,
    ode=False,
    visualization_list=None,
    confidence_model=None,
    confidence_data_list=None,
    confidence_model_args=None,
    t_schedule=None,
    batch_size=32,
    no_final_step_noise=False,
    pivot=None,
    return_full_trajectory=False,
    temp_sampling=1.0,
    temp_psi=0.0,
    temp_sigma_data=0.5,
    return_features=False,

    # NEW: Option-A simple physics guidance
    physics_potentials=None,
    physics_step_size: float = 0.05,
    use_physics: bool = True,
    physics_debug: bool = False,
    physics_trace: bool = False,
    physics_last_steps: int = -1,
):
    N = len(data_list)
    trajectory = []
    logger = get_logger()

    if return_features:
        lig_features, rec_features = [], []
        assert batch_size >= N, "Not implemented yet"

    loader = DataLoader(data_list, batch_size=batch_size)
    assert not (return_full_trajectory or return_features or pivot), "Not implemented yet"

    mask_rotate = torch.from_numpy(data_list[0]['ligand'].mask_rotate[0]).to(device)

    # Confidence model loader
    confidence = None
    if confidence_model is not None:
        confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
        confidence = []

    # Initialize physics if enabled
    if not use_physics:
        physics_potentials = None
    elif physics_potentials is None:
        physics_potentials = build_default_potentials()

    with torch.no_grad():
        for batch_id, complex_graph_batch in enumerate(loader):

            b = complex_graph_batch.num_graphs
            n = len(complex_graph_batch['ligand'].pos) // b
            complex_graph_batch = complex_graph_batch.to(device)

            for t_idx in range(inference_steps):

                # ===== Schedules =====
                t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]

                dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
                dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
                dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

                tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)

                # ===== Possibly crop =====
                if hasattr(model_args, 'crop_beyond') and model_args.crop_beyond is not None:
                    mod_complex_graph_batch = copy.deepcopy(complex_graph_batch).to_data_list()
                    for batch in mod_complex_graph_batch:
                        crop_beyond(batch, tr_sigma * 3 + model_args.crop_beyond, model_args.all_atoms)
                    mod_complex_graph_batch = Batch.from_data_list(mod_complex_graph_batch)
                else:
                    mod_complex_graph_batch = complex_graph_batch

                # ===== Set time =====
                set_time(
                    mod_complex_graph_batch,
                    t_schedule[t_idx] if t_schedule is not None else None,
                    t_tr,
                    t_rot,
                    t_tor,
                    b,
                    'all_atoms' in model_args and model_args.all_atoms,
                    device,
                )

                # ===== DiffDock score model =====
                tr_score, rot_score, tor_score = model(mod_complex_graph_batch)[:3]

                # NaN handling
                mean_scores = torch.mean(tr_score, dim=-1)
                num_nans = torch.sum(torch.isnan(mean_scores))
                if num_nans > 0:
                    name = complex_graph_batch['name']
                    if isinstance(name, list): name = name[0]
                    logger.warning(
                        f"Complex {name} Batch {batch_id+1} Iter {t_idx}: "
                        f"{num_nans}/{mean_scores.numel()} NaNs"
                    )
                    eps = 0.01 * torch.nanmean(tr_score.abs())
                    tr_score.nan_to_num_(nan=eps, posinf=eps, neginf=-eps)
                    rot_score.nan_to_num_(nan=eps, posinf=eps, neginf=-eps)
                    tor_score.nan_to_num_(nan=eps, posinf=eps, neginf=-eps)

                # ===== Noise schedules =====
                tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(
                    model_args.tr_sigma_max / model_args.tr_sigma_min)))
                rot_g = rot_sigma * torch.sqrt(torch.tensor(2 * np.log(
                    model_args.rot_sigma_max / model_args.rot_sigma_min)))

                # ===== Translation update =====
                if ode:
                    tr_perturb = 0.5 * tr_g**2 * dt_tr * tr_score
                    rot_perturb = 0.5 * rot_score * dt_rot * rot_g**2
                else:
                    tr_z = torch.zeros((min(batch_size, N), 3), device=device) \
                        if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.randn((min(batch_size, N), 3), device=device)

                    rot_z = torch.zeros((min(batch_size, N), 3), device=device) \
                        if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.randn((min(batch_size, N), 3), device=device)

                    tr_perturb = tr_g**2 * dt_tr * tr_score + tr_g * np.sqrt(dt_tr) * tr_z
                    rot_perturb = rot_g**2 * dt_rot * rot_score + rot_g * np.sqrt(dt_rot) * rot_z

                # ===== Torsion =====
                if not model_args.no_torsion:
                    tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(
                        model_args.tor_sigma_max / model_args.tor_sigma_min)))

                    if ode:
                        tor_perturb = 0.5 * tor_g**2 * dt_tor * tor_score
                    else:
                        tor_z = torch.zeros_like(tor_score, device=device) \
                            if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                            else torch.randn_like(tor_score, device=device)

                        tor_perturb = tor_g**2 * dt_tor * tor_score + tor_g * np.sqrt(dt_tor) * tor_z
                else:
                    tor_perturb = None

                # ===== Modify ligand coordinates (DiffDock step) =====
                candidate_pos_flat = modify_conformer_batch(
                    complex_graph_batch['ligand'].pos,
                    complex_graph_batch,
                    tr_perturb,
                    rot_perturb,
                    tor_perturb if not model_args.no_torsion else None,
                    mask_rotate,
                )

                # ============================================================
                #             Simple Physics Guidance
                # ============================================================
                apply_physics = (
                    use_physics
                    and physics_potentials is not None
                    and physics_step_size > 0.0
                    and (physics_last_steps < 0 or t_idx >= inference_steps - physics_last_steps)
                )

                if apply_physics:
                    coords_before_physics = candidate_pos_flat.clone() if physics_trace else None
                    full_grad = torch.zeros_like(candidate_pos_flat)

                    with torch.enable_grad():
                        for i in range(b):

                            global_idx = batch_id * batch_size + i
                            if global_idx >= len(data_list):
                                continue

                            start = i * n
                            end = (i + 1) * n
                            coords_i = candidate_pos_flat[start:end].detach().clone().requires_grad_(True)

                            feats_i = getattr(data_list[global_idx], "ligand_feats", None)
                            if feats_i is None:
                                continue

                            # Ensure features live on the same device as coords_i.
                            feats_i = {
                                key: (val.to(coords_i.device) if torch.is_tensor(val) else val)
                                for key, val in feats_i.items()
                            }

                            # Skip physics if feature indices don't match coordinate count.
                            max_bad = False
                            n_atoms_i = coords_i.shape[0]
                            for key in ("bond_index", "angle_index"):
                                if key in feats_i and torch.is_tensor(feats_i[key]) and feats_i[key].numel() > 0:
                                    if feats_i[key].max() >= n_atoms_i:
                                        max_bad = True
                                        break
                            if max_bad:
                                continue

                            E_i = coords_i.new_tensor(0.0)
                            debug_terms = []
                            for pot in physics_potentials:
                                E_i = E_i + pot.weight * pot.energy(coords_i, feats_i)
                                if physics_debug:
                                    debug_terms.append((pot.__class__.__name__, float(E_i.detach().cpu())))

                            (grad_i,) = torch.autograd.grad(E_i, coords_i, retain_graph=False)
                            full_grad[start:end] = grad_i.detach()

                            if physics_debug and batch_id == 0 and i == 0:
                                grad_norm = grad_i.detach().norm().item()
                                # Only log once per step to avoid spam.
                                logger.info(
                                    f"[Physics] step {t_idx}: grad_norm={grad_norm:.4f}; "
                                    + "; ".join([f"{n}={v:.4f}" for n, v in debug_terms])
                                )

                    candidate_pos_flat = candidate_pos_flat - physics_step_size * full_grad

                    if physics_trace and coords_before_physics is not None:
                        disp = candidate_pos_flat - coords_before_physics
                        # Report only for first sample in batch to keep logs concise.
                        disp_sample = disp[:n]
                        mean_disp = disp_sample.norm(dim=-1).mean().item()
                        max_disp = disp_sample.norm(dim=-1).max().item()
                        logger.info(
                            f"[PhysicsTrace] step {t_idx}: mean_disp={mean_disp:.4f} Å; max_disp={max_disp:.4f} Å"
                        )

                # Commit physics + diffdock update
                complex_graph_batch['ligand'].pos = candidate_pos_flat

                # Visualization
                if visualization_list is not None:
                    for idx_b in range(b):
                        visualization_list[batch_id * batch_size + idx_b].add(
                            (
                                complex_graph_batch['ligand'].pos[idx_b*n:n*(idx_b+1)].detach().cpu()
                                + data_list[batch_id * batch_size + idx_b].original_center.detach().cpu()
                            ),
                            part=1, order=t_idx + 2,
                        )

            # write back to data_list
            for i in range(b):
                data_list[batch_id * batch_size + i]['ligand'].pos = \
                    complex_graph_batch['ligand'].pos[i*n:(i+1)*n]

            # Visualization (unchanged)
            if visualization_list is not None:
                for idx, visualization in enumerate(visualization_list):
                    visualization.add(
                        data_list[idx]['ligand'].pos.detach().cpu()
                        + data_list[idx].original_center.detach().cpu(),
                        part=1, order=2,
                    )

            # Confidence model
            if confidence_model is not None:
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos.cpu()

                    if hasattr(confidence_model_args, 'crop_beyond') and confidence_model_args.crop_beyond is not None:
                        confidence_complex_graph_batch = confidence_complex_graph_batch.to_data_list()
                        for batch in confidence_complex_graph_batch:
                            crop_beyond(batch, confidence_model_args.crop_beyond, confidence_model_args.all_atoms)
                        confidence_complex_graph_batch = Batch.from_data_list(confidence_complex_graph_batch)

                    confidence_complex_graph_batch = confidence_complex_graph_batch.to(device)
                    set_time(confidence_complex_graph_batch, 0, 0, 0, 0, b, confidence_model_args.all_atoms, device)
                    out = confidence_model(confidence_complex_graph_batch)
                else:
                    out = confidence_model(complex_graph_batch)

                if isinstance(out, tuple):
                    out = out[0]

                confidence.append(out)

    if confidence_model is not None:
        confidence = torch.cat(confidence, dim=0)
        confidence = torch.nan_to_num(confidence, nan=-1000)

    if return_full_trajectory:
        return data_list, confidence, trajectory
    elif return_features:
        lig_features = torch.cat(lig_features, dim=0)
        rec_features = torch.cat(rec_features, dim=0)
        return data_list, confidence, lig_features, rec_features

    return data_list, confidence


def compute_affinity(data_list, affinity_model, affinity_data_list, device, parallel, all_atoms, include_miscellaneous_atoms):

    with torch.no_grad():
        if affinity_model is not None:
            assert parallel <= len(data_list)
            loader = DataLoader(data_list, batch_size=parallel)
            complex_graph_batch = next(iter(loader)).to(device)
            positions = complex_graph_batch['ligand'].pos

            assert affinity_data_list is not None
            complex_graph = affinity_data_list[0]
            N = complex_graph['ligand'].num_nodes
            complex_graph['ligand'].x = complex_graph['ligand'].x.repeat(parallel, 1)
            complex_graph['ligand'].edge_mask = complex_graph['ligand'].edge_mask.repeat(parallel)
            complex_graph['ligand', 'ligand'].edge_index = torch.cat(
                [N * i + complex_graph['ligand', 'ligand'].edge_index for i in range(parallel)], dim=1)
            complex_graph['ligand', 'ligand'].edge_attr = complex_graph['ligand', 'ligand'].edge_attr.repeat(parallel, 1)
            complex_graph['ligand'].pos = positions

            affinity_loader = DataLoader([complex_graph], batch_size=1)
            affinity_batch = next(iter(affinity_loader)).to(device)
            set_time(affinity_batch, 0, 0, 0, 0, 1, all_atoms, device, include_miscellaneous_atoms=include_miscellaneous_atoms)
            _, affinity = affinity_model(affinity_batch)
        else:
            affinity = None

    return affinity
