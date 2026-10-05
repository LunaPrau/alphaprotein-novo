# Copyright 2026 DeepMind Technologies Limited
#
# AlphaProtein Novo source code is licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structure metrics for protein design evaluation."""

import collections
from collections.abc import Mapping, Sequence
import logging
from typing import Any

from alphafold3 import structure
from alphafold3.cpp import membership
from alphafold3.cpp import string_array
from alphaprotein_novo.data import structure_utils
from alphaprotein_novo.metrics import alignment
from alphaprotein_novo.metrics import distance_difference
from alphaprotein_novo.metrics import geometry
from alphaprotein_novo.metrics import ligand_metrics
from alphaprotein_novo.metrics import structure_constants
import numpy as np

# ---------------------------------------------------------------------------
# Motif and Fixed Residue Metrics
# ---------------------------------------------------------------------------


def _get_matching_structures(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
) -> tuple[structure.Structure, structure.Structure]:
  """Returns decoy and reference structures with non-matching atoms removed."""
  decoy_structure_matched = decoy_structure.order_and_drop_atoms_to_match(
      reference_structure,
      allow_missing_atoms=True,
  )
  reference_structure_matched = (
      reference_structure.order_and_drop_atoms_to_match(
          decoy_structure_matched,
          allow_missing_atoms=True,
      )
  )
  num_dropped_atoms = (
      reference_structure.num_atoms - reference_structure_matched.num_atoms
  )
  if num_dropped_atoms > 0:
    logging.warning(
        '%d atoms were dropped from the reference structure because they were'
        ' not found in the decoy structure.',
        num_dropped_atoms,
    )
  return decoy_structure_matched, reference_structure_matched


def _align_on_atom_subset(
    *,
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    indices: np.ndarray,
) -> structure.Structure:
  """Align `decoy_structure` to `reference_structure` over the subset of atoms at `indices`."""
  new_coords = geometry.align(
      x=decoy_structure.coords,
      y=reference_structure.coords,
      x_indices=indices,
      y_indices=indices,
  )
  decoy_aligned = decoy_structure.copy_and_update_coords(coords=new_coords)

  return decoy_aligned


def flip_symmetric_sidechains(
    struct: structure.Structure,
    equivalent_atoms: Mapping[str, Mapping[str, str]],
) -> structure.Structure:
  """Swaps coordinates of all atoms with their symmetry partners based on sidechain flips.

  Args:
    struct: Structure to swap coordinates in.
    equivalent_atoms: Mapping from 3-letter amino acid names to a mapping
      between an atom name and its permutable partner.

  Returns:
    Structure with coordinates of each atom swapped with its symmetry
    alternative.
  """
  alt_coords = struct.coords.copy()
  for res_name, res_equivalent_atoms in equivalent_atoms.items():
    for k, v in res_equivalent_atoms.items():
      mask1 = (struct.res_name == res_name) & (struct.atom_name == k)
      mask2 = (struct.res_name == res_name) & (struct.atom_name == v)
      if mask1.sum() == mask2.sum() and mask2.sum() > 0:
        alt_coords[mask1] = struct.coords[mask2]

  struct_alt = struct.copy_and_update_coords(coords=alt_coords)

  return struct_alt


def rmsd_with_atom_permutations(
    *,
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    equivalent_atoms: Mapping[str, Mapping[str, str]] | None = None,
) -> float:
  """Computes RMSD accounting for atom permutations due to sidechain flips.

  Does not align structures, so this must be done before calling this function.

  Args:
    decoy_structure: Structure to get metrics for.
    reference_structure: Structure to compare against.
    equivalent_atoms: Mapping from 3-letter amino acid names to a mapping
      between an atom name and its permutable partner. If None, then RMSD is
      computed without accounting for atom permutations.

  Returns:
    RMSD over all atoms between decoy and reference.
  """
  res_arrays = reference_structure.to_res_arrays(include_missing_residues=False)
  ref_coords = res_arrays.atom_positions
  atom_mask = res_arrays.atom_mask
  if equivalent_atoms:
    reference_structure_alt = flip_symmetric_sidechains(
        reference_structure, equivalent_atoms
    )
    ref_coords_alt = reference_structure_alt.to_res_arrays(
        include_missing_residues=False
    ).atom_positions
    ref_coords_both = np.stack(
        [ref_coords, ref_coords_alt]
    )  # (2, Nres, Natoms, 3)
  else:
    ref_coords_both = ref_coords[None]  # (1, Nres, Natoms, 3)

  decoy_coords = decoy_structure.to_res_arrays(
      include_missing_residues=False
  ).atom_positions

  deviations = np.sum(
      np.square(decoy_coords - ref_coords_both), axis=-1
  )  # (Nperm, Nres, Natoms)
  res_deviations = np.sum(deviations * atom_mask, axis=-1)  # (Nperm, Nres)
  min_res_deviations = np.min(res_deviations, axis=0)  # (Nres,)
  rmsd = np.sqrt(np.sum(min_res_deviations) / atom_mask.sum())

  return rmsd


def protein_motif_metrics(
    *,
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    fixed_atom_mask: np.ndarray | None = None,
    compute_ligand_metrics: bool = False,
    expand_motif_to_full_residues: bool = False,
) -> Mapping[str, float]:
  """Computes RMSD metrics over protein motifs.

  RMSDs over sidechain atoms account for symmetry flips.

  Args:
    decoy_structure: Structure to obtain metrics for.
    reference_structure: Structure that defines what to align/compare on. The
      `reference_structure` must contain a subset of the atoms of the
      `decoy_structure`. Any filtering specified by restricting options is done
      on the `reference_structure`.
    fixed_atom_mask: Mask with 1's for atoms of `reference_structure` that
      represent the motif, and 0's otherwise. If `None`, then the
      `atom_b_factor` field of `reference_structure` will be used instead.
    compute_ligand_metrics: Whether to compute motif-with-ligand and
      ligand-aligned motif RMSD metrics.
    expand_motif_to_full_residues: Whether to expand the motif to full residues.

  Returns:
    Mapping of metric names to metrics.

  Raises:
    ValueError: If the atom_b_factor field does not cast to a 0/1 binary mask.
    ValueError: If the fixed atom mask is not set in the atom_b_factor field of
      the`reference_structure`.
  """
  if fixed_atom_mask is None:
    if reference_structure.atom_b_factor is None:
      raise ValueError(
          'Reference structure must have atom_b_factor field set to a fixed'
          ' atom mask, but it is None.'
      )
    fixed_atom_mask = reference_structure.atom_b_factor.astype(np.int32)
    if not membership.isin(fixed_atom_mask, {0, 1}).all():
      raise ValueError(
          'Expected the `atom_b_factor` field to represent a fixed atom mask'
          ' with 0/1 elements.'
      )

  output_metrics = {}

  # Filter to fixed atoms on the protein.
  reference_motif = reference_structure.filter(
      fixed_atom_mask == 1
  ).filter_to_entity_type(protein=True)

  if expand_motif_to_full_residues:
    reference_motif = structure_utils.filter_to_residues_with_any_atom_fixed(
        reference_structure
    ).filter_to_entity_type(protein=True)

  ref_motif_res = set(zip(reference_motif.chain_id, reference_motif.res_id))
  decoy_protein = decoy_structure.filter_to_entity_type(protein=True)
  decoy_res = set(zip(decoy_protein.chain_id, decoy_protein.res_id))
  output_metrics['num_missing_motif_residues'] = len(ref_motif_res - decoy_res)

  try:
    decoy_motif = decoy_structure.order_and_drop_atoms_to_match(
        reference_motif,
    )
    reference_motif_matched = reference_motif
    # If no MissingAtomError is raised, then the decoy motif has all the atoms
    # of the reference motif.
    output_metrics['num_missing_atoms'] = 0
  except structure.MissingAtomError:
    # Filter decoy and reference to the available matching fixed motif atoms.
    decoy_motif = decoy_structure.order_and_drop_atoms_to_match(
        reference_motif, allow_missing_atoms=True
    )
    if decoy_motif.num_atoms > 0:
      reference_motif_matched = reference_motif.order_and_drop_atoms_to_match(
          decoy_motif, allow_missing_atoms=True
      )
    else:
      # Fall back to atom_b_factor if present on decoy.
      decoy_fixed_atom_mask = decoy_structure.atom_b_factor.astype(np.int32)
      decoy_motif = decoy_structure.filter(
          decoy_fixed_atom_mask == 1
      ).filter_to_entity_type(protein=True)
      if decoy_motif.num_atoms == reference_motif.num_atoms:
        reference_motif_matched = reference_motif
      else:
        reference_motif_matched = reference_motif.filter(
            np.zeros(reference_motif.num_atoms, dtype=bool)
        )

    output_metrics['num_missing_atoms'] = max(
        0, reference_motif.num_atoms - decoy_motif.num_atoms
    )

  if (
      reference_motif_matched.num_atoms == 0
      or decoy_motif.num_atoms != reference_motif_matched.num_atoms
  ):
    output_metrics['motif_allatom_rmsd'] = np.nan
    output_metrics['motif_bb_aligned_allatom_rmsd'] = np.nan
  else:
    aligned_decoy_motif = _align_on_atom_subset(
        decoy_structure=decoy_motif,
        reference_structure=reference_motif_matched,
        indices=np.where(reference_motif_matched.atom_name)[0],
    )
    output_metrics['motif_allatom_rmsd'] = rmsd_with_atom_permutations(
        decoy_structure=aligned_decoy_motif,
        reference_structure=reference_motif_matched,
        equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
    )

    reference_motif_bb = reference_motif_matched.filter(
        atom_name=structure_constants.PROTEIN_BACKBONE_ATOMS
    )
    if reference_motif_bb.num_atoms == 0:
      output_metrics['motif_bb_aligned_allatom_rmsd'] = np.nan
    else:
      decoy_aligned = _align_on_atom_subset(
          decoy_structure=decoy_motif,
          reference_structure=reference_motif_matched,
          indices=np.where(
              string_array.isin(
                  reference_motif_matched.atom_name,
                  set(structure_constants.PROTEIN_BACKBONE_ATOMS),
              )
          )[0],
      )
      output_metrics['motif_bb_aligned_allatom_rmsd'] = (
          rmsd_with_atom_permutations(
              decoy_structure=decoy_aligned,
              reference_structure=reference_motif_matched,
              equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
          )
      )

  output_metrics['motif_with_ligand_allatom_rmsd'] = np.nan
  output_metrics['ligand_aligned_motif_allatom_rmsd'] = np.nan
  if compute_ligand_metrics:
    reference_motif_with_ligand = structure.concat([
        reference_motif,
        reference_structure.filter_to_entity_type(ligand=True),
    ])
    decoy_motif_with_ligand = structure.concat(
        [decoy_motif, decoy_structure.filter_to_entity_type(ligand=True)]
    )
    try:
      decoy_motif_with_ligand = (
          decoy_motif_with_ligand.order_and_drop_atoms_to_match(
              reference_motif_with_ligand
          )
      )
    except structure.MissingAtomError:
      decoy_motif_with_ligand, reference_motif_with_ligand = (
          _get_matching_structures(
              decoy_structure=decoy_motif_with_ligand,
              reference_structure=reference_motif_with_ligand,
          )
      )

    if (
        reference_motif_with_ligand.filter_to_entity_type(ligand=True).num_atoms
        > 0
    ):
      decoy_aligned = _align_on_atom_subset(
          decoy_structure=decoy_motif_with_ligand,
          reference_structure=reference_motif_with_ligand,
          indices=np.where(reference_motif_with_ligand.atom_name)[0],
      )
      output_metrics['motif_with_ligand_allatom_rmsd'] = (
          rmsd_with_atom_permutations(
              decoy_structure=decoy_aligned,
              reference_structure=reference_motif_with_ligand,
              equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
          )
      )

    if (
        reference_motif_with_ligand.filter_to_entity_type(ligand=True).num_atoms
        > 1
    ):
      decoy_aligned_on_ligand = _align_on_atom_subset(
          decoy_structure=decoy_motif_with_ligand,
          reference_structure=reference_motif_with_ligand,
          indices=np.where(reference_motif_with_ligand.is_ligand_mask)[0],
      )
      output_metrics['ligand_aligned_motif_allatom_rmsd'] = (
          rmsd_with_atom_permutations(
              decoy_structure=decoy_aligned_on_ligand,
              reference_structure=reference_motif_with_ligand,
              equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
          )
      )

  return output_metrics


def fixed_residue_metrics(
    *,
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
) -> dict[str, float]:
  """Computes metrics based on fixed residues as inferred by `atom_b_factor`.

  Considers a residue as fixed if any of its atoms have a b_factor of 1.

  Args:
    decoy_structure: Structure to compute metrics for.
    reference_structure: Structure that defines what to align/compare on. The
      `reference_structure` must contain a subset of the atoms of the
      `decoy_structure`.

  Returns:
    Dictionary of fixed residue metrics.
  """
  decoy_protein_struct = decoy_structure.filter_to_entity_type(protein=True)
  reference_protein_struct = reference_structure.filter_to_entity_type(
      protein=True
  )

  ret = {}
  ret['fixed_res_bb_aligned_allatom_sc_rmsd'] = np.nan
  ret['fixed_res_bb_aligned_fixed_atom_rmsd'] = np.nan
  ret['fixed_res_bb_aligned_bb_sc_rmsd'] = np.nan
  if reference_protein_struct.atom_b_factor is not None and np.all(
      membership.isin(
          reference_protein_struct.atom_b_factor.astype(np.int32),
          {0, 1},
      )
  ):
    try:
      ref_motif_fixed_residues = (
          structure_utils.filter_to_residues_with_any_atom_fixed(
              reference_protein_struct
          )
      )
      ref_motif_fixed_residues_bb = ref_motif_fixed_residues.filter(
          atom_name=structure_constants.PROTEIN_BACKBONE_ATOMS
      )

      decoy_motif_fixed_residues = (
          decoy_protein_struct.order_and_drop_atoms_to_match(
              ref_motif_fixed_residues
          )
      )
      decoy_motif_fixed_residues_bb = (
          decoy_protein_struct.order_and_drop_atoms_to_match(
              ref_motif_fixed_residues_bb
          )
      )

      if ref_motif_fixed_residues.num_atoms > 1:
        bb_aligned_decoy_motif = _align_on_atom_subset(
            decoy_structure=decoy_motif_fixed_residues,
            reference_structure=ref_motif_fixed_residues,
            indices=np.where(
                string_array.isin(
                    ref_motif_fixed_residues.atom_name,
                    set(structure_constants.PROTEIN_BACKBONE_ATOMS),
                )
            )[0],
        )
        ret['fixed_res_bb_aligned_allatom_sc_rmsd'] = (
            rmsd_with_atom_permutations(
                decoy_structure=bb_aligned_decoy_motif,
                reference_structure=ref_motif_fixed_residues,
                equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
            )
        )
        ref_motif_fixed_atoms = ref_motif_fixed_residues.filter(
            ref_motif_fixed_residues.atom_b_factor == 1
        )
        bb_aligned_decoy_motif_fixed_atoms = (
            bb_aligned_decoy_motif.order_and_drop_atoms_to_match(
                ref_motif_fixed_atoms
            )
        )
        ret['fixed_res_bb_aligned_fixed_atom_rmsd'] = (
            rmsd_with_atom_permutations(
                decoy_structure=bb_aligned_decoy_motif_fixed_atoms,
                reference_structure=ref_motif_fixed_atoms,
                equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
            )
        )
      else:
        ret['fixed_res_bb_aligned_allatom_sc_rmsd'] = np.nan
        ret['fixed_res_bb_aligned_fixed_atom_rmsd'] = np.nan
      if ref_motif_fixed_residues_bb.num_atoms > 1:
        bb_aligned_decoy_motif_bb = _align_on_atom_subset(
            decoy_structure=decoy_motif_fixed_residues_bb,
            reference_structure=ref_motif_fixed_residues_bb,
            indices=np.where(ref_motif_fixed_residues_bb.atom_name)[0],
        )
        ret['fixed_res_bb_aligned_bb_sc_rmsd'] = rmsd_with_atom_permutations(
            decoy_structure=bb_aligned_decoy_motif_bb,
            reference_structure=ref_motif_fixed_residues_bb,
            equivalent_atoms=structure_constants.EQUIVALENT_SIDECHAIN_ATOMS,
        )
      else:
        ret['fixed_res_bb_aligned_bb_sc_rmsd'] = np.nan
    except (structure.MissingAtomError, ValueError) as e:
      logging.warning('Failed to compute fixed residue metrics: %s', e)
      ret['fixed_res_bb_aligned_allatom_sc_rmsd'] = np.nan
      ret['fixed_res_bb_aligned_fixed_atom_rmsd'] = np.nan
      ret['fixed_res_bb_aligned_bb_sc_rmsd'] = np.nan
  return ret


def ligand_atom_neighbors(
    struct: structure.Structure,
    ca_dist_cutoffs: Sequence[int] = (6, 8, 12),
    allatom_dist_cutoffs: Sequence[int] = (4, 6, 8),
) -> tuple[Sequence[float], Sequence[float]]:
  """Count number of protein atoms that are near ligand atoms.

  Args:
    struct: Structure to analyze.
    ca_dist_cutoffs: Distances at which to count C-alpha neighbors of ligand
      atoms.
    allatom_dist_cutoffs: Distances at which to count heavy atom neighbors of
      ligand atoms.

  Returns:
    ca_neighbors: Mean over all cutoff distances of the number of C-alpha
      neighbors, list of values for each ligand atom.
    allatom_neighbors: Mean over all cutoff distances of the number of heavy
      atom neighbors, list of values for each ligand atom.
  """
  ligand = struct.filter_to_entity_type(ligand=True)
  return structure_utils.atom_neighbors(
      struct=struct,
      atoms=structure_utils.get_atoms(ligand),
      ca_dist_cutoffs=ca_dist_cutoffs,
      allatom_dist_cutoffs=allatom_dist_cutoffs,
  )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def structure_metrics(
    *,
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    gt_structure: structure.Structure | None = None,
    compute_ligand_metrics: bool = True,
    compute_motif_metrics: bool = True,
) -> dict[str, Any]:
  """Compute some accuracy measures for positions.

  Note that alignment metrics might not be symmetric, and hence it makes a
  difference which structure is passed as the reference structure.
  For example, to compute lddt, the reference structure defines what residues
  to compare the two structures on.

  Args:
    decoy_structure: Structure to obtain metrics for.
    reference_structure: Structure that defines what to align/compare on.
    gt_structure: Ground truth structure that contains the ground truth motif
      atom positions. The non-motif atoms can just have zeros for the coords. If
      this is not provided, motif metrics will be calculated with respect to the
      reference_structure.
    compute_ligand_metrics: If True, compute the ligand metrics. This requires
      both Structures to have the same atom types to align.
    compute_motif_metrics: If True, compute motif metrics. This requires that
      `gt_structure.atom_b_factor` (or `reference_structure.atom_b_factor`)
      contains 0 or 1 indicating whether the atom is part of a fixed motif.

  Returns:
    Metrics based on the positions of the two structures.
  """

  decoy_structure_protein_only = decoy_structure.filter_to_entity_type(
      protein=True
  )
  reference_structure_protein_only = reference_structure.filter_to_entity_type(
      protein=True
  )

  ret = {}
  alignment_scores_result = alignment.alignment_scores(
      decoy_structure_protein_only,
      reference_structure_protein_only,
  )
  ret['gdt_ha'] = alignment_scores_result.gdt_ha[0]
  ret['tm_score'] = alignment_scores_result.tm_score[0]
  ret['rmsd'] = alignment_scores_result.rmsd[0]
  ret['lddt'] = distance_difference.lddt(
      decoy_structure_protein_only,
      reference_structure_protein_only,
      one_atom_per_residue=True,
  ).lddt
  has_any_ligand = (
      decoy_structure.filter_to_entity_type(ligand=True).num_atoms > 0
      or reference_structure.filter_to_entity_type(ligand=True).num_atoms > 0
  )
  ligands_match = ligand_metrics.has_matching_ligands(
      decoy_structure, reference_structure
  )
  if (
      decoy_structure.num_chains >= 2
      and reference_structure.num_chains >= 2
      and (not has_any_ligand or ligands_match)
  ):
    try:
      ret['interface_lddt'] = distance_difference.lddt(
          decoy_structure,
          reference_structure,
          one_atom_per_residue=True,
          multichain_mode=distance_difference.MultiChainMode.BETWEEN_CHAINS_ONLY,
      ).lddt
    except (structure.MissingAtomError, ValueError) as e:
      logging.warning('Failed to compute interface_lddt: %s', e)
      ret['interface_lddt'] = np.nan
  else:
    ret['interface_lddt'] = np.nan

  # Motif metrics.
  if compute_motif_metrics:
    if gt_structure is None:
      gt_structure = reference_structure
    decoy_structure_matched, reference_structure_matched = (
        _get_matching_structures(
            decoy_structure_protein_only,
            reference_structure_protein_only,
        )
    )
    decoy_structure_with_updated_b_factors = (
        decoy_structure_matched.copy_and_update_atoms(
            atom_b_factor=reference_structure_matched.atom_b_factor
        )
    )
    decoy_ligand = decoy_structure.filter_to_entity_type(ligand=True)
    if decoy_ligand.num_atoms > 0:
      decoy_structure_with_updated_b_factors = structure.concat([
          decoy_structure_with_updated_b_factors,
          decoy_ligand.copy_and_update_atoms(
              atom_b_factor=np.zeros(decoy_ligand.num_atoms)
          ),
      ])
    ret.update(
        protein_motif_metrics(
            decoy_structure=decoy_structure_with_updated_b_factors,
            reference_structure=gt_structure,
            compute_ligand_metrics=compute_ligand_metrics,
        )
    )
    if (
        reference_structure.atom_b_factor is not None
        and membership.isin(
            reference_structure.atom_b_factor.astype(np.int32), {0, 1}
        ).all()
    ):
      ret.update({
          f'expanded_motif/{k}': v
          for k, v in protein_motif_metrics(
              decoy_structure=decoy_structure_with_updated_b_factors,
              reference_structure=reference_structure,
              compute_ligand_metrics=compute_ligand_metrics,
              expand_motif_to_full_residues=True,
          ).items()
      })

  # Fixed residue metrics.
  fixed_metrics = fixed_residue_metrics(
      decoy_structure=decoy_structure,
      reference_structure=reference_structure,
  )
  ret['fixed_res_bb_aligned_allatom_sc_rmsd'] = fixed_metrics.get(
      'fixed_res_bb_aligned_allatom_sc_rmsd', np.nan
  )

  # Ligand metrics.
  if compute_ligand_metrics:
    decoy_ligand = decoy_structure.filter_to_entity_type(ligand=True)
    if len(decoy_ligand.chains) >= 1:
      if ligands_match:
        try:
          ligand_pocket_aligned_metrics = (
              ligand_metrics.compute_pocket_aligned_metrics(
                  decoy_structure, reference_structure
              )
          )
          rmsd_per_ligand_chain_id = (
              ligand_pocket_aligned_metrics.rmsd_per_ligand_chain_id
          )
          rmsd_per_ligand_values = list(rmsd_per_ligand_chain_id.values())
          ret['mean_pocket_bb_aligned_ligand_rmsd'] = np.mean(
              rmsd_per_ligand_values
          )
          rmsds_per_ligand = collections.defaultdict(list)
          for chain_id, rmsd in rmsd_per_ligand_chain_id.items():
            ligand_name = decoy_ligand.filter(chain_id=chain_id).res_name[0]
            rmsds_per_ligand[ligand_name].append(rmsd)
          for ligand_name, rmsd_list in rmsds_per_ligand.items():
            ret[f'pocket_bb_aligned_ligand_rmsd/{ligand_name}'] = np.mean(
                rmsd_list
            )
          ret['mean_pocket_bb_rmsd'] = np.mean(
              ligand_pocket_aligned_metrics.pocket_bb_rmsds
          )
        except (structure.MissingAtomError, ValueError) as e:
          logging.warning(
              'Failed to compute ligand pocket aligned metrics: %s', e
          )
          ret['mean_pocket_bb_aligned_ligand_rmsd'] = np.nan
          ret['mean_pocket_bb_rmsd'] = np.nan

      try:
        ret['percent_ligand_bb_clashes_1_5'] = (
            ligand_metrics.percent_backbone_clashes(
                decoy_structure, threshold=1.5
            )
        )

        # Ligand atom neighbors.
        ca_neighbors, _ = ligand_atom_neighbors(decoy_structure)
        ret['ligand_ca_neighbors'] = ca_neighbors
        ret['ligand_ca_neighbors_min'] = np.min(ca_neighbors)
        ret['ligand_ca_neighbors_mean'] = float(np.mean(ca_neighbors))
      except (structure.MissingAtomError, ValueError) as e:
        logging.warning(
            'Failed to compute ligand clash/neighbor metrics: %s', e
        )
        ret['percent_ligand_bb_clashes_1_5'] = np.nan
        ret['ligand_ca_neighbors'] = []
        ret['ligand_ca_neighbors_min'] = np.nan
        ret['ligand_ca_neighbors_mean'] = np.nan

  return ret
