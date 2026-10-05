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

"""Unindexed motif matching, reference alignment, and sample preparation."""

import collections
from collections.abc import Mapping
import dataclasses
import json
from typing import Any

from alphafold3 import structure
from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.data import motif_spec
from alphaprotein_novo.data import residue_mapping
from alphaprotein_novo.model import types as types_lib
import numpy as np
from scipy import optimize

ATOM37_ATOM_NAMES = atom_types.ATOM37
ATOM37_NAME_TO_IDX = {name: idx for idx, name in enumerate(ATOM37_ATOM_NAMES)}


@dataclasses.dataclass(frozen=True, kw_only=True)
class UnindexedMotifData:
  """Container for unindexed motif data needed for geometric matching.

  Attributes:
    unindexed_motif_atom_positions: Motif atom positions, shape [N, 37, 3].
    unindexed_motif_aatype: Motif amino acid types, shape [N].
    unindexed_motif_atom_mask: Motif atom mask, shape [N, 37].
    unindexed_motif_aatype_mask: Motif residue mask, shape [N].
  """

  unindexed_motif_atom_positions: np.ndarray
  unindexed_motif_aatype: np.ndarray
  unindexed_motif_atom_mask: np.ndarray
  unindexed_motif_aatype_mask: np.ndarray

  @classmethod
  def from_diffusion_input(
      cls, dinput: types_lib.DiffusionInput
  ) -> 'UnindexedMotifData':
    """Extracts UnindexedMotifData from DiffusionInput."""
    if dinput.unindexed_motif_atom_positions is None:
      raise ValueError(
          'DiffusionInput does not contain unindexed_motif_atom_positions.'
      )
    aatype = np.asarray(dinput.unindexed_motif_aatype)
    aatype_mask = (
        np.asarray(dinput.unindexed_motif_aatype_mask, dtype=bool)
        if dinput.unindexed_motif_aatype_mask is not None
        else np.ones(len(aatype), dtype=bool)
    )
    return cls(
        unindexed_motif_atom_positions=np.asarray(
            dinput.unindexed_motif_atom_positions
        ),
        unindexed_motif_aatype=aatype,
        unindexed_motif_atom_mask=np.asarray(dinput.unindexed_motif_atom_mask),
        unindexed_motif_aatype_mask=aatype_mask,
    )


@dataclasses.dataclass
class UnindexedMatchingResult:
  """Result of matching unindexed motif residues to sampled structure.

  Attributes:
    diffused_index_map: Map from motif residue index to "chain_id+res_id" string
    insertion_rmsd: Overall RMSD across all matched residues
    insertion_rmsd_by_residue: RMSD for each motif residue index
    matched_residue_ids: List of res_ids in sampled_struct that were matched
    motif_atom_mask: Per-atom mask indicating which atoms are motif atoms (value
      > 0 for motif atoms, 0 for non-motif atoms). Same shape as
      sampled_struct.atom_b_factor.
  """

  diffused_index_map: dict[int, str]
  insertion_rmsd: float
  insertion_rmsd_by_residue: dict[int, float]
  matched_residue_ids: list[int]
  motif_atom_mask: np.ndarray

  def get_motif_residue_map(
      self,
      unindexed_motif_residues: str,
  ) -> dict[tuple[str, int], tuple[str, int]]:
    """Get mapping from initial motif residues to sampled structure residues.

    Args:
      unindexed_motif_residues: The unindexed motif residues string containing
        source motif residue identities (e.g., "A12,A15-17").

    Returns:
      Dictionary mapping source (chain_id, res_id) to target (chain_id, res_id).
    """
    source_residues = list(
        residue_mapping.chain_and_residue_pairs(unindexed_motif_residues)
    )
    motif_residue_map = {}
    for motif_idx, chain_res_str in self.diffused_index_map.items():
      if motif_idx < len(source_residues):
        source_chain, source_res = source_residues[motif_idx]
        target_chain = chain_res_str[0]
        target_res = int(chain_res_str[1:])
        motif_residue_map[(source_chain, source_res)] = (
            target_chain,
            target_res,
        )
      else:
        raise ValueError(
            f'Motif residue index {motif_idx} is out of bounds for source'
            f' residues {len(source_residues)}.'
        )
    return motif_residue_map


@dataclasses.dataclass(frozen=True)
class UnindexedMotifPreparationResult:
  """Container for outputs of unindexed motif preparation.

  Attributes:
    sampled_struct: Sampled structure with atom_b_factor updated to indicate
      motif and ligand fixed atoms (1.0 for fixed, 0.0 for non-fixed).
    reference_motif_struct: Reference structure containing the original motif
      coordinates aligned to sampled_struct's atom ordering, with
      atom_b_factor=1.0 for motif/ligand atoms.
    sampled_motif_str: Reconstructed motif string indicating positions of all
      matched motif residues and fixed chains.
    residue_map: Updated ResidueMap reflecting matched motif positions.
    fixed_seq_mask: Boolean/integer sequence mask indicating which residues
      should remain fixed during sequence redesign (LigandMPNN).
    fixed_atom_mask: Boolean/integer per-atom mask.
    metrics: Dictionary of diagnostic metrics (insertion RMSD, unmatched counts,
      predicted vs matched comparisons).
  """

  sampled_struct: structure.Structure
  reference_motif_struct: structure.Structure
  sampled_motif_str: str
  residue_map: residue_mapping.ResidueMap | None
  fixed_seq_mask: np.ndarray
  fixed_atom_mask: np.ndarray
  metrics: dict[str, Any]


def process_unindexed_outputs(
    sampled_struct: structure.Structure,
    *,
    unindexed_motif_atom_positions: np.ndarray,
    unindexed_motif_aatype: np.ndarray,
    unindexed_motif_atom_mask: np.ndarray,
    unindexed_motif_aatype_mask: np.ndarray | None = None,
    distance_threshold: float = 200.0,
    force_correct_aatype: bool = True,
) -> UnindexedMatchingResult:
  """Match unindexed motif residues using proximity-based matching.

  For each motif residue, atoms are matched to protein atoms in the sampled
  structure using linear_sum_assignment on distance matrices, constrained by
  matching atom names and amino acid types. The closest matched residue
  is selected for each motif residue.

  Args:
    sampled_struct: Diffusion output structure.
    unindexed_motif_atom_positions: Pre-aligned motif atom positions, shape
      [num_motif_res, num_atoms, 3].
    unindexed_motif_aatype: Motif amino acid types, shape [num_motif_res].
    unindexed_motif_atom_mask: Motif atom mask, shape [num_motif_res,
      num_atoms].
    unindexed_motif_aatype_mask: Mask for valid motif residues, shape
      [num_motif_res].
    distance_threshold: Maximum distance (Angstroms) for a valid match.
    force_correct_aatype: If True, only match motif residues to protein residues
      with the same amino acid type. Default is True.

  Returns:
    UnindexedMatchingResult with RMSD metrics and matched structure atom masks.
  """
  motif_positions = np.array(unindexed_motif_atom_positions)
  motif_aatypes = np.array(unindexed_motif_aatype)
  motif_atom_mask = np.array(unindexed_motif_atom_mask)

  if unindexed_motif_aatype_mask is None:
    motif_aatype_mask = np.ones(len(motif_aatypes), dtype=bool)
  else:
    motif_aatype_mask = np.array(unindexed_motif_aatype_mask, dtype=bool)

  # Motif reseq residues will have a nonzero motif_atom_mask (from backbone
  # atoms), but will have zero aatype and zero motif_aatype_mask.
  motif_reseq_res_mask = ~motif_aatype_mask & (motif_atom_mask.sum(axis=1) > 0)

  # Remove ligand aatypes if they come after the protein motif segment.
  invalid_indices = np.where(motif_aatypes > 19)[0]
  if invalid_indices.size > 0:
    first_invalid = invalid_indices[0]
    motif_aatype_mask[first_invalid:] = False

  num_motif_residues = int((motif_aatype_mask | motif_reseq_res_mask).sum())

  sampled_struct_protein = sampled_struct.filter_to_entity_type(protein=True)
  protein_coords = sampled_struct_protein.coords
  protein_atom_names = sampled_struct_protein.atom_name
  protein_res_ids = sampled_struct_protein.res_id
  protein_chain_ids = sampled_struct_protein.chain_id
  protein_res_names = sampled_struct_protein.res_name

  diffused_index_map = {}
  insertion_rmsd_by_residue = {}
  matched_residue_ids = []
  motif_to_residue_matches = {}
  all_rmsds = []
  used_residues = set()

  for motif_idx in range(num_motif_residues):
    motif_mask = motif_atom_mask[motif_idx]
    if not motif_mask.any():
      continue

    motif_atom_indices = np.where(motif_mask)[0]
    motif_res_coords = motif_positions[motif_idx][motif_atom_indices]
    motif_res_atom_names = np.array(
        [ATOM37_ATOM_NAMES[i] for i in motif_atom_indices]
    )
    motif_reseq_res = motif_reseq_res_mask[motif_idx]

    # Filter out protein atoms from already-matched residues.
    available_mask = np.ones(len(protein_coords), dtype=bool)
    for chain_id, res_id in used_residues:
      residue_mask = (protein_res_ids == res_id) & (
          protein_chain_ids == chain_id
      )
      available_mask &= ~residue_mask

    # Filter to only residues with matching aatype (unless motif is reseq).
    if force_correct_aatype and not motif_reseq_res:
      motif_aa_1letter = residue_names.PROTEIN_TYPES_ONE_LETTER[
          motif_aatypes[motif_idx]
      ]
      motif_aa_3letter = residue_names.PROTEIN_COMMON_ONE_TO_THREE[
          motif_aa_1letter
      ]
      aatype_mask = protein_res_names == motif_aa_3letter
      available_mask &= aatype_mask

    # Skip if no available atoms
    if not available_mask.any():
      continue

    # Compute distances only to available atoms
    available_coords = protein_coords[available_mask]
    available_atom_names = protein_atom_names[available_mask]
    available_res_ids = protein_res_ids[available_mask]
    available_chain_ids = protein_chain_ids[available_mask]

    dists = np.linalg.norm(
        motif_res_coords[:, None, :] - available_coords[None, :, :], axis=-1
    )

    matching_atom_name = (
        motif_res_atom_names[:, None] == available_atom_names[None, :]
    )
    dists[~matching_atom_name] = np.inf
    dists[dists > distance_threshold] = np.inf

    if not np.isfinite(dists).any():
      continue

    try:
      row_ind, col_ind = optimize.linear_sum_assignment(dists)
    except ValueError as e:
      if 'cost matrix is infeasible' in str(e):
        continue
      raise

    valid_matches = dists[row_ind, col_ind] < np.inf
    if not valid_matches.any():
      continue

    col_ind_valid = col_ind[valid_matches]
    matched_res_ids = available_res_ids[col_ind_valid]
    matched_chain_ids = available_chain_ids[col_ind_valid]

    # If atoms span multiple residues, pick majority.
    if len(set(matched_res_ids)) > 1 or len(set(matched_chain_ids)) > 1:
      pair_counts = collections.Counter(zip(matched_chain_ids, matched_res_ids))
      (chain_id, res_id), _ = pair_counts.most_common(1)[0]
    else:
      res_id = matched_res_ids[0]
      chain_id = matched_chain_ids[0]

    # Doublecheck this residue wasn't already matched.
    if (chain_id, res_id) in used_residues:
      continue

    # Refine assignment to specific residue.
    token_match = (protein_res_ids == res_id) & (protein_chain_ids == chain_id)
    dists_refined = np.full(
        (len(motif_res_coords), len(protein_coords)), np.inf, dtype=np.float32
    )

    token_indices = np.where(token_match)[0]
    dists_refined[:, token_indices] = np.linalg.norm(
        motif_res_coords[:, None, :] - protein_coords[None, token_indices, :],
        axis=-1,
    )

    # Apply atom name and distance filtering
    matching_atom_name_refined = (
        motif_res_atom_names[:, None] == protein_atom_names[None, :]
    )
    dists_refined[~matching_atom_name_refined] = np.inf
    dists_refined[dists_refined > distance_threshold] = np.inf

    try:
      row_ind_final, col_ind_final = optimize.linear_sum_assignment(
          dists_refined
      )
    except ValueError as e:
      if 'cost matrix is infeasible' in str(e):
        continue
      raise

    valid_final = dists_refined[row_ind_final, col_ind_final] < np.inf
    if valid_final.any():
      diff = (
          motif_res_coords[row_ind_final[valid_final]]
          - protein_coords[col_ind_final[valid_final]]
      )
      residue_rmsd = float(np.sqrt((diff**2).sum(-1).mean()))
    else:
      residue_rmsd = float('nan')

    matched_indices = motif_atom_indices[row_ind_final[valid_final]]

    motif_to_residue_matches[motif_idx] = (res_id, chain_id, matched_indices)
    matched_residue_ids.append(res_id)
    diffused_index_map[motif_idx] = f'{chain_id}{res_id}'
    insertion_rmsd_by_residue[motif_idx] = residue_rmsd
    if np.isfinite(residue_rmsd):
      all_rmsds.append(residue_rmsd)

    # Mark this residue as used to prevent re-matching
    used_residues.add((chain_id, res_id))

  if all_rmsds:
    insertion_rmsd = float(np.mean(all_rmsds))
  else:
    insertion_rmsd = float('nan')

  # Mark identified motif atoms with B-factor = 1.
  b_factors = np.zeros(sampled_struct.num_atoms, dtype=np.float32)
  for _, (
      res_id,
      chain_id,
      matched_indices,
  ) in motif_to_residue_matches.items():
    residue_mask = (sampled_struct.res_id == res_id) & (
        sampled_struct.chain_id == chain_id
    )
    for atom37_idx in matched_indices:
      atom_name = ATOM37_ATOM_NAMES[atom37_idx]
      atom_mask = residue_mask & (sampled_struct.atom_name == atom_name)
      b_factors[atom_mask] = 1.0

  # Mark ligands with B-factor = 1.
  ligand_mask = sampled_struct.chain_type == 'non-polymer'
  b_factors[ligand_mask] = 1.0

  return UnindexedMatchingResult(
      diffused_index_map=diffused_index_map,
      insertion_rmsd=insertion_rmsd,
      insertion_rmsd_by_residue=insertion_rmsd_by_residue,
      matched_residue_ids=matched_residue_ids,
      motif_atom_mask=b_factors,
  )


def create_aligned_reference_structure(
    sampled_struct: structure.Structure,
    motif_data: UnindexedMotifData,
    diffused_index_map: dict[int, str],
) -> structure.Structure:
  """Creates a reference structure aligned to sampled_struct atom ordering.

  The output structure has the same atoms as sampled_struct, with:
  - Motif atoms: coordinates from motif_data, b_factor=1.0
  - Non-motif atoms: zero coordinates, b_factor=0.0
  - Ligand atoms: coordinates from sampled_struct (unchanged), b_factor=1.0

  Args:
    sampled_struct: The sampled structure to align to.
    motif_data: Unindexed motif data containing original motif coordinates.
    diffused_index_map: Mapping from motif residue index to "chain_id+res_id"
      string (e.g., {0: "A119", 1: "A120"}).

  Returns:
    Structure with same atom ordering as sampled_struct, with motif positions
    and b-factors set appropriately.
  """
  new_coords = np.zeros_like(sampled_struct.coords)
  new_b_factors = np.zeros(sampled_struct.num_atoms, dtype=np.float32)

  aatype_mask = motif_data.unindexed_motif_aatype_mask.astype(bool).copy()

  # Motif reseq residues will have a nonzero motif_atom_mask (from the
  # backbone atoms), but will have zero aatype and zero motif_aatype_mask.
  motif_reseq_res_mask = ~aatype_mask & (
      motif_data.unindexed_motif_atom_mask.sum(axis=1) > 0
  )
  aatype_mask |= motif_reseq_res_mask

  ligand_indices = np.where(motif_data.unindexed_motif_aatype > 19)[0]
  if ligand_indices.size > 0:
    first_ligand = ligand_indices[0]
    aatype_mask[first_ligand:] = False

  num_motif_residues = int(aatype_mask.sum())
  valid_to_original_idx = np.where(aatype_mask)[0]

  for motif_res_idx in range(num_motif_residues):
    original_idx = valid_to_original_idx[motif_res_idx]
    motif_atom_mask = motif_data.unindexed_motif_atom_mask[original_idx]
    motif_atom_positions = motif_data.unindexed_motif_atom_positions[
        original_idx
    ]

    if motif_res_idx in diffused_index_map:
      target_key = diffused_index_map[motif_res_idx]
      target_chain = target_key[0]
      target_res_id = int(target_key[1:])

      for atom37_idx in np.where(motif_atom_mask > 0)[0]:
        atom_name = ATOM37_ATOM_NAMES[atom37_idx]
        atom_coords = motif_atom_positions[atom37_idx]

        atom_mask = (
            (sampled_struct.chain_id == target_chain)
            & (sampled_struct.res_id == target_res_id)
            & (sampled_struct.atom_name == atom_name)
        )

        if np.any(atom_mask):
          new_coords[atom_mask] = atom_coords
          new_b_factors[atom_mask] = 1.0

  ligand_mask = sampled_struct.chain_type == 'non-polymer'
  new_coords[ligand_mask] = sampled_struct.coords[ligand_mask]
  new_b_factors[ligand_mask] = 1.0

  return sampled_struct.copy_and_update_atoms(
      atom_x=new_coords[:, 0],
      atom_y=new_coords[:, 1],
      atom_z=new_coords[:, 2],
      atom_b_factor=new_b_factors,
  )


def validate_matched_motif_residues_are_on_design_chains(
    protein_motif_residue_map: Mapping[tuple[str, int], tuple[str, int]],
    motif_str: str,
) -> None:
  """Validates that matched unindexed motif residues are on design chains.

  Args:
    protein_motif_residue_map: Mapping from reference to designed residues.
    motif_str: Motif specification string.

  Raises:
    ValueError: If a matched residue is not on a protein design chain.
  """
  if not protein_motif_residue_map:
    return

  designable_chain_ids = {
      chr(ord('A') + i)
      for i, chain_str in enumerate(motif_str.split('/'))
      if any(seg and seg[0].isdigit() for seg in chain_str.split(','))
  }
  for ref_res, designed_res in protein_motif_residue_map.items():
    if designed_res[0] not in designable_chain_ids:
      raise ValueError(
          f'Matched motif residue {ref_res} was mapped to '
          f'{designed_res} which is not on a protein design chain. '
          f'Valid protein design chains: {designable_chain_ids}'
      )


def get_ligand_residue_map(
    motif_struct: structure.Structure,
    parsed_spec: motif_spec.MotifSpec,
) -> dict[tuple[str, int], tuple[str, int]]:
  """Returns map from original motif ligand residue IDs to target residue IDs."""
  motif_ligands = motif_struct.filter_to_entity_type(ligand=True)
  if motif_ligands.num_atoms == 0:
    return {}

  motif_ligand_chain_and_res_ids = set(
      (str(c), int(r))
      for c, r in zip(motif_ligands.chain_id, motif_ligands.res_id)
  )
  if not motif_ligand_chain_and_res_ids:
    return {}

  if (residue_map := parsed_spec.residue_map) is None:
    raise ValueError(
        'ResidueMap must be provided in parsed_spec for ligand matching.'
    )
  source2target = residue_map.get_source_resid_to_target_resid_dict()

  missing_ligands = [
      (chain_id, res_id)
      for chain_id, res_id in motif_ligand_chain_and_res_ids
      if (chain_id, res_id) not in source2target
  ]
  if missing_ligands:
    raise ValueError(
        f"Motif ligands {missing_ligands} not found in parsed_spec's "
        f'residue_map. Full residue_map: {source2target}'
    )

  return {
      (chain_id, res_id): source2target[(chain_id, res_id)]
      for chain_id, res_id in motif_ligand_chain_and_res_ids
  }


def prepare_unindexed_motif_sample(
    *,
    sampled_struct: structure.Structure,
    spec: motif_spec.MotifSpec,
    unindexed_motif_data: (
        UnindexedMotifData | types_lib.DiffusionInput | None
    ) = None,
    motif_struct: structure.Structure | None = None,
    predicted_motif_indices_mask_logits: np.ndarray | None = None,
) -> UnindexedMotifPreparationResult:
  """Prepares an unindexed motif sample for evaluation and sequence redesign.

  Matches unindexed motif residues to the sampled structure using geometric
  Hungarian assignment, marks motif and ligand atoms in the structure's
  B-factors,
  builds fixed sequence and atom masks, reconstructs the motif specification
  string
  and residue map, and generates an aligned reference structure for structural
  metrics.

  If the provided spec is not unindexed (i.e. indexed), returns the structure
  and
  masks as-is with default zero metrics.

  Args:
    sampled_struct: Sampled structure output from reverse diffusion.
    spec: Parsed or unparsed MotifSpec.
    unindexed_motif_data: UnindexedMotifData container or DiffusionInput
      containing unindexed motif coordinates and masks.
    motif_struct: Optional reference motif structure. If omitted, defaults to
      spec.input_struct.
    predicted_motif_indices_mask_logits: Optional model logits predicting which
      residues are motif residues.

  Returns:
    An UnindexedMotifPreparationResult containing the updated structures, masks,
    motif string, residue map, and diagnostic metrics.
  """
  if not spec.is_unindexed:
    if isinstance(unindexed_motif_data, types_lib.DiffusionInput):
      fixed_seq_mask = np.asarray(unindexed_motif_data.fixed_seq_mask)
      fixed_atom_mask = np.asarray(unindexed_motif_data.fixed_atom_mask)
      if (
          motif_struct is None
          and unindexed_motif_data.crop_cond_atom_positions is not None
          and unindexed_motif_data.crop_cond_aatype is not None
          and np.any(fixed_atom_mask)
      ):
        motif_struct = motif_spec.get_motif_gtstruct(unindexed_motif_data)
    else:
      fixed_seq_mask = np.zeros(
          sampled_struct.num_residues(count_unresolved=False), dtype=np.int32
      )
      fixed_atom_mask = np.zeros(sampled_struct.num_atoms, dtype=np.int32)
    ref_struct = motif_struct or spec.input_struct or sampled_struct
    return UnindexedMotifPreparationResult(
        sampled_struct=sampled_struct,
        reference_motif_struct=ref_struct,
        sampled_motif_str=spec.sampled_motif_str or spec.motif_str,
        residue_map=spec.residue_map,
        fixed_seq_mask=fixed_seq_mask,
        fixed_atom_mask=fixed_atom_mask,
        metrics={
            'unindexed_motif_insertion_rmsd': 0.0,
            'unindexed_motif_insertion_rmsd_by_residue': '{}',
            'unindexed_motif_num_unmatched_residues': 0,
            'unindexed_motif_predicted_residues': '',
            'unindexed_motif_matched_residues': '',
            'unindexed_motif_matched_not_predicted': '',
            'unindexed_geom_missing_res': 0,
            'unindexed_pred_missing_res': 0,
        },
    )

  if unindexed_motif_data is None:
    raise ValueError(
        'unindexed_motif_data is None. This must be present if using '
        'unindexed motif conditioning with geometric matching.'
    )
  if isinstance(unindexed_motif_data, types_lib.DiffusionInput):
    motif_data = UnindexedMotifData.from_diffusion_input(unindexed_motif_data)
  else:
    motif_data = unindexed_motif_data

  if motif_struct is None:
    if spec.input_struct is None:
      spec = spec.with_input_struct()
    motif_struct = spec.input_struct
  if motif_struct is None:
    raise ValueError(
        'motif_struct could not be resolved from input or spec.input_struct.'
    )

  unindexed_motif_residues = str(spec.unindexed_motif_residues or '')

  # Geometric matching using pre-aligned motif coordinates.
  matching_result = process_unindexed_outputs(
      sampled_struct=sampled_struct,
      unindexed_motif_atom_positions=motif_data.unindexed_motif_atom_positions,
      unindexed_motif_aatype=motif_data.unindexed_motif_aatype,
      unindexed_motif_atom_mask=motif_data.unindexed_motif_atom_mask,
      unindexed_motif_aatype_mask=motif_data.unindexed_motif_aatype_mask,
  )

  # Get the motif_atom_mask from geometric matching.
  motif_atom_mask = (matching_result.motif_atom_mask > 0).astype(np.int32)
  ligand_mask = (sampled_struct.chain_type == 'non-polymer').astype(np.int32)
  fixed_atom_mask = motif_atom_mask | ligand_mask

  # Add ligand atoms to the sequence mask.
  ligand_struct = sampled_struct.filter_to_entity_type(ligand=True)
  num_ligand_atoms = len(ligand_struct.atom_b_factor)
  ligand_seq_mask = np.ones(num_ligand_atoms, dtype=np.int32)

  # Output structure with motif b-factors.
  sampled_struct_with_motif_atoms = sampled_struct.copy_and_update_atoms(
      atom_b_factor=fixed_atom_mask.astype(float)
  )

  has_matched_motif_residues = bool(matching_result.matched_residue_ids)
  if has_matched_motif_residues:
    protein_motif_residue_map = matching_result.get_motif_residue_map(
        unindexed_motif_residues
    )
  else:
    protein_motif_residue_map = {}

  validate_matched_motif_residues_are_on_design_chains(
      protein_motif_residue_map=protein_motif_residue_map,
      motif_str=spec.motif_str,
  )

  chain_a_struct = sampled_struct.filter(chain_id='A')
  chain_a_len = chain_a_struct.num_residues(count_unresolved=False)

  matched_protein_motif_str = residue_mapping.source_target_dict_to_motif_str(
      source_resid_to_target_resid=protein_motif_residue_map,
      designable_chain_length=chain_a_len,
  )
  fixed_chains_motif_str = residue_mapping.get_fixed_chains(
      spec.sampled_motif_str or spec.motif_str or ''
  )
  if fixed_chains_motif_str:
    sampled_motif_str = matched_protein_motif_str + '/' + fixed_chains_motif_str
  else:
    sampled_motif_str = matched_protein_motif_str

  insertion_rmsd_by_residue_json = json.dumps(
      matching_result.insertion_rmsd_by_residue
  )

  atom_mask = motif_data.unindexed_motif_atom_mask
  ligand_indices = np.where(motif_data.unindexed_motif_aatype > 19)[0]
  if ligand_indices.size > 0:
    first_ligand_index = ligand_indices[0]
    atom_mask = atom_mask[:first_ligand_index]
  num_motif_residues = int((atom_mask.sum(axis=1) > 0).sum())

  num_unmatched_residues = num_motif_residues - len(
      matching_result.diffused_index_map
  )

  # Make sure fixed_seq_mask accounts for reseq residues.
  residue_map = residue_mapping.motif_str_to_residue_map(
      motif_str=sampled_motif_str,
      input_struct=motif_struct,
      reseq_residues=spec.reseq_residues,
  )
  protein_seq_mask = residue_map.fixed_seq_mask[:chain_a_len].astype(np.int32)
  fixed_seq_mask = np.concatenate([protein_seq_mask, ligand_seq_mask])

  reference_motif_struct = create_aligned_reference_structure(
      sampled_struct=sampled_struct,
      motif_data=motif_data,
      diffused_index_map=matching_result.diffused_index_map,
  )

  predicted_residues = []
  if predicted_motif_indices_mask_logits is not None:
    predicted_mask = np.argmax(predicted_motif_indices_mask_logits, axis=-1)
    predicted_indices = np.nonzero(predicted_mask)[0]
    protein_chain_a_struct = sampled_struct.filter(chain_id='A')
    protein_res_ids_chain_a = protein_chain_a_struct.res_id
    unique_res_ids_chain_a, unique_indices_chain_a = np.unique(
        protein_res_ids_chain_a, return_index=True
    )
    unique_res_ids_chain_a = unique_res_ids_chain_a[
        np.argsort(unique_indices_chain_a)
    ]
    for pred_idx in predicted_indices:
      if pred_idx < len(unique_res_ids_chain_a):
        predicted_residues.append(int(unique_res_ids_chain_a[pred_idx]))

  matched_residues = [
      int(res_id) for res_id in matching_result.matched_residue_ids
  ]
  matched_not_predicted = sorted(
      list(set(matched_residues) - set(predicted_residues))
  )

  metrics = {
      'unindexed_motif_insertion_rmsd': matching_result.insertion_rmsd,
      'unindexed_motif_insertion_rmsd_by_residue': (
          insertion_rmsd_by_residue_json
      ),
      'unindexed_motif_num_unmatched_residues': num_unmatched_residues,
      'unindexed_motif_predicted_residues': ','.join(
          map(str, predicted_residues)
      ),
      'unindexed_motif_matched_residues': ','.join(map(str, matched_residues)),
      'unindexed_motif_matched_not_predicted': ','.join(
          map(str, matched_not_predicted)
      ),
      'unindexed_geom_missing_res': num_motif_residues - len(matched_residues),
      'unindexed_pred_missing_res': (
          num_motif_residues - len(predicted_residues)
      ),
  }

  return UnindexedMotifPreparationResult(
      sampled_struct=sampled_struct_with_motif_atoms,
      reference_motif_struct=reference_motif_struct,
      sampled_motif_str=sampled_motif_str,
      residue_map=residue_map,
      fixed_seq_mask=fixed_seq_mask,
      fixed_atom_mask=fixed_atom_mask,
      metrics=metrics,
  )
