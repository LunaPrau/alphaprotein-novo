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

"""Kemp eliminase hydrogen bonding and active-site geometry metrics."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Final

from alphafold3 import structure
from alphafold3.cpp import string_array
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.metrics import catalytic_motifs
import numpy as np
import scipy.spatial.distance

BondTuple = tuple[tuple[str, int, str], tuple[str, int, str]]

# Roles for each type of motif residue in a Kemp eliminase motif. Each
# residue is mapped to potential atoms that can satisfy this role.
KE_MOTIF_ROLES: Final[Mapping[str, Mapping[str, Sequence[str]]]] = {
    'base': {'D': ('OD1', 'OD2'), 'E': ('OE1', 'OE2'), 'H': ('NE2', 'ND1')},
    'oah': {
        'N': ('ND2',),
        'Q': ('NE2',),
        'S': ('OG',),
        'T': ('OG1',),
        'Y': ('OH',),
        'H': ('NE2', 'ND1'),
        'K': ('NZ',),
        'R': ('NE', 'NH1', 'NH2'),
    },
    'polarizer': {'D': ('OD1', 'OD2'), 'E': ('OE1', 'OE2')},
    'aromatic_res': {
        'F': ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
        'W': ('CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'),
        'Y': ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
    },
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class KempEliminaseLigandMetadata:
  """Metadata for a Kemp eliminase ligand.

  Attributes:
    acid_atom: The name of the atom in the ligand that interacts with the
      catalytic base.
    oxyanion_atom: The name of the atom in the ligand that interacts with the
      oxyanion hole.
  """

  acid_atom: str
  oxyanion_atom: str


KEMP_ELIMINASE_LIGAND_METADATA: Final[
    Mapping[str, KempEliminaseLigandMetadata]
] = {
    'H5J': KempEliminaseLigandMetadata(
        acid_atom='CAE',
        oxyanion_atom='OAH',
    ),
    '6NT': KempEliminaseLigandMetadata(
        acid_atom='N3',
        oxyanion_atom='N1',
    ),
}


def get_valid_role_combinations(
    struct: structure.Structure,
    motif_res_indices: Sequence[int],
) -> list[dict[str, int]]:
  """Determines all valid mappings of residues to catalytic roles.

  Maps the given motif residue indices to potential role combinations,
  verifying that the amino acid type at each index matches the expected type for
  the role. Since multiple residues might have the same type and role assignment
  is ambiguous (e.g., 3 residues could be Base-OAH-Polarizer or
  Base-OAH-Aromatic), multiple valid combinations may exist.

  Args:
    struct: The protein structure.
    motif_res_indices: List of residue indices involved in the motif.

  Returns:
    A list of dictionaries. Each dictionary represents a valid assignment
    of roles to residue indices (e.g., {'base': 10, 'oah': 25}).

  Raises:
    ValueError: If the number of indices is not 2, 3, or 4.
  """
  num_residues = len(motif_res_indices)
  candidate_role_sets = []

  # Always require Base and OAH.
  # Determine possible additional roles based on residue count.
  if num_residues == 2:
    candidate_role_sets.append(('base', 'oah'))
  elif num_residues == 3:
    candidate_role_sets.append(('base', 'oah', 'polarizer'))
    candidate_role_sets.append(('base', 'oah', 'aromatic_res'))
  elif num_residues == 4:
    candidate_role_sets.append(('base', 'polarizer', 'oah', 'aromatic_res'))
  else:
    raise ValueError(
        f'Unsupported number of motif residues: {num_residues}. Expected 2,'
        ' 3, or 4.'
    )

  valid_combinations = []
  for roles in candidate_role_sets:
    role_defs = {r: tuple(KE_MOTIF_ROLES[r].keys()) for r in roles}
    valid_combinations.extend(
        catalytic_motifs.generate_valid_role_combinations(
            struct, motif_res_indices, role_defs
        )
    )

  return valid_combinations


def _get_best_hbond_interaction(
    struct: structure.Structure,
    res_chain_id: str,
    res_id: int,
    candidate_atom_names: Sequence[str],
    target_atom_tuple: tuple[str, int, str],
) -> tuple[dict[str, Any], str]:
  """Computes H-bond metrics for all candidate atoms and returns the best one.

  Args:
    struct: The structure to compute H-bond metrics for.
    res_chain_id: Chain ID of the residue.
    res_id: Residue ID.
    candidate_atom_names: List of atom names to consider for the interaction.
    target_atom_tuple: Tuple (chain_id, res_id, atom_name) of the target atom.

  Returns:
    A tuple containing:
      - The H-bond metrics dictionary for the best interaction (shortest dist).
      - The name of the atom used in the best interaction.
  """
  pairs = [
      ((res_chain_id, res_id, atom_name), target_atom_tuple)
      for atom_name in candidate_atom_names
  ]
  best_result, (best_atom_tuple, _) = (
      catalytic_motifs.select_best_hbond_candidate(
          struct, pairs, prefer_hbond=False
      )
  )
  return best_result, best_atom_tuple[2]


def compute_hbond_metrics_for_role_combination(
    struct: structure.Structure,
    role_mapping: Mapping[str, int],
    ligand_acid_atom_tuple: tuple[str, int, str],
    ligand_oxyanion_atom_tuple: tuple[str, int, str],
) -> tuple[dict[str, Any], list[BondTuple]]:
  """Computes H-bond metrics for a specific role mapping.

  Args:
    struct: The protein structure.
    role_mapping: Mapping for roles ('base', 'polarizer', 'oah') to residue
      indices.
    ligand_acid_atom_tuple: Target tuple for acid-base interaction.
    ligand_oxyanion_atom_tuple: Target tuple for oxyanion interaction.

  Returns:
    Dictionary of metrics. Keys are prefixed with 'acid_base/', 'oxyanion_oah/',
    or 'base_dyad/'.
  """
  protein_struct = struct.filter_to_entity_type(protein=True)
  protein_chain_id = protein_struct.chain_id[0]

  # Map res_id to AA type 1-letter for the relevant residues.
  relevant_indices = list(role_mapping.values())
  relevant_struct = protein_struct.filter(res_id=relevant_indices)
  res_type_map = {}
  for resi, res_name in zip(relevant_struct.res_id, relevant_struct.res_name):
    aa_1 = residue_names.CCD_NAME_TO_ONE_LETTER.get(res_name, 'X')
    res_type_map[resi] = aa_1

  # Base interaction.
  base_resi = role_mapping['base']
  base_aatype = res_type_map[base_resi]
  base_atom_names = KE_MOTIF_ROLES['base'][base_aatype]

  acid_base_result, used_base_atom = _get_best_hbond_interaction(
      struct,
      protein_chain_id,
      base_resi,
      base_atom_names,
      ligand_acid_atom_tuple,
  )
  acid_base_bond: BondTuple = (
      (protein_chain_id, base_resi, used_base_atom),
      ligand_acid_atom_tuple,
  )

  # OAH interaction.
  oah_resi = role_mapping['oah']
  oah_aatype = res_type_map[oah_resi]
  oah_atom_names = KE_MOTIF_ROLES['oah'][oah_aatype]

  oxyanion_oah_result, used_oah_atom = _get_best_hbond_interaction(
      struct,
      protein_chain_id,
      oah_resi,
      oah_atom_names,
      ligand_oxyanion_atom_tuple,
  )
  oxyanion_oah_bond: BondTuple = (
      (protein_chain_id, oah_resi, used_oah_atom),
      ligand_oxyanion_atom_tuple,
  )

  # Base dyad interaction.
  base_dyad_result = {}
  base_dyad_bond: BondTuple | None = None
  if 'polarizer' in role_mapping and role_mapping['polarizer'] is not None:
    dyad_resi = role_mapping['polarizer']
    dyad_aatype = res_type_map[dyad_resi]
    dyad_atom_names = KE_MOTIF_ROLES['polarizer'][dyad_aatype]

    # Identify the target atom on the base.
    remaining_base_atoms = [a for a in base_atom_names if a != used_base_atom]
    if not remaining_base_atoms:
      raise ValueError(
          'No remaining base atoms for dyad interaction. '
          f'Base atoms: {base_atom_names}, Used: {used_base_atom}'
      )

    base_dyad_target_atom = remaining_base_atoms[0]
    base_dyad_target_tuple = (
        protein_chain_id,
        base_resi,
        base_dyad_target_atom,
    )

    base_dyad_result, used_dyad_atom = _get_best_hbond_interaction(
        struct,
        protein_chain_id,
        dyad_resi,
        dyad_atom_names,
        base_dyad_target_tuple,
    )
    base_dyad_bond = (
        (protein_chain_id, dyad_resi, used_dyad_atom),
        base_dyad_target_tuple,
    )

  acid_base_result = {f'acid_base/{k}': v for k, v in acid_base_result.items()}
  oxyanion_oah_result = {
      f'oxyanion_oah/{k}': v for k, v in oxyanion_oah_result.items()
  }
  base_dyad_result = {f'base_dyad/{k}': v for k, v in base_dyad_result.items()}

  hbond_bonds: list[BondTuple] = [acid_base_bond, oxyanion_oah_bond]
  if base_dyad_bond is not None:
    hbond_bonds.append(base_dyad_bond)

  return (
      acid_base_result | oxyanion_oah_result | base_dyad_result,
      hbond_bonds,
  )


def compute_hbond_metrics(
    struct: structure.Structure,
    ligand_acid_atom_tuple: tuple[str, int, str],
    ligand_oxyanion_atom_tuple: tuple[str, int, str],
    role_mappings: Sequence[dict[str, int]],
) -> tuple[dict[str, Any], Mapping[str, int], list[BondTuple]]:
  """Finds the role combination with the smallest mean of H-bond distances.

  Computes H-bond metrics for each provided role mapping. The "best" mapping
  is defined as the one minimizing the mean of the distances for the expected
  interactions:
    - Acid-Base
    - Oxyanion Hole (OAH)
    - Base Dyad (if a polarizer is present)

  Args:
    struct: The protein structure.
    ligand_acid_atom_tuple: Target tuple for acid-base interaction.
    ligand_oxyanion_atom_tuple: Target tuple for oxyanion interaction.
    role_mappings: List of possible role assignments.

  Returns:
    A tuple containing metrics for the best role mapping and the mapping itself.

  Raises:
    ValueError: If role_mappings is empty.
  """
  if not role_mappings:
    raise ValueError('No role mappings provided.')

  best_metrics = {}
  best_mapping: Mapping[str, int] = {}
  best_bonds: list[BondTuple] = []
  min_mean_dist = float('inf')

  for mapping in role_mappings:
    metrics, bonds = compute_hbond_metrics_for_role_combination(
        struct,
        mapping,
        ligand_acid_atom_tuple,
        ligand_oxyanion_atom_tuple,
    )

    current_dist_sum = 0.0
    num_interactions = 0

    # Acid-Base.
    current_dist_sum += metrics['acid_base/distance']
    num_interactions += 1

    # Oxyanion-OAH.
    current_dist_sum += metrics['oxyanion_oah/distance']
    num_interactions += 1

    # Base dyad (only if polarizer exists).
    if mapping.get('polarizer') is not None:
      current_dist_sum += metrics['base_dyad/distance']
      num_interactions += 1

    current_mean_dist = current_dist_sum / num_interactions

    if current_mean_dist < min_mean_dist:
      min_mean_dist = current_mean_dist
      best_metrics = metrics
      best_mapping = mapping
      best_bonds = bonds

  return best_metrics, best_mapping, best_bonds


def kemp_hbond_metrics(
    struct: structure.Structure,
    motif_residues: Sequence[int] | str | None = None,
    ligand_res_name: str | None = None,
) -> dict[str, float]:
  """Computes hydrogen bonding metrics for Kemp eliminase designs.

  Args:
    struct: The complex structure containing protein and bound Kemp ligand.
    motif_residues: Optional motif specification. Can be a sequence of 1-based
      residue IDs, a motif string (e.g., '25,A1,60,A2'), or None to auto-detect
      fixed residues with `atom_b_factor > 0.0`.
    ligand_res_name: Optional ligand CCD code (e.g., '6NT' or 'H5J'). If None,
      auto-detected from the structure.

  Returns:
    Dictionary of calculated H-bonding metrics (e.g. 'acid_base/distance',
    'oxyanion_oah/distance', 'acid_base/dihedral2', etc.).

  Raises:
    ValueError: If ligand is missing or not a known Kemp eliminase ligand, or
      if fewer than 2 catalytic residues are found.
  """
  ligand_struct = struct.filter_to_entity_type(ligand=True)
  if ligand_struct.num_residues(count_unresolved=False) == 0:
    raise ValueError('No ligand found in structure for Kemp H-bond metrics.')

  if ligand_res_name is None:
    ligand_names = list(set(ligand_struct.res_name))
    for candidate in KEMP_ELIMINASE_LIGAND_METADATA:
      if candidate in ligand_names:
        ligand_res_name = candidate
        break
    if ligand_res_name is None:
      ligand_res_name = ligand_struct.res_name[0]

  if ligand_res_name not in KEMP_ELIMINASE_LIGAND_METADATA:
    raise ValueError(
        f'Unsupported Kemp eliminase ligand: {ligand_res_name}. Expected one'
        f' of {list(KEMP_ELIMINASE_LIGAND_METADATA.keys())}.'
    )

  ligand_meta = KEMP_ELIMINASE_LIGAND_METADATA[ligand_res_name]
  lig_filtered = ligand_struct.filter(res_name=[ligand_res_name])
  lig_chain_id = lig_filtered.chain_id[0]
  lig_res_id = int(lig_filtered.res_id[0])

  ligand_acid_tuple = (lig_chain_id, lig_res_id, ligand_meta.acid_atom)
  ligand_oxyanion_tuple = (
      lig_chain_id,
      lig_res_id,
      ligand_meta.oxyanion_atom,
  )

  if motif_residues is None:
    # Auto-detect fixed motif residues from b-factor > 0.
    prot = struct.filter_to_entity_type(protein=True)
    fixed_prot = prot.filter(atom_b_factor=prot.atom_b_factor > 0.0)
    cat_res_indices = sorted(list(set(fixed_prot.res_id)))
  elif isinstance(motif_residues, str):
    cat_res_indices = catalytic_motifs.get_motif_residue_indices(motif_residues)
  else:
    cat_res_indices = list(motif_residues)

  if len(cat_res_indices) < 2:
    raise ValueError(
        f'At least 2 catalytic residues required, found {len(cat_res_indices)}.'
    )

  role_mappings = get_valid_role_combinations(struct, cat_res_indices)
  if not role_mappings:
    raise ValueError(
        'No valid catalytic role combinations found for residues'
        f' {cat_res_indices}.'
    )

  metrics, _, _ = compute_hbond_metrics(
      struct,
      ligand_acid_tuple,
      ligand_oxyanion_tuple,
      role_mappings,
  )
  return metrics


def cat_base_tip_ca_neighbors_mean(
    struct: structure.Structure,
    cat_ligand_atom_names: set[str] | Sequence[str] | None = None,
    ca_dist_cutoffs: Sequence[int | float] = (6, 8, 12),
    motif_residues: Sequence[int] | str | None = None,
) -> float:
  """Number of CA neighbors for the closest motif residue tip atom to the TS.

  Args:
    struct: Structure to analyze containing protein and ligand.
    cat_ligand_atom_names: Names of reactive catalytic ligand atoms (e.g. {'N3',
      'CAE'}). If None, automatically inferred from known Kemp eliminase ligands
      (e.g. 'N3' for 6NT, 'CAE' for H5J).
    ca_dist_cutoffs: Distance cutoffs in Angstroms for CA neighbor counting
      (default: (6, 8, 12)).
    motif_residues: Optional motif residue indices (1-indexed) or motif spec
      string (e.g. '82,A1,38,A2'). If None, inferred from atom_b_factor > 0.

  Returns:
    Mean number of protein CA neighbors around the catalytic base tip atom
    across cutoffs.

  Raises:
    ValueError: If protein or ligand is missing, no motif residues are found, or
      no catalytic ligand atoms are found in the ligand.
  """
  prot = struct.filter_to_entity_type(protein=True)
  if prot.num_residues(count_unresolved=False) == 0:
    raise ValueError('No protein found in structure.')

  lig = struct.filter_to_entity_type(ligand=True)
  if lig.num_residues(count_unresolved=False) == 0:
    raise ValueError('No ligand found in structure.')

  # Determine motif residues
  if motif_residues is None:
    fixed_mask = prot.atom_b_factor > 0.0
    if np.any(fixed_mask):
      motif = prot.filter(atom_b_factor=fixed_mask)
    else:
      motif = prot
  elif isinstance(motif_residues, str):
    cat_res_indices = catalytic_motifs.get_motif_residue_indices(motif_residues)
    motif = prot.filter(res_id=cat_res_indices)
  else:
    motif = prot.filter(res_id=list(motif_residues))

  if motif.num_residues(count_unresolved=False) == 0:
    raise ValueError('No motif residues found in protein structure.')

  # Determine catalytic ligand atom names
  if cat_ligand_atom_names is None:
    cat_atoms = set()
    for lig_name in set(lig.res_name):
      if lig_name in KEMP_ELIMINASE_LIGAND_METADATA:
        cat_atoms.add(KEMP_ELIMINASE_LIGAND_METADATA[lig_name].acid_atom)
    if not cat_atoms:
      cat_atoms = {'N3', 'CAE'}
  else:
    cat_atoms = set(cat_ligand_atom_names)

  cat_lig_mask = string_array.isin(lig.atom_name, cat_atoms)
  if not np.any(cat_lig_mask):
    raise ValueError(
        f'None of catalytic ligand atoms {cat_atoms} found in ligand (found'
        f' {set(lig.atom_name)}).'
    )

  dist = scipy.spatial.distance.cdist(
      motif.coords,
      lig.coords[cat_lig_mask],
  ).min(axis=1)

  idx_tip_atom = int(np.argmin(dist))
  tip_coord = motif.coords[[idx_tip_atom]]

  ca_mask = prot.atom_name == 'CA'
  if not np.any(ca_mask):
    raise ValueError('No CA atoms found in protein.')
  ca_coords = prot.coords[ca_mask]

  d = scipy.spatial.distance.cdist(tip_coord, ca_coords)
  cutoffs = np.array(ca_dist_cutoffs)[:, None, None]
  result = d[None, ...] < cutoffs
  ca_neighbors = result.sum(axis=-1).mean(axis=1)
  return float(ca_neighbors.mean())


def kemp_structure_metrics(
    struct: structure.Structure,
    *,
    motif_residues: Sequence[int] | str | None = None,
    ligand_res_name: str | None = None,
    ca_dist_cutoffs: Sequence[int | float] = (6, 8, 12),
) -> dict[str, Any]:
  """Unified driver computing Kemp eliminase H-bonding and pocket neighbor metrics.

  Args:
    struct: Structure containing protein and Kemp eliminase ligand.
    motif_residues: Optional motif residue indices (1-indexed) or motif spec
      string (e.g. '82,A1,38,A2'). If None, inferred from atom_b_factor > 0.
    ligand_res_name: Optional ligand 3-letter code (e.g. '6NT', 'H5J').
    ca_dist_cutoffs: Distance cutoffs in Angstroms for CA neighbor counting.

  Returns:
    Dictionary containing all Kemp eliminase H-bonding metrics plus
    cat_base_tip_ca_neighbors_mean.
  """
  metrics = kemp_hbond_metrics(
      struct,
      motif_residues=motif_residues,
      ligand_res_name=ligand_res_name,
  )
  metrics['cat_base_tip_ca_neighbors_mean'] = cat_base_tip_ca_neighbors_mean(
      struct,
      ca_dist_cutoffs=ca_dist_cutoffs,
      motif_residues=motif_residues,
  )
  return metrics
