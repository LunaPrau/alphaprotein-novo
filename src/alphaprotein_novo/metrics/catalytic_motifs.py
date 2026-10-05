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

"""Shared active-site catalytic motif utilities and role assignment."""

from collections.abc import Collection, Mapping, Sequence
import itertools
from typing import Any

from alphafold3 import structure
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.metrics import hbonds
import numpy as np

AtomTuple = tuple[str, int, str]


def get_motif_residue_indices(motif_str: str) -> list[int]:
  """Calculates the motif residue indices for the given motif string.

  The motif string format consists of segments separated by commas. Integers
  represent gaps (number of residues), and other strings (e.g., 'A1') represent
  motif residues. The function returns the 1-based indices of the motif
  residues. The string may contain a suffix starting with '/', which is ignored.

  Example: '25,A1,60,A2,72,A3,41/B1' -> [26, 87, 160]

  Args:
    motif_str: The sampled motif string.

  Returns:
    List of 1-based residue indices for the motif positions.
  """
  parts = motif_str.split('/')[0].split(',')
  indices = []
  current_pos = 0
  for part in parts:
    if part.isdigit():
      current_pos += int(part)
    else:
      current_pos += 1
      indices.append(current_pos)
  return indices


def generate_valid_role_combinations(
    struct: structure.Structure,
    candidate_residue_ids: Sequence[int],
    role_definitions: Mapping[str, Collection[str]],
) -> list[dict[str, int]]:
  """Generates all valid role-to-residue mappings matching amino acid types.

  Args:
    struct: Structure containing protein atoms.
    candidate_residue_ids: Sequence of candidate residue IDs (1-based).
    role_definitions: Mapping from role name (e.g. 'base') to a collection of
      allowed amino acid types (1-letter e.g. {'D', 'E'} or 3-letter e.g.
      {'ASP', 'GLU'}).

  Returns:
    List of dictionaries mapping role names to assigned residue IDs.
  """
  prot = struct.filter_to_entity_type(protein=True)
  motif_struct = prot.filter(res_id=list(candidate_residue_ids))
  res_type_map = {}
  for res_id, res_name in zip(motif_struct.res_id, motif_struct.res_name):
    one_letter = residue_names.CCD_NAME_TO_ONE_LETTER.get(res_name, 'X')
    res_type_map[res_id] = (one_letter, res_name.upper())

  role_names = list(role_definitions.keys())
  k = len(role_names)
  if k > len(candidate_residue_ids):
    return []

  valid_combinations = []
  for perm in itertools.permutations(candidate_residue_ids, k):
    mapping = dict(zip(role_names, perm))
    valid = True
    for role, r_id in mapping.items():
      allowed = set(role_definitions[role])
      actual_1, actual_3 = res_type_map.get(r_id, ('X', 'UNK'))
      if actual_1 not in allowed and actual_3 not in allowed:
        valid = False
        break
    if valid:
      valid_combinations.append(mapping)

  return valid_combinations


def select_best_hbond_candidate(
    struct: structure.Structure,
    candidates: Sequence[tuple[AtomTuple, AtomTuple]],
    thresholds: Mapping[str, tuple[float, float]] = (
        hbonds.LAUKO_HBOND_THRESHOLDS
    ),
    *,
    prefer_hbond: bool = True,
) -> tuple[dict[str, Any], tuple[AtomTuple, AtomTuple]]:
  """Evaluates H-bonding across candidate atom pairs and returns the best pair.

  Args:
    struct: Structure containing atoms.
    candidates: Sequence of `(atom1, atom2)` tuples to evaluate.
    thresholds: H-bond distance and angle thresholds.
    prefer_hbond: If True, immediately returns the first candidate that
      satisfies the H-bond threshold; otherwise selects strictly by minimum
      interatomic distance.

  Returns:
    Tuple `(best_info, best_pair)` from `determine_hbond_by_thresholding`.

  Raises:
    ValueError: If `candidates` is empty.
  """
  if not candidates:
    raise ValueError('No H-bond candidate pairs provided.')

  best_info: dict[str, Any] | None = None
  best_pair = candidates[0]
  for pair in candidates:
    atom1, atom2 = pair
    info = hbonds.determine_hbond_by_thresholding(
        struct,
        atom1=atom1,
        atom2=atom2,
        thresholds=thresholds,
    )
    if prefer_hbond and info.get('hbond'):
      return info, pair
    dist = info.get('distance', float('inf'))
    best_dist = (
        best_info.get('distance', float('inf'))
        if best_info is not None
        else float('inf')
    )
    if best_info is None or (
        not np.isnan(dist) and (np.isnan(best_dist) or dist < best_dist)
    ):
      best_info = info
      best_pair = pair

  assert best_info is not None
  return best_info, best_pair


def evaluate_hbond_pair(
    struct: structure.Structure,
    atom1: AtomTuple,
    atom2_candidates: Sequence[AtomTuple],
    thresholds: Mapping[str, tuple[float, float]] = (
        hbonds.LAUKO_HBOND_THRESHOLDS
    ),
) -> dict[str, Any]:
  """Evaluates H-bonding between atom1 and one or more candidate atom2 atoms."""
  if not atom2_candidates:
    raise ValueError('No atom2 candidates provided.')
  pairs = [(atom1, cand) for cand in atom2_candidates]
  best_info, _ = select_best_hbond_candidate(
      struct, pairs, thresholds=thresholds, prefer_hbond=True
  )
  return best_info
