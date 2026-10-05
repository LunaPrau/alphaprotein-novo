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

"""Multi-state consistency metrics across folded states and seeds."""

import collections
from collections.abc import Collection, Mapping, Sequence
import dataclasses
from typing import Any

from alphafold3 import structure
from alphafold3.cpp import string_array
from alphaprotein_novo.metrics import geometry
from alphaprotein_novo.metrics import structure_constants
import numpy as np


@dataclasses.dataclass(frozen=True)
class AtomSelection:
  """Specifies an atom selection within a structure.

  Attributes:
    atom_name: Name of the atom (e.g., 'FE', 'Fe1', 'N1', 'N4').
    chain_type: Optional chain type filter (e.g., ligand chain types).
    res_name: Optional residue name filter (e.g., 'HEM', 'pd00').
  """

  atom_name: str
  chain_type: str | Sequence[str] | None = None
  res_name: str | None = None

  def select_coords(self, struct: structure.Structure) -> np.ndarray:
    """Selects matching atom coordinates from the structure.

    Args:
      struct: The macromolecular structure to query.

    Returns:
      Coordinate array of shape (3,) if a single atom matches, or (K, 3) if
      multiple atoms match.

    Raises:
      ValueError: If no matching atoms are found.
    """
    filtered = struct.filter(atom_name=self.atom_name)
    if self.chain_type is not None:
      chain_types = (
          [self.chain_type]
          if isinstance(self.chain_type, str)
          else list(self.chain_type)
      )
      filtered = filtered.filter(chain_type=chain_types)
    if self.res_name is not None:
      filtered = filtered.filter(res_name=self.res_name)

    if filtered.num_atoms == 0:
      raise ValueError(
          f'No atoms found for selection atom_name={self.atom_name!r}, '
          f'chain_type={self.chain_type!r}, res_name={self.res_name!r}'
      )
    return filtered.coords[0] if filtered.num_atoms == 1 else filtered.coords


@dataclasses.dataclass(frozen=True)
class StateSpec:
  """Specification of a folded state and its key atom selections.

  Attributes:
    name: Unique identifier for the state (e.g., 'substrate', 'PIP_R').
    structure: The 3D Structure object for this state.
    atom_selections: Mapping from semantic atom roles (e.g., 'fe', 'n') to
      AtomSelection objects.
  """

  name: str
  structure: structure.Structure
  atom_selections: Mapping[str, AtomSelection]


def align_structure_to_reference(
    decoy: structure.Structure,
    reference: structure.Structure,
    backbone_atoms: Sequence[str] = structure_constants.PROTEIN_BACKBONE_ATOMS,
) -> structure.Structure:
  """Rigid-body aligns decoy structure to reference structure on protein backbone.

  Args:
    decoy: Structure to align.
    reference: Reference structure.
    backbone_atoms: Sequence of atom names to use for alignment (default: ('N',
      'CA', 'C')).

  Returns:
    New structure with coordinates transformed into reference frame.

  Raises:
    ValueError: If atom counts differ or no alignment atoms are found.
  """
  bb_atoms = set(backbone_atoms)
  decoy_mask = (~decoy.is_ligand_mask) & string_array.isin(
      decoy.atom_name, bb_atoms
  )
  ref_mask = (~reference.is_ligand_mask) & string_array.isin(
      reference.atom_name, bb_atoms
  )

  decoy_bb = decoy.coords[decoy_mask]
  ref_bb = reference.coords[ref_mask]

  if decoy_bb.shape[0] != ref_bb.shape[0]:
    raise ValueError(
        'Decoy and reference protein chains have different number of alignment '
        f'atoms: {decoy_bb.shape[0]} vs {ref_bb.shape[0]}.'
    )
  if decoy_bb.shape[0] == 0:
    raise ValueError(
        'No alignment atoms found matching the specified backbone atoms.'
    )

  r = geometry.transform_ls(
      decoy_bb - decoy_bb.mean(0), ref_bb - ref_bb.mean(0)
  )
  new_coords = np.matmul(decoy.coords - decoy_bb.mean(0), r.T) + ref_bb.mean(0)
  return decoy.copy_and_update_coords(coords=new_coords)


def compute_atom_self_rmsd(
    aligned_states: Sequence[tuple[structure.Structure, AtomSelection]],
) -> float:
  """Computes the centroid RMSD of selected atoms across aligned structures.

  Args:
    aligned_states: Sequence of (aligned_structure, atom_selection) pairs.

  Returns:
    RMSD value across the state coordinates, or np.nan if fewer than 2 states.
  """
  coords = [sel.select_coords(st) for st, sel in aligned_states]
  return geometry.centroid_rmsd(coords)


def compute_multi_state_consistency(
    states: Sequence[StateSpec],
    reference_structure: structure.Structure | None = None,
    align_to_first: bool = False,
    backbone_atoms: Sequence[str] = structure_constants.PROTEIN_BACKBONE_ATOMS,
) -> dict[str, float]:
  """Computes cross-state atom self-RMSD metrics across folded states.

  Each state is rigid-body aligned to the reference structure on its protein
  backbone. Then, for each atom role defined in the state specifications, the
  centroid RMSD of the aligned atom positions across all available states is
  computed.

  Args:
    states: Sequence of StateSpec objects for each state.
    reference_structure: Structure to align all states to.
    align_to_first: If True and reference_structure is None, use
      states[0].structure as reference.
    backbone_atoms: Backbone atom names to align on.

  Returns:
    Dictionary mapping '{role}_self_rmsd' to the computed centroid RMSD.

  Raises:
    ValueError: If reference_structure is None and align_to_first is False.
  """
  if not states:
    return {}

  if reference_structure is None:
    if align_to_first:
      reference_structure = states[0].structure
    else:
      raise ValueError(
          'A reference structure must be provided or align_to_first=True.'
      )

  aligned_states_by_name = {}
  for state in states:
    aligned = align_structure_to_reference(
        state.structure, reference_structure, backbone_atoms=backbone_atoms
    )
    aligned_states_by_name[state.name] = (aligned, state.atom_selections)

  atom_keys = set()
  for state in states:
    atom_keys.update(state.atom_selections.keys())

  results = {}
  for key in sorted(atom_keys):
    pairs = []
    for state in states:
      if key in state.atom_selections:
        aligned_st, sels = aligned_states_by_name[state.name]
        pairs.append((aligned_st, sels[key]))
    results[f'{key}_self_rmsd'] = compute_atom_self_rmsd(pairs)

  return results


def reduce_seed_metrics(
    metric_res: Mapping[str, Any],
    seed_suffixes: Collection[str],
) -> dict[str, Any]:
  """Reduce per-seed metrics to summary statistics.

  Returns a new dict containing all original entries plus computed summary
  statistics (mean, median, min, max, std for numeric values, majority for
  strings). Metrics that are not per-seed (i.e. not ending with /{seed}) are
  left unchanged.

  Args:
    metric_res: Metric dict mapping '{prefix}/{seed}' to values.
    seed_suffixes: The set of seed suffix strings (e.g. {'seed_0', 'seed_1'}).

  Returns:
    A new dict with original entries and added summary statistics.
  """
  result = dict(metric_res)
  if not seed_suffixes:
    return result

  ordered_seeds = sorted(seed_suffixes)
  metric_prefixes = sorted(
      {k.rsplit('/', 1)[0] for k in metric_res.keys() if '/' in k}
  )
  for metric_prefix in metric_prefixes:
    vals = [
        metric_res[f'{metric_prefix}/{seed}']
        for seed in ordered_seeds
        if f'{metric_prefix}/{seed}' in metric_res
    ]
    if not vals:
      continue
    # String-valued columns: take the most common non-empty value.
    if any(isinstance(v, str) for v in vals):
      non_empty = [v for v in vals if isinstance(v, str) and v]
      result[f'{metric_prefix}/majority'] = (
          collections.Counter(non_empty).most_common(1)[0][0]
          if non_empty
          else ''
      )
      continue

    # Numeric columns: filter out None and NaN before reducing.
    num_vals = [
        float(v)
        for v in vals
        if isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        and not np.isnan(float(v))
    ]
    for stat_name, stat_fn in [
        ('mean', np.mean),
        ('median', np.median),
        ('min', np.min),
        ('max', np.max),
        ('std', np.std),
    ]:
      result[f'{metric_prefix}/{stat_name}'] = (
          float(stat_fn(num_vals)) if num_vals else float(np.nan)
      )

  return result
