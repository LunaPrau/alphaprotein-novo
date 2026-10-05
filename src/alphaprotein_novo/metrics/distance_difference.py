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

"""LDDT and related distance matrix-based metrics.

Pure Python/NumPy reimplementation of the C++ lDDT computation.
"""

import dataclasses
import enum
from typing import Final

from absl import logging
from alphafold3 import structure
import numpy as np

CUTOFF_RADIUS: Final[float] = 15.0  # Angstroms.
_LDDT_TOLERANCES: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 4.0)


@enum.unique
class MultiChainMode(enum.Enum):
  """Modes of evaluating multiple chains."""

  ALL = 1
  BETWEEN_CHAINS_ONLY = 2
  WITHIN_CHAINS_ONLY = 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LDDTResult:
  lddt: float
  lddt_per_res: np.ndarray
  num_considered_pairs: int


def _group_by_residue_index(
    residue_indices: np.ndarray,
) -> list[tuple[int, int]]:
  """Group atom indices by residue index, returning (begin, end) ranges."""
  if len(residue_indices) == 0:
    return []
  ranges = []
  begin = 0
  for i in range(1, len(residue_indices)):
    if residue_indices[i] != residue_indices[i - 1]:
      if residue_indices[i] < residue_indices[i - 1]:
        raise ValueError('Residue indices must be monotonically increasing.')
      ranges.append((begin, i))
      begin = i
  ranges.append((begin, len(residue_indices)))
  return ranges


def _per_residue_sum(
    residue_ranges: list[tuple[int, int]], values: np.ndarray
) -> np.ndarray:
  """Sum values within each residue range."""
  result = np.zeros(len(residue_ranges), dtype=np.int64)
  for ri, (begin, end) in enumerate(residue_ranges):
    result[ri] = values[begin:end].sum()
  return result


def _lddt_from_coords(
    decoy_coords: np.ndarray,
    gt_coords: np.ndarray,
    *,
    residue_indices: np.ndarray,
    cutoff_radius: float,
    min_sequence_separation: int,
    tolerances: np.ndarray,
    gt_pairwise_mask: np.ndarray | None,
) -> tuple[float, np.ndarray, int]:
  """Compute the LDDT of two Nx3 np arrays of coordinates.

  Operates on the upper-triangular portion of the distance matrix (excluding
  diagonal) for efficiency.

  Args:
    decoy_coords: [N, 3] float32 array of decoy coordinates.
    gt_coords: [N, 3] float32 array of ground truth coordinates.
    residue_indices: [N] int64 array of monotonically increasing residue
      indices.
    cutoff_radius: Only pairs with gt distance below this are considered.
    min_sequence_separation: Minimum residue index separation to consider.
    tolerances: 1-D float32 array of distance tolerance thresholds.
    gt_pairwise_mask: Optional [N, N] bool mask. If provided, only pairs where
      the mask is True are considered.

  Returns:
    Tuple of (lddt_score, lddt_per_residue, num_considered_pairs).
  """
  if decoy_coords.shape != gt_coords.shape:
    raise ValueError(
        f'{decoy_coords.shape=} and {gt_coords.shape=} must match.'
    )

  num_atoms = decoy_coords.shape[0]

  if gt_pairwise_mask is not None and gt_pairwise_mask.size > 0:
    if np.all(~gt_pairwise_mask):
      residue_ranges = _group_by_residue_index(residue_indices)
      return 0.0, np.zeros(len(residue_ranges)), 0

  # Ensure types.
  residue_indices = residue_indices.astype(np.int64, copy=False)
  decoy_coords = decoy_coords.astype(np.float32, copy=False)
  gt_coords = gt_coords.astype(np.float32, copy=False)

  num_considered_pairs = 0
  per_atom_considered = np.zeros(num_atoms, dtype=np.int64)
  per_atom_in_cutoff = np.zeros(num_atoms, dtype=np.int64)

  # Iterate over upper-triangular pairs (i, j>i) as in the C++ implementation.
  for i in range(num_atoms):
    j_slice = slice(i + 1, num_atoms)

    # Sequence separation.
    res_idx_dist = np.abs(residue_indices[i] - residue_indices[j_slice])
    long_range = res_idx_dist >= min_sequence_separation

    # Ground truth and decoy distances.
    gt_dist = np.linalg.norm(gt_coords[j_slice] - gt_coords[i], axis=1)
    decoy_dist = np.linalg.norm(decoy_coords[j_slice] - decoy_coords[i], axis=1)

    is_considered = long_range & (gt_dist < cutoff_radius)

    if gt_pairwise_mask is not None and gt_pairwise_mask.size > 0:
      is_considered = is_considered & gt_pairwise_mask[i, i + 1 :]

    n_considered = int(np.sum(is_considered))
    num_considered_pairs += n_considered
    per_atom_considered[i] += n_considered
    per_atom_considered[i + 1 :] += is_considered.astype(np.int64)

    # For each tolerance, count pairs within tolerance.
    dist_diff = np.abs(gt_dist - decoy_dist)
    for tol in tolerances:
      in_cutoff = is_considered & (dist_diff <= tol)
      per_atom_in_cutoff[i] += int(np.sum(in_cutoff))
      per_atom_in_cutoff[i + 1 :] += in_cutoff.astype(np.int64)

  # Average across tolerances.
  av_in_cutoff = per_atom_in_cutoff.sum() / (2.0 * len(tolerances))

  residue_ranges = _group_by_residue_index(residue_indices)

  total_per_residue = _per_residue_sum(residue_ranges, per_atom_in_cutoff)
  conserved_per_residue = _per_residue_sum(
      residue_ranges, per_atom_considered * len(tolerances)
  )

  eps = 5e-7
  lddt_per_residue = (
      100.0
      * total_per_residue.astype(np.float64)
      / (eps + conserved_per_residue.astype(np.float64))
  )
  lddt_score = 100.0 * av_in_cutoff / (float(num_considered_pairs) + eps)

  return lddt_score, lddt_per_residue, num_considered_pairs


def lddt(
    decoy: structure.Structure,
    gt: structure.Structure,
    *,
    cutoff_radius: float = CUTOFF_RADIUS,
    min_sequence_separation: int = 1,
    one_atom_per_residue: bool = False,
    fail_if_missing_residues: bool = True,
    multichain_mode: MultiChainMode = MultiChainMode.ALL,
) -> LDDTResult:
  """Compute the LDDTs between two structures.

  NB: This is different to the official implementation used at
  https://swissmodel.expasy.org/lddt because it doesn't include any steric clash
  terms or terms for bond lengths and angles, and doesn't ignore missing
  residues.

  Args:
    decoy: A `Structure` instance representing the decoy structure.
    gt: A `Structure` instance representing the ground truth structure.
    cutoff_radius: Only pairs of residues with distance below this value in the
      ground truth are considered.
    min_sequence_separation: Only consider distances between atoms in residues
      with at least this sequence separation.
    one_atom_per_residue: Whether to score on only the carbon-alpha atoms.
    fail_if_missing_residues: If True, raises if residues in gt are missing from
      decoy.
    multichain_mode: One of MultiChainMode: ALL or BETWEEN_CHAINS_ONLY.

  Returns:
    An LDDTResult.
  """
  if min_sequence_separation < 0:
    raise ValueError('min_sequence_separation must be >= 0.')
  if not one_atom_per_residue:
    logging.log_first_n(
        logging.WARNING,
        (
            'metrics.lddt gives different results to the LDDT '
            'used in CAMEO. The latter is available as metrics.official_lddt. '
            'In testing the results differed by no more than 2 LDDT.'
        ),
        1,
    )
  if one_atom_per_residue:
    gt = gt.filter_polymers_to_single_atom_per_res()
  else:
    gt = gt.without_hydrogen()

  if not fail_if_missing_residues:
    gt = gt.order_and_drop_atoms_to_match(decoy, allow_missing_atoms=True)

  gt_pairwise_mask = None
  if multichain_mode == MultiChainMode.BETWEEN_CHAINS_ONLY:
    gt_pairwise_mask = np.ones([gt.num_atoms, gt.num_atoms], dtype=bool)
    for start, end in gt.iter_chain_ranges():
      gt_pairwise_mask[start:end, start:end] = False
  elif multichain_mode == MultiChainMode.WITHIN_CHAINS_ONLY:
    if gt.num_chains == 1:
      gt_pairwise_mask = np.ones([gt.num_atoms, gt.num_atoms], dtype=bool)
    else:
      gt_pairwise_mask = np.zeros([gt.num_atoms, gt.num_atoms], dtype=bool)
      for start, end in gt.iter_chain_ranges():
        gt_pairwise_mask[start:end, start:end] = True
  elif multichain_mode != MultiChainMode.ALL:
    raise NotImplementedError(
        f'MultiChainMode {multichain_mode} is not implemented in this module. '
        'Only ALL, BETWEEN_CHAINS_ONLY, and WITHIN_CHAINS_ONLY are supported.'
    )

  decoy = decoy.order_and_drop_atoms_to_match(gt)

  lddt_score, lddt_per_res_scores, num_considered_pairs = _lddt_from_coords(
      decoy_coords=decoy.coords,
      gt_coords=gt.coords,
      residue_indices=structure.multichain_residue_index(gt),
      cutoff_radius=cutoff_radius,
      min_sequence_separation=min_sequence_separation,
      tolerances=np.array(_LDDT_TOLERANCES, dtype=np.float32),
      gt_pairwise_mask=gt_pairwise_mask,
  )

  return LDDTResult(
      lddt=lddt_score,
      lddt_per_res=lddt_per_res_scores,
      num_considered_pairs=num_considered_pairs,
  )
