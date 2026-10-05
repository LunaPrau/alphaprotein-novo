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

"""Alignment based metrics (GDT-HA, GDT-TS, TM-score, RMSD)."""

from collections.abc import Callable, Sequence
from typing import Final, NamedTuple, TypeVar

from alphafold3 import structure
from alphaprotein_novo.metrics import geometry
import numpy as np

_GDT_THRESHOLDS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 4.0, 8.0)


class EmptyAlignedStructureError(ValueError):
  """Error raised when prediction and ground truth have no overlapping atoms."""


class ScoreResult(NamedTuple):
  gdt_ts: float
  gdt_ha: float
  tm_score: float


class AlignmentScoresResult(NamedTuple):
  gdt_ts: np.ndarray
  gdt_ha: np.ndarray
  tm_score: np.ndarray
  rmsd: np.ndarray


def _get_next_res_to_align(
    dists: np.ndarray, init_threshold: float
) -> np.ndarray:
  """Get indices of residues within the distance threshold for alignment."""
  threshold = init_threshold
  mask = dists < threshold
  while np.sum(mask) < 3:
    threshold += 0.5
    mask = dists < threshold
  return np.where(mask)[0]


def _align_coords(
    target: np.ndarray, indices_to_align: np.ndarray, x: np.ndarray
) -> np.ndarray:
  """Align x to target using the given indices."""
  sliced_x = x[indices_to_align]
  sliced_target = target[indices_to_align]

  sliced_x_mean = sliced_x.mean(axis=0)
  sliced_target_mean = sliced_target.mean(axis=0)

  t = geometry.transform_ls(
      sliced_x - sliced_x_mean, sliced_target - sliced_target_mean
  )
  aligned = (x - sliced_x_mean) @ t.T + sliced_target_mean
  return aligned


def _get_final_gdt(
    best_counts: list[int], num_res: int, is_gdt_ts: bool
) -> float:
  """Compute GDT-TS or GDT-HA from best fragment counts."""
  threshold_offset = 1 if is_gdt_ts else 0
  count_sum = sum(
      best_counts[i]
      for i in range(
          threshold_offset, len(_GDT_THRESHOLDS) - 1 + threshold_offset
      )
  )
  return 100.0 * count_sum / ((len(_GDT_THRESHOLDS) - 1) * num_res + 1e-8)


def score(
    decoy_coords: np.ndarray,
    gt_coords: np.ndarray,
    *,
    min_init_set_size: int = 4,
    max_init_set_sizes: int = 6,
    max_num_iter: int = 25,
) -> ScoreResult:
  """Compute GDT-TS, GDT-HA and TM-score for two coordinate arrays.

  Args:
    decoy_coords: [N, 3] numpy array of decoy atom coordinates.
    gt_coords: [N, 3] numpy array of ground truth atom coordinates.
    min_init_set_size: Minimum size of initial fragment set.
    max_init_set_sizes: Maximum number of different initial set sizes to try.
    max_num_iter: Maximum number of alignment iterations per fragment.

  Returns:
    A ScoreResult with gdt_ts, gdt_ha, and tm_score.
  """
  if decoy_coords.shape != gt_coords.shape:
    raise ValueError(
        'decoy_coords.shape and gt_coords.shape must match. Found: '
        f'{decoy_coords.shape} and {gt_coords.shape}.'
    )
  if decoy_coords.ndim != 2 or decoy_coords.shape[1] != 3:
    raise ValueError(
        f'Expected coordinates of shape (N, 3), got {decoy_coords.shape}.'
    )

  num_res = gt_coords.shape[0]

  best_counts = [0] * len(_GDT_THRESHOLDS)
  best_unnormalized_tm = 0.0

  # The heuristic for avg_dis comes from Zhang's original TM-score paper and
  # is roughly the average distance between residues for a protein of length
  # nseq. Ref: https://zhanglab.ccmb.med.umich.edu/TM-score/TM-score.pdf
  avg_dis = 1.24 * (max(num_res, 15) - 15) ** (1.0 / 3.0) - 1.8
  search_dis = max(min(avg_dis, 8.0), 4.5)

  # Build list of initial set sizes.
  init_set_sizes = []
  init_set_size = num_res
  for _ in range(max_init_set_sizes):
    init_set_size = max(init_set_size // 2, min_init_set_size)
    init_set_sizes.append(init_set_size)
    if init_set_size == min_init_set_size:
      break

  for iss in init_set_sizes:
    for start_idx in range(num_res - iss + 1):
      res_to_align = np.arange(start_idx, start_idx + iss)

      for search_i in range(max_num_iter):
        aligned_decoy = _align_coords(gt_coords, res_to_align, decoy_coords)
        dists = np.linalg.norm(gt_coords - aligned_decoy, axis=1)

        # Update best GDT counts.
        for ti, thresh in enumerate(_GDT_THRESHOLDS):
          count = int(np.sum(dists < thresh))
          if count > best_counts[ti]:
            best_counts[ti] = count

        # Update best TM-score.
        unnormalized_tm = float(np.sum(1.0 / (1.0 + (dists / avg_dis) ** 2)))
        if unnormalized_tm > best_unnormalized_tm:
          best_unnormalized_tm = unnormalized_tm

        # Determine next set of residues to align.
        init_search_threshold = (
            search_dis - 1.0 if search_i == 0 else search_dis + 1.0
        )
        prev_res_to_align = res_to_align
        res_to_align = _get_next_res_to_align(dists, init_search_threshold)

        if (
            search_i > 1
            and len(prev_res_to_align) == len(res_to_align)
            and np.array_equal(prev_res_to_align, res_to_align)
        ):
          break

  gdt_ts = _get_final_gdt(best_counts, num_res, is_gdt_ts=True)
  gdt_ha = _get_final_gdt(best_counts, num_res, is_gdt_ts=False)
  tm_score = 100.0 * (best_unnormalized_tm / (num_res + 1e-8))

  return ScoreResult(gdt_ts=gdt_ts, gdt_ha=gdt_ha, tm_score=tm_score)


def deviations_from_coords(
    decoy_coords: np.ndarray,
    gt_coords: np.ndarray,
    *,
    align_idxs: np.ndarray | None = None,
    include_idxs: np.ndarray | None = None,
) -> np.ndarray:
  """Returns the raw per-atom deviations used in RMSD computation."""
  if decoy_coords.shape != gt_coords.shape:
    raise ValueError(
        'decoy_coords.shape and gt_coords.shape must match.Found: %s and %s.'
        % (decoy_coords.shape, gt_coords.shape)
    )
  # Include and align all residues unless specified otherwise.
  if include_idxs is None:
    include_idxs = np.arange(decoy_coords.shape[0])
  if align_idxs is None:
    align_idxs = include_idxs
  aligned_decoy_coords = geometry.align(
      x=decoy_coords,
      y=gt_coords,
      x_indices=align_idxs,
      y_indices=align_idxs,
  )
  deviations = np.linalg.norm(
      aligned_decoy_coords[include_idxs] - gt_coords[include_idxs], axis=1
  )
  return deviations


def rmsd_from_coords(
    decoy_coords: np.ndarray,
    gt_coords: np.ndarray,
    *,
    align_idxs: np.ndarray | None = None,
    include_idxs: np.ndarray | None = None,
) -> float:
  """Computes the *aligned* RMSD of two Mx3 np arrays of coordinates."""
  deviations = deviations_from_coords(
      decoy_coords, gt_coords, align_idxs=align_idxs, include_idxs=include_idxs
  )
  return np.sqrt(np.mean(np.square(deviations)))


def rmsd(
    decoy: structure.Structure,
    gt: structure.Structure,
    *,
    one_atom_per_residue: bool = True,
    fail_if_missing_residues: bool = True,
) -> np.ndarray:
  """Compute the aligned RMSDs between two structures."""
  return np.array(
      _apply_metric_to_structure(
          metric_fn=rmsd_from_coords,
          decoy=decoy,
          ground_truth=gt,
          one_atom_per_residue=one_atom_per_residue,
          fail_if_missing_residues=fail_if_missing_residues,
      )
  )


def alignment_scores(
    decoy: structure.Structure,
    gt: structure.Structure,
    *,
    residue_ranges: Sequence[tuple[int, int]] | None = None,
    one_atom_per_residue: bool = True,
    fail_if_missing_residues: bool = True,
) -> AlignmentScoresResult:
  """Compute all alignment-based scores between two structures.

  WARNING 1: Setting fail_if_missing_residues=False will compute the
    scores only on a subset of atoms that match exactly between
    the decoy and the ground truth.

  WARNING 2: The first step in computing alignment scores is pairing decoy and
    ground truth atoms. This is done using (chain_id, res_id, atom_name).
    Therefore if the chain_id differs between the decoy and the ground truth,
    the output scores will be zero/undefined.

  Args:
    decoy: A `Structure` instance representing the decoy structure.
    gt: A `Structure` instance representing the ground truth structure.
    residue_ranges: Residue range to apply metric to.
    one_atom_per_residue: Whether to score on only the carbon-alpha atoms.
    fail_if_missing_residues: If True, raises an exception if atoms are missing.

  Returns:
    A named tuple with fields `gdt_ts`, `gdt_ha`, `tm_score` and `rmsd`.
  """

  if residue_ranges is not None:
    if any(start > end for start, end in residue_ranges):
      raise ValueError(f'Invalid residue range (start > end): {residue_ranges}')

    active_ranges = residue_ranges

    def is_relevant(res_index: int) -> bool:
      return any(start <= res_index <= end for start, end in active_ranges)

    gt = gt.filter(res_id=is_relevant, apply_per_element=True)
    decoy = decoy.filter(res_id=is_relevant, apply_per_element=True)

  scores = _apply_metric_to_structure(
      metric_fn=score,
      decoy=decoy,
      ground_truth=gt,
      one_atom_per_residue=one_atom_per_residue,
      fail_if_missing_residues=fail_if_missing_residues,
      sort_chains=True,
  )
  rmsd_val = rmsd(
      decoy,
      gt,
      one_atom_per_residue=one_atom_per_residue,
      fail_if_missing_residues=fail_if_missing_residues,
  )

  return AlignmentScoresResult(
      gdt_ts=np.array([s.gdt_ts for s in scores]),
      gdt_ha=np.array([s.gdt_ha for s in scores]),
      tm_score=np.array([s.tm_score for s in scores]),
      rmsd=rmsd_val,
  )


MetricRetType = TypeVar('MetricRetType')


def _apply_metric_to_structure(
    metric_fn: Callable[[np.ndarray, np.ndarray], MetricRetType],
    decoy: structure.Structure,
    ground_truth: structure.Structure,
    *,
    one_atom_per_residue: bool = True,
    fail_if_missing_residues: bool = True,
    sort_chains: bool = True,
) -> Sequence[MetricRetType]:
  """Extracts atom coords and applies metric_fn to them.

  Args:
    metric_fn: Callable which will be used to compute the metric value.
    decoy: Structure instance for the prediction(s).
    ground_truth: Structure instance for the ground truth(s).
    one_atom_per_residue: Whether to score on only the carbon-alpha atoms.
    fail_if_missing_residues: If True, raises an exception if atoms are missing.
    sort_chains: If True, reorder chains for stable metrics.

  Returns:
    A sequence of scores, each as returned by metric_fn.

  Raises:
    EmptyAlignedStructureError: If no atoms remain after matching.
  """
  if not fail_if_missing_residues:
    ground_truth = ground_truth.order_and_drop_atoms_to_match(
        decoy, allow_missing_atoms=True
    )
  if one_atom_per_residue:
    ground_truth = ground_truth.filter_polymers_to_single_atom_per_res()
  else:
    ground_truth = ground_truth.without_hydrogen()
  decoy = decoy.order_and_drop_atoms_to_match(ground_truth)

  if decoy.num_atoms == 0 or ground_truth.num_atoms == 0:
    raise EmptyAlignedStructureError(
        'After dropping atoms to match between the decoy and the ground truth, '
        f'there were {decoy.num_atoms} atoms left in the decoy and '
        f'{ground_truth.num_atoms} atoms left in the ground truth. Ensure that '
        'the chains you want to score have the same IDs in the two Structures. '
        'Structure.rename_{auth,label}_asym_ids can be used to rename chains.'
    )

  if sort_chains:
    decoy = decoy.with_sorted_chains
    ground_truth = ground_truth.with_sorted_chains

  return [metric_fn(decoy.coords, ground_truth.coords)]
