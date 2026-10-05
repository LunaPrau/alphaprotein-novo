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

"""Metrics for evaluating sterics (steric clashes)."""

from alphafold3 import structure
from alphafold3.cpp import string_array
from alphaprotein_novo.metrics import structure_constants
import numpy as np
from scipy.spatial import distance


def compute_distance_matrix(
    struct: structure.Structure,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes an all-atom distance matrix for a batch element of a `Structure`.

  Args:
    struct: The `Structure` structure to compute distances of.

  Returns:
    A tuple with:
      * a numpy array of shape [N, N] with the distance matrix,
      * a numpy matrix of shape [N] with the multichain residue indices,
      * a numpy matrix of shape [N] which is True in the positions
        corresponding to backbone atoms.
    The latter two matrices identify the atoms that correspond to each
    row/column in the distance matrix.
  """
  coords = struct.coords
  distances = distance.cdist(coords, coords)
  residue_indices = structure.multichain_residue_index(struct)

  is_backbone_atom = string_array.isin(
      struct.atom_name, set(structure_constants.PROTEIN_BACKBONE_ATOMS)
  )
  is_backbone_atom = np.logical_and(is_backbone_atom, struct.is_protein_mask)
  return distances, residue_indices, is_backbone_atom


def count_steric_clashes_from_distances(
    distances: np.ndarray,
    residue_indices: np.ndarray,
    is_backbone_atom: np.ndarray,
    threshold: float,
    min_sequence_separation: int,
    atom_mask: np.ndarray | None = None,
    normalize: bool = True,
) -> float:
  """Count steric clashes from a distance matrix.

  Typically `distances`, `residue_indices`, `is_backbone_atom` will be values
  returned by `compute_distance_matrix`. This allows for calculating multiple
  steric clash metrics without needing to recompute a distance matrix each time.

  Args:
    distances: An NxN distance matrix of inter-atom distances.
    residue_indices: A length N array of multichain residue indices - i.e.
      monotonically increasing. These are used to compute the sequence
      separation between atoms.
    is_backbone_atom: A length N array of boolean values denoting whether each
      atom is a backbone atom. This is used to exclude interactions between
      backbone atoms in neighboring atoms. This argument is only relevant when
      min_sequence_separation is 1.
    threshold: The distance threshold defining a clash (in angstroms).
    min_sequence_separation: Only distances between atoms at least this many of
      residues apart are considered. This must be at least 1 (i.e. we never
      consider distances within a residue).
    atom_mask: An optional 1-D length N array. Only consider clashes involving
      these atoms. Normalization is also set to the number of these atoms.
    normalize: If True, return the percent of considered atoms that participate
      in a clash (percent clashes). Otherwise, return the number of atoms
      considered that participate in a clash. Both options consider `atom_mask`,
      if provided.

  Returns:
    The percentage of atoms that participate in a steric clash.
  """
  if min_sequence_separation < 1:
    raise ValueError('min_sequence_separation must be >= 1.')
  num_atom = distances.shape[0]

  if residue_indices.shape != is_backbone_atom.shape:
    raise ValueError(
        'residue_indices and is_backbone_atom must have the same shape, got '
        f'{residue_indices.shape} and {is_backbone_atom.shape}'
    )

  if atom_mask is not None:
    if atom_mask.shape != distances.shape[:1]:
      raise ValueError(
          'atom_mask shape must be 1-D mask that matches distances shape, got '
          f'{atom_mask.shape} and {distances.shape[:1]}'
      )

  clash_count = 0
  # Manually iterate over atoms to prevent creating N x X matrices.
  for atom_i in range(num_atom):
    if atom_mask is not None and not atom_mask[atom_i]:
      continue
    seq_dists = np.abs(residue_indices[atom_i] - residue_indices)
    is_short_range = seq_dists < min_sequence_separation
    excluded = is_short_range

    if min_sequence_separation == 1:
      # We also need to exclude distances between backbone atoms in each residue
      # and those of its neighbours. If we have a min_sequence_separation > 1
      # then this is already taken care of. If not then we need to exclude these
      # manually.
      is_neighboring = seq_dists == 1
      is_backbone_interaction = np.logical_and(
          is_backbone_atom, is_backbone_atom[atom_i]
      )
      excluded = np.logical_or(
          is_short_range,
          np.logical_and(is_backbone_interaction, is_neighboring),
      )

    distances_included = distances[atom_i] + excluded * 1e9

    clashes = distances_included < threshold
    clash_count += np.any(clashes, axis=-1)

  if normalize:
    if atom_mask is not None:
      return (100.0 * clash_count) / sum(atom_mask)  # pyrefly: ignore[bad-return]
    else:
      return (100.0 * clash_count) / num_atom
  else:
    return clash_count  # pyrefly: ignore[bad-return]
