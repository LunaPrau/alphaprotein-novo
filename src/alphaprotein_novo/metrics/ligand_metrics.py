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

"""Metrics specific to ligands."""

from collections.abc import Callable
import dataclasses

from absl import logging
from alphafold3 import structure
from alphafold3.cpp import string_array
from alphaprotein_novo.metrics import geometry
from alphaprotein_novo.metrics import sterics
from alphaprotein_novo.metrics import structure_constants
import numpy as np
from scipy.spatial import distance


def safe_get_alignment_transform(
    x: np.ndarray,
    y: np.ndarray,
    included_idxs: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
  """Safe version of pocket_extraction.get_alignment_transform."""
  if x[included_idxs, :].shape[0] == 0:
    # Can't calculate alignment if there are no atoms; for this case, return the
    # identity transformation.
    logging.warning(
        'Returning identity transform from `safe_get_alignment_transform()`.'
        ' This might result in distorted metrics if they are computed based on'
        ' non-aligned structures.'
    )
    return lambda x: x
  return geometry.get_alignment_transform(
      x=x, y=y, x_indices=included_idxs, y_indices=included_idxs
  )


def get_pocket_aligned_transform(
    decoy_struc: structure.Structure,
    gt_struc_one_lig: structure.Structure,
    pocket_radius: float = 10.0,
    atom_names: tuple[str, ...] | None = ('N', 'CA', 'C'),
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns a transform to align the decoy to the pocket in the gt structure."""
  gt_ligand = gt_struc_one_lig.filter_to_entity_type(ligand=True)
  if gt_ligand.num_atoms == 0:
    return lambda x: x

  atom_distances = distance.cdist(gt_struc_one_lig.coords, gt_ligand.coords)
  pocket_mask = np.logical_and(
      np.any(atom_distances <= pocket_radius, axis=1),
      np.logical_not(gt_struc_one_lig.is_ligand_mask),
  )

  gt_pocket = gt_struc_one_lig.filter(pocket_mask)
  if atom_names is not None:
    gt_pocket = gt_pocket.filter(atom_name=atom_names)
  decoy_pocket = decoy_struc.order_and_drop_atoms_to_match(gt_pocket)
  if decoy_pocket.num_atoms != gt_pocket.num_atoms or gt_pocket.num_atoms == 0:
    return lambda x: x

  return safe_get_alignment_transform(
      decoy_pocket.coords,
      gt_pocket.coords,
      included_idxs=np.arange(gt_pocket.num_atoms),
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PocketAlignedMetrics:
  """Results from compute_pocket_aligned_metrics."""

  rmsd_per_ligand_chain_id: dict[str, float]
  pocket_bb_rmsds: list[float]
  pocket_allatom_rmsds: list[float]
  pal_bb_center_dists: list[float]

# Distance threshold to consider two heavy atoms to be covalently bonded.
# Higher than typical (1.3-1.5Å) to include partial bonds in transition states.
_COVALENT_BOND_CUTOFF = 1.9


def _ligand_atom_keys(
    struc: structure.Structure,
) -> set[tuple[str, int, str, str]]:
  """Returns (chain_id, res_id, res_name, atom_name) keys for heavy ligand atoms."""
  lig = struc.without_hydrogen().filter_to_entity_type(ligand=True)
  return set(zip(lig.chain_id, lig.res_id, lig.res_name, lig.atom_name))


def _ligand_bonded_atom_pairs(
    struc: structure.Structure,
    bond_cutoff: float = _COVALENT_BOND_CUTOFF,
) -> set[tuple[str, int, str, str, str]]:
  """Returns non-metal covalently bonded heavy-atom pairs within each ligand residue."""
  lig = struc.without_hydrogen().filter_to_entity_type(ligand=True)
  if lig.num_atoms < 2:
    return set()
  dists = distance.cdist(lig.coords, lig.coords)
  i_idxs, j_idxs = np.where(
      np.triu((dists > 0.0) & (dists <= bond_cutoff), k=1)
  )
  pairs: set[tuple[str, int, str, str, str]] = set()
  for i, j in zip(i_idxs, j_idxs):
    if (
        str(lig.atom_element[i]).upper() == 'FE'
        or str(lig.atom_element[j]).upper() == 'FE'
    ):
      continue
    if (
        lig.chain_id[i] == lig.chain_id[j]
        and lig.res_id[i] == lig.res_id[j]
        and lig.res_name[i] == lig.res_name[j]
    ):
      a1, a2 = str(lig.atom_name[i]), str(lig.atom_name[j])
      if a1 > a2:
        a1, a2 = a2, a1
      pairs.add((
          str(lig.chain_id[i]),
          int(lig.res_id[i]),
          str(lig.res_name[i]),
          a1,
          a2,
      ))
  return pairs


def has_matching_ligands(
    decoy_struc: structure.Structure,
    reference_structure: structure.Structure,
) -> bool:
  """Returns True if decoy and reference have matching ligand heavy atoms and bonds.

  Requires identical `(chain_id, res_id, res_name, atom_name)` heavy-atom key
  sets and compatible intra-residue covalent bond graphs (rejecting isomers or
  different SMILES atom-numbering topologies that happen to share atom names,
  such as 4MU-Ac `es` vs `ti1S`, while allowing partially formed
  transition-state
  bonds as in carbene transfer or DEHP `es` vs `ti1`).

  Args:
    decoy_struc: The decoy structure to evaluate.
    reference_structure: The reference structure to compare against.
  """
  decoy_keys = _ligand_atom_keys(decoy_struc)
  ref_keys = _ligand_atom_keys(reference_structure)
  if not decoy_keys or decoy_keys != ref_keys:
    return False
  decoy_bonds = _ligand_bonded_atom_pairs(decoy_struc)
  ref_bonds = _ligand_bonded_atom_pairs(reference_structure)
  return decoy_bonds <= ref_bonds or ref_bonds <= decoy_bonds


def compute_pocket_aligned_metrics(
    decoy_struc: structure.Structure,
    reference_structure: structure.Structure,
    pocket_radius: float = 10.0,
) -> PocketAlignedMetrics:
  """Computes ligand metrics using a single pocket alignment pass.

  Consolidates pocket-aligned ligand RMSD, pocket backbone RMSD, pocket
  all-atom RMSD, and center distance into a single loop that shares pocket
  selection and reuses Kabsch transforms. This avoids redundant cdist and
  order_and_drop_atoms_to_match calls.

  Args:
    decoy_struc: The decoy structure to evaluate.
    reference_structure: The ground truth reference structure.
    pocket_radius: Distance threshold to define pocket atoms.

  Returns:
    A PocketAlignedMetrics named tuple containing per-ligand pocket backbone
    RMSDs, pocket all-atom RMSDs, pocket-aligned ligand RMSDs, and center
    distances.
  """
  if not has_matching_ligands(decoy_struc, reference_structure):
    raise structure.MissingAtomError(
        'Decoy and reference structures do not have matching ligand heavy'
        ' atoms.'
    )
  reference_structure = reference_structure.without_hydrogen()
  decoy_struc = decoy_struc.without_hydrogen().order_and_drop_atoms_to_match(
      reference_structure
  )

  decoy_protein_struc = decoy_struc.filter_to_entity_type(protein=True)
  decoy_protein_chains = decoy_protein_struc.chains

  gt_protein_struc = reference_structure.filter_to_entity_type(protein=True)
  gt_ligand_struc = reference_structure.filter_to_entity_type(ligand=True)
  gt_protein_chains = gt_protein_struc.chains
  gt_ligand_chains = gt_ligand_struc.chains

  rmsd_per_ligand_chain_id: dict[str, float] = {}
  pocket_bb_rmsds: list[float] = []
  pocket_allatom_rmsds: list[float] = []
  pal_bb_center_dists: list[float] = []

  for chain_id in gt_ligand_chains:
    decoy_one_lig = decoy_struc.filter(
        chain_id=decoy_protein_chains + (chain_id,)
    )
    gt_one_lig = reference_structure.filter(
        chain_id=gt_protein_chains + (chain_id,)
    )

    gt_ligand = gt_one_lig.filter_to_entity_type(ligand=True)
    atom_distances = distance.cdist(gt_one_lig.coords, gt_ligand.coords)
    pocket_mask = np.logical_and(
        np.any(atom_distances <= pocket_radius, axis=1),
        np.logical_not(gt_one_lig.is_ligand_mask),
    )

    # Backbone alignment: use boolean masking instead of
    # order_and_drop_atoms_to_match since decoy_one_lig and gt_one_lig are
    # already atom-aligned.
    pocket_bb_mask = np.logical_and(
        pocket_mask,
        string_array.isin(gt_one_lig.atom_name, {'N', 'CA', 'C', 'O'}),
    )
    gt_pocket_bb_coords = gt_one_lig.coords[pocket_bb_mask]
    decoy_pocket_bb_coords = decoy_one_lig.coords[pocket_bb_mask]

    transform_bb = safe_get_alignment_transform(
        decoy_pocket_bb_coords,
        gt_pocket_bb_coords,
        included_idxs=np.arange(gt_pocket_bb_coords.shape[0]),
    )

    # 1. Pocket BB RMSD
    aligned_pocket_bb = transform_bb(decoy_pocket_bb_coords)
    aligned_pocket_bb_sq_norm = np.square(
        np.linalg.norm(aligned_pocket_bb - gt_pocket_bb_coords, axis=1)
    )
    pocket_bb_rmsds.append(np.sqrt(np.mean(aligned_pocket_bb_sq_norm)))

    # 2. Pocket Aligned Ligand RMSD (using BB transform)
    decoy_ligand = decoy_one_lig.filter_to_entity_type(ligand=True)
    aligned_decoy_ligand_coords = transform_bb(decoy_ligand.coords)
    deviations = np.linalg.norm(
        aligned_decoy_ligand_coords - gt_ligand.coords, axis=1
    )
    rmsd_per_ligand_chain_id[chain_id] = np.sqrt(np.mean(np.square(deviations)))

    # 3. Center Distance (using BB transform)
    aligned_decoy_ligand_center = np.mean(aligned_decoy_ligand_coords, axis=0)
    gt_ligand_center = np.mean(gt_ligand.coords, axis=0)
    pal_bb_center_dists.append(
        np.linalg.norm(aligned_decoy_ligand_center - gt_ligand_center)  # pyrefly: ignore[bad-argument-type]
    )

    # 4. Pocket All-Atom Alignment and RMSD: use boolean masking directly
    # since structures are already atom-aligned.
    gt_pocket_all_coords = gt_one_lig.coords[pocket_mask]
    decoy_pocket_all_coords = decoy_one_lig.coords[pocket_mask]
    transform_all = safe_get_alignment_transform(
        decoy_pocket_all_coords,
        gt_pocket_all_coords,
        included_idxs=np.arange(gt_pocket_all_coords.shape[0]),
    )
    aligned_pocket_all = transform_all(decoy_pocket_all_coords)
    aligned_pocket_all_sq_norm = np.square(
        np.linalg.norm(aligned_pocket_all - gt_pocket_all_coords, axis=1)
    )
    pocket_allatom_rmsds.append(np.sqrt(np.mean(aligned_pocket_all_sq_norm)))

  return PocketAlignedMetrics(
      rmsd_per_ligand_chain_id=rmsd_per_ligand_chain_id,
      pocket_bb_rmsds=pocket_bb_rmsds,
      pocket_allatom_rmsds=pocket_allatom_rmsds,
      pal_bb_center_dists=pal_bb_center_dists,
  )


def percent_backbone_clashes(
    decoy_struc: structure.Structure, threshold: float = 2.0
) -> float:
  """Returns the percent of ligand atoms that clash with protein backbone atoms.

  This function iterates over each ligand chain individually, computing clashes
  between that ligand and the protein backbone only. This avoids counting
  inter-ligand clashes as backbone clashes.

  Args:
    decoy_struc: The decoy structure.
    threshold: The threshold distance to consider a clash.

  Returns:
    Percent of ligand atoms that clash with protein backbone atoms.
  """

  # Protein backbone atoms.
  protein_only = decoy_struc.filter_to_entity_type(protein=True)
  protein_bb_only = protein_only.filter(
      atom_name=structure_constants.PROTEIN_BACKBONE_ATOMS
  )

  ligand_only = decoy_struc.filter_to_entity_type(ligand=True)
  ligand_chains = ligand_only.chains

  if not ligand_chains:
    return 0.0

  # Iterate over each ligand individually, count clashes with protein backbone,
  # then aggregate: total_clashing_atoms / total_ligand_atoms.
  total_clashing_atoms = 0
  total_ligand_atoms = 0

  for chain_id in ligand_chains:
    single_ligand = ligand_only.filter(chain_id=chain_id)
    # Combine this single ligand with protein backbone only.
    proteinbb_and_single_ligand = structure.concat(
        [protein_bb_only, single_ligand]
    )
    # Count clashes for this ligand (using percent_atom_clashes logic).
    distances, residue_indices, is_backbone_atom = (
        sterics.compute_distance_matrix(proteinbb_and_single_ligand)
    )
    num_clashing = sterics.count_steric_clashes_from_distances(
        distances=distances,
        residue_indices=residue_indices,
        is_backbone_atom=is_backbone_atom,
        threshold=threshold,
        min_sequence_separation=1,
        atom_mask=proteinbb_and_single_ligand.is_ligand_mask,
        normalize=False,
    )
    total_clashing_atoms += num_clashing
    total_ligand_atoms += single_ligand.num_atoms

  return 100.0 * total_clashing_atoms / total_ligand_atoms
