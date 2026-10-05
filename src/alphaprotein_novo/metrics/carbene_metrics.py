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

"""Carbene transfer metrics for protein design evaluation."""

from typing import Any

from alphafold3 import structure
from alphaprotein_novo.metrics import ligand_metrics
import numpy as np


def alanine_ratio(
    struct_or_seq: structure.Structure | str,
) -> float:
  """Computes the ratio of alanine residues in a protein structure or sequence.

  Args:
    struct_or_seq: Structure or sequence string to compute alanine ratio for.

  Returns:
    Ratio of alanine residues (between 0.0 and 1.0). If the structure or
    sequence is empty, returns 0.0. An empty structure has no present protein
    residues (i.e. `present_residues` is empty after filtering to protein
    entities, meaning there are no resolved protein atoms).
  """
  if isinstance(struct_or_seq, str):
    if not struct_or_seq:
      return 0.0
    return struct_or_seq.count('A') / len(struct_or_seq)

  protein_only = struct_or_seq.filter_to_entity_type(protein=True)
  if protein_only.present_residues.size == 0:
    return 0.0
  return float((protein_only.present_residues.name == 'ALA').mean())


def split_cofactor_ts_rmsds(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    *,
    heme_res_name: str = 'HEM',
    ts_res_name: str | None = None,
    pocket_radius: float = 10.0,
) -> dict[str, float]:
  """Computes pocket-aligned ligand RMSDs for heme and transition state.

  Args:
    decoy_structure: The predicted decoy structure.
    reference_structure: The reference structure (e.g. designed/sampled motif).
    heme_res_name: Residue name for the heme cofactor (default 'HEM').
    ts_res_name: Optional residue name for the transition state ligand. If None,
      auto-detected as non-heme ligands.
    pocket_radius: Distance cutoff in Angstroms for defining the pocket.

  Returns:
    Dictionary containing:
      'heme_rmsd': Pocket-aligned RMSD of the heme cofactor.
      'transition_state_rmsd': Pocket-aligned RMSD of the transition state
      fragment.

  Raises:
    ValueError: If protein or ligand chains are missing or if heme/TS ligands
      cannot be resolved unambiguously.
  """
  if ts_res_name is not None and ts_res_name == heme_res_name:
    raise ValueError(
        'Transition state residue name cannot match heme_res_name '
        f'({heme_res_name!r}).'
    )

  ref_prot = reference_structure.filter_to_entity_type(protein=True)
  if not ref_prot.chains:
    raise ValueError('No protein chain found in reference structure.')
  prot_chain_id = ref_prot.chain_id[0]

  decoy_prot = decoy_structure.filter_to_entity_type(protein=True)
  if not decoy_prot.chains:
    raise ValueError('No protein chain found in decoy structure.')
  decoy_prot_chain_id = decoy_prot.chain_id[0]

  ref_ligands = reference_structure.filter_to_entity_type(ligand=True)
  if not ref_ligands.chains:
    raise ValueError('No ligands found in reference structure.')

  decoy_ligands = decoy_structure.filter_to_entity_type(ligand=True)
  if not decoy_ligands.chains:
    raise ValueError('No ligands found in decoy structure.')

  ref_heme = reference_structure.filter(res_name=heme_res_name)
  heme_chain_ids = np.unique(ref_heme.chain_id)
  if heme_chain_ids.size == 0:
    raise ValueError(
        f'No {heme_res_name} cofactor found in reference structure.'
    )
  if heme_chain_ids.size > 1:
    raise ValueError(
        f'Multiple {heme_res_name} cofactors found in reference structure: '
        f'{list(heme_chain_ids)}.'
    )
  heme_chain = heme_chain_ids[0]

  decoy_heme = decoy_structure.filter(res_name=heme_res_name)
  decoy_heme_chain_ids = np.unique(decoy_heme.chain_id)
  if decoy_heme_chain_ids.size == 0:
    raise ValueError(f'No {heme_res_name} cofactor found in decoy structure.')
  if decoy_heme_chain_ids.size > 1:
    raise ValueError(
        f'Multiple {heme_res_name} cofactors found in decoy structure: '
        f'{list(decoy_heme_chain_ids)}.'
    )
  decoy_heme_chain = decoy_heme_chain_ids[0]

  if ts_res_name is not None:
    ref_ts = reference_structure.filter(res_name=ts_res_name)
    ts_chain_ids = list(np.unique(ref_ts.chain_id))
    decoy_ts = decoy_structure.filter(res_name=ts_res_name)
    decoy_ts_chain_ids = list(np.unique(decoy_ts.chain_id))
  else:
    ts_chain_ids = list(
        np.unique(ref_ligands.filter_out(chain_id=heme_chain).chain_id)
    )
    decoy_ts_chain_ids = list(
        np.unique(decoy_ligands.filter_out(chain_id=decoy_heme_chain).chain_id)
    )

  if not ts_chain_ids:
    raise ValueError('No transition state ligand found in reference structure.')
  if len(ts_chain_ids) > 1:
    raise ValueError(
        'Multiple transition state ligands found in reference structure: '
        f'{ts_chain_ids}.'
    )
  ts_chain = ts_chain_ids[0]

  if not decoy_ts_chain_ids:
    raise ValueError('No transition state ligand found in decoy structure.')
  if len(decoy_ts_chain_ids) > 1:
    raise ValueError(
        'Multiple transition state ligands found in decoy structure: '
        f'{decoy_ts_chain_ids}.'
    )
  decoy_ts_chain = decoy_ts_chain_ids[0]

  folded_heme_complex = decoy_structure.filter(
      chain_id=[decoy_prot_chain_id, decoy_heme_chain]
  )
  sampled_heme_complex = reference_structure.filter(
      chain_id=[prot_chain_id, heme_chain]
  )
  folded_ts_complex = decoy_structure.filter(
      chain_id=[decoy_prot_chain_id, decoy_ts_chain]
  )
  sampled_ts_complex = reference_structure.filter(
      chain_id=[prot_chain_id, ts_chain]
  )

  heme_pal = ligand_metrics.compute_pocket_aligned_metrics(
      folded_heme_complex,
      sampled_heme_complex,
      pocket_radius=pocket_radius,
  )
  heme_rmsd = float(list(heme_pal.rmsd_per_ligand_chain_id.values())[0])

  ts_pal = ligand_metrics.compute_pocket_aligned_metrics(
      folded_ts_complex,
      sampled_ts_complex,
      pocket_radius=pocket_radius,
  )
  ts_rmsd = float(list(ts_pal.rmsd_per_ligand_chain_id.values())[0])

  return {
      'heme_rmsd': heme_rmsd,
      'transition_state_rmsd': ts_rmsd,
  }


def carbene_structure_metrics(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure | None = None,
    *,
    heme_res_name: str = 'HEM',
    ts_res_name: str | None = None,
    pocket_radius: float = 10.0,
) -> dict[str, Any]:
  """Computes carbene transfer structure metrics.

  Args:
    decoy_structure: The predicted structure to evaluate.
    reference_structure: Optional reference structure. If provided,
      pocket-aligned ligand RMSDs are computed for the heme cofactor and
      transition state ligand.
    heme_res_name: Residue name for the heme cofactor (default 'HEM').
    ts_res_name: Optional residue name for the transition state ligand. If None,
      auto-detected as non-heme ligands.
    pocket_radius: Distance cutoff in Angstroms for defining the pocket.

  Returns:
    Dictionary containing structure metrics:
      'alanine_ratio': Fraction of alanine residues in the decoy protein.
      'heme_rmsd': Pocket-aligned RMSD of the heme cofactor (if reference
        is provided).
      'transition_state_rmsd': Pocket-aligned RMSD of the transition state
        fragment (if reference is provided).
  """
  protein_only = decoy_structure.filter_to_entity_type(protein=True)
  metrics_dict = {
      'alanine_ratio': alanine_ratio(protein_only),
  }
  if reference_structure is not None:
    metrics_dict.update(
        split_cofactor_ts_rmsds(
            decoy_structure,
            reference_structure,
            heme_res_name=heme_res_name,
            ts_res_name=ts_res_name,
            pocket_radius=pocket_radius,
        )
    )
  return metrics_dict
