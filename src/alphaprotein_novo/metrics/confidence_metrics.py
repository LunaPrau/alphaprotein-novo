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

"""Model confidence post-processing metrics (pLDDT and PDE reductions)."""

from collections.abc import Mapping, Sequence
from typing import Any

from alphafold3 import structure
import numpy as np


def compute_pde_metrics(
    chain_pair_pde_mean: np.ndarray | Sequence[Sequence[float]],
    struct: structure.Structure,
) -> dict[str, float]:
  """Computes symmetrized chain-pair and interface PDE metrics for complexes.

  Args:
    chain_pair_pde_mean: 2D array or nested sequence of chain pair mean PDEs.
    struct: Structure with protein and ligand entities.

  Returns:
    Dictionary containing:
      - chain_pair_pde_mean_sym: Symmetrized PDE between primary chains.
      - protein_ligand_ipde_mean: Mean PDE across protein-ligand chain pairs.
      - protein_pde_mean: Mean intra-chain PDE for protein chains.
      - ligand_pde_mean: Mean intra-chain PDE for ligand chains.
  """
  chain_pair_pde_mean = np.asarray(chain_pair_pde_mean, dtype=float)
  chain_id_to_zero_based_index = {
      chain_id: i for i, chain_id in enumerate(struct.chains)
  }
  protein_struct = struct.filter_to_entity_type(protein=True)
  ligand_struct = struct.filter_to_entity_type(ligand=True)
  protein_chain_int_indices = [
      chain_id_to_zero_based_index[chain_id]
      for chain_id in protein_struct.chains
  ]
  ligand_chain_int_indices = [
      chain_id_to_zero_based_index[chain_id]
      for chain_id in ligand_struct.chains
  ]

  v_sym = (chain_pair_pde_mean + chain_pair_pde_mean.T) / 2
  sub_matrix = v_sym[
      np.ix_(protein_chain_int_indices, ligand_chain_int_indices)
  ]
  protein_ligand_ipde_mean = (
      float(np.mean(sub_matrix)) if sub_matrix.size > 0 else float(np.nan)
  )
  protein_pde_mean = (
      float(np.mean(np.diag(v_sym)[protein_chain_int_indices]))
      if protein_chain_int_indices
      else float(np.nan)
  )
  ligand_pde_mean = (
      float(np.mean(np.diag(v_sym)[ligand_chain_int_indices]))
      if ligand_chain_int_indices
      else float(np.nan)
  )
  if protein_chain_int_indices and ligand_chain_int_indices:
    p_idx = protein_chain_int_indices[0]
    l_idx = ligand_chain_int_indices[0]
    sym_val = float(v_sym[p_idx, l_idx])
  elif v_sym.size > 1:
    sym_val = float(v_sym[0, 1])
  else:
    sym_val = float(np.nan)

  return {
      'chain_pair_pde_mean_sym': sym_val,
      'protein_ligand_ipde_mean': protein_ligand_ipde_mean,
      'protein_pde_mean': protein_pde_mean,
      'ligand_pde_mean': ligand_pde_mean,
  }


def compute_plddt_summary(
    plddt: np.ndarray | Sequence[float],
    struct: structure.Structure,
) -> dict[str, float]:
  """Computes mean pLDDT for protein and ligand atoms separately.

  Args:
    plddt: 1D array or sequence of per-atom pLDDT values matching
      struct.num_atoms.
    struct: Structure containing protein and optional ligand atoms.

  Returns:
    Dictionary containing:
      - protein_plddt_mean: Mean pLDDT for protein atoms.
      - ligand_plddt_mean: Mean pLDDT for ligand atoms.
  """
  plddt = np.asarray(plddt, dtype=float)
  struct_with_plddt = struct.copy_and_update_atoms(atom_b_factor=plddt)
  p_struct = struct_with_plddt.filter_to_entity_type(protein=True)
  protein_plddt = (
      float(np.mean(p_struct.atom_b_factor))
      if p_struct.num_atoms > 0
      else float(np.nan)
  )
  l_struct = struct_with_plddt.filter_to_entity_type(ligand=True)
  ligand_plddt = (
      float(np.mean(l_struct.atom_b_factor))
      if l_struct.num_atoms > 0
      else float(np.nan)
  )
  return {
      'protein_plddt_mean': protein_plddt,
      'ligand_plddt_mean': ligand_plddt,
  }


def calculate_confidence_metrics(
    metrics: Mapping[str, Any],
    struct: structure.Structure,
    prefix: str,
) -> dict[str, Any]:
  """Computes customized confidence metrics for evaluation and inference.

  Args:
    metrics: Raw input metrics dictionary.
    struct: Structure corresponding to predictions.
    prefix: Metrics prefix ('complex' or 'monomer').

  Returns:
    Dictionary containing post-processed confidence metrics.
  """
  customized_metrics = {}
  for k, v in metrics.items():
    if k == 'chain_pair_pde_mean':
      if prefix == 'monomer':
        if 'predicted_distance_error' not in metrics:
          pde_arr = np.asarray(v, dtype=float)
          if pde_arr.size > 0:
            customized_metrics['predicted_distance_error'] = float(
                np.mean(pde_arr)
            )
      elif prefix == 'complex':
        customized_metrics.update(compute_pde_metrics(v, struct))
    elif (
        k in ('plddt', 'per_atom_plddt')
        and isinstance(v, (np.ndarray, Sequence))
        and not isinstance(v, (str, bytes))
    ):
      if np.asarray(v).size == struct.num_atoms:
        customized_metrics.update(compute_plddt_summary(v, struct))
    else:
      customized_metrics[k] = v
  return customized_metrics


def compute_confidence_metrics(
    metrics: Mapping[str, Any],
    struct: structure.Structure,
    *,
    prefix: str | None = None,
) -> dict[str, Any]:
  """Computes confidence metrics and reductions for a structure.

  Args:
    metrics: Raw input confidence metrics dictionary.
    struct: Structure corresponding to predictions.
    prefix: Optional metrics prefix ('complex' or 'monomer'). If None,
      automatically determined based on whether ligand atoms or multiple chains
      are present.

  Returns:
    Dictionary containing post-processed confidence metrics.
  """
  if prefix is None:
    has_ligand = struct.filter_to_entity_type(ligand=True).num_atoms > 0
    prefix = 'complex' if has_ligand or len(struct.chains) > 1 else 'monomer'
  return calculate_confidence_metrics(metrics, struct, prefix=prefix)
