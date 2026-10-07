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

"""Classes and functions for specifying a protein design problem."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Self

from absl import logging
from alphafold3 import structure
from alphafold3.constants import atom_types
from alphafold3.constants import residue_names
from alphafold3.structure import mmcif
from alphaprotein_novo.data import data_constants
from alphaprotein_novo.data import pipeline_utils
from alphaprotein_novo.data import residue_mapping
from alphaprotein_novo.data import structure_utils
from alphaprotein_novo.model import denoiser
from alphaprotein_novo.model import types as types_lib
import jax
import numpy as np


def renumber_motif_string(
    motif_str: str,
    renamed_ch_res_by_ch_res: Mapping[
        residue_mapping.ResTuple, residue_mapping.ResTuple
    ],
) -> str:
  """Replace `motif_str` residue ids with new ones from `res_id_map`.

  Can be used to renumber a motif string from author to internal numbering, or
  from input ("source") to design ("target") numbering.

  For example, if `motif_str` is "A1,10,C2-4/D3" and `renamed_ch_res_by_ch_res`
  is {('A', 1): ('A', 1), ('C', 2): ('A', 3), ('C', 3): ('A', 4),
  ('C', 4): ('A', 5), ('D', 3): ('B', 1)}, then the returned motif string will
  be "A1,10,A3-5/B1".

  Args:
    motif_str: The motif string to renumber.
    renamed_ch_res_by_ch_res: A mapping from (chain_id, residue_id) to
      (chain_id, residue_id) for all residues in the input structure. Can be
      obtained from `get_author_chain_residue_index_map` or
      `residue_map.get_input_to_target_residue_map()`.

  Returns:
    The renumbered motif string.
  """
  if not motif_str:
    return motif_str

  def _motif_segment_to_internal(segment: str) -> str:
    ch, res = residue_mapping.separate_chain_id_from_res_ids(segment)
    res_split = res.split('-')
    internal_ch_res = [
        renamed_ch_res_by_ch_res[(ch, int(r))] for r in res_split
    ]
    if len({c for c, _ in internal_ch_res}) > 1:
      raise ValueError(
          f'Segment {segment} in input numbering maps to multiple'
          f' chains in output numbering: {internal_ch_res}.'
      )
    internal_segment = internal_ch_res[0][0] + '-'.join(
        [str(r) for _, r in internal_ch_res]
    )
    return internal_segment

  # Left of "|" is a comma-separated list of motifs to place in random
  # placeholders "{}" in the motif string.
  prefix = None
  if '|' in motif_str:
    prefix, motif_str = motif_str.split('|')
    internal_segments = []
    for segment in prefix.split(','):
      if segment[0].isalpha():
        internal_segment = _motif_segment_to_internal(segment)
      else:
        raise ValueError(
            'String left of "|" must be comma-delimited motif ranges,'
            f' i.e. start with chain letter. Input: {prefix}'
        )
      internal_segments.append(internal_segment)
    prefix = ','.join(internal_segments)

  internal_chains = []
  for chain in motif_str.split('/'):
    internal_segments = []
    for segment in chain.split(','):
      if segment[0].isalpha():
        internal_segment = _motif_segment_to_internal(segment)
      else:
        internal_segment = segment
      internal_segments.append(internal_segment)
    internal_chains.append(','.join(internal_segments))
  internal_motif_str = '/'.join(internal_chains)
  if prefix:
    internal_motif_str = f'{prefix}|{internal_motif_str}'

  return internal_motif_str


def renumber_motif_atoms(
    motif_atoms: str,
    renamed_ch_res_by_ch_res: Mapping[
        residue_mapping.ResTuple, residue_mapping.ResTuple
    ],
) -> str:
  """Replace `motif_atoms` residue ids with new ones from `res_id_map`.

  Can be used to renumber a motif atom string from author to internal numbering,
  or from input ("source") to design ("target") numbering.

  Args:
    motif_atoms: A string of atoms, in internal format, e.g. "A1/CA;B3/CB".
    renamed_ch_res_by_ch_res: A map from (chain_id, residue_id) to
      (new_chain_id, new_residue_id).

  Returns:
    The renumbered motif atom string, in the same format as `motif_atoms`.
  """
  if not motif_atoms:
    return ''
  new_res_tokens = []
  for res_token in motif_atoms.split():
    res_str, atoms_str = res_token.split(':')
    ch, res = residue_mapping.separate_chain_id_from_res_ids(res_str)
    new_ch, new_res = renamed_ch_res_by_ch_res[(ch, int(res))]
    new_res_token = f'{new_ch}{new_res}:{atoms_str}'
    new_res_tokens.append(new_res_token)
  return ' '.join(new_res_tokens)


def get_author_chain_residue_index_map(
    struct: structure.Structure,
) -> tuple[
    Mapping[residue_mapping.ResTuple, residue_mapping.ResTuple],
    Mapping[residue_mapping.ResTuple, residue_mapping.ResTuple],
]:
  """Returns dict mapping from author to internal (chain_id, res_id)."""
  auth2internal = {}
  internal2auth = {}
  for (
      internal_chain_id,
      res_id_map,
  ) in struct.author_naming_scheme.auth_seq_id.items():
    auth_chain_id = struct.author_naming_scheme.auth_asym_id[internal_chain_id]
    for internal_res_id, auth_res_id in res_id_map.items():
      internal_res_id = int(internal_res_id)
      auth_res_id = int(auth_res_id)
      if (auth_chain_id, auth_res_id) in auth2internal:
        logging.warning(
            'Duplicate author chain/residue index: %s/%s. Overwriting.',
            auth_chain_id,
            auth_res_id,
        )
      auth2internal[(auth_chain_id, auth_res_id)] = (
          internal_chain_id,
          internal_res_id,
      )
      internal2auth[(internal_chain_id, internal_res_id)] = (
          auth_chain_id,
          auth_res_id,
      )
  return auth2internal, internal2auth


def _diffusion_input_to_feature_dict(
    diffusion_input: types_lib.DiffusionInput,
) -> dict[str, Any]:
  """Convert DiffusionInput to feature dict."""
  # Convert Jax arrays to np.arrays to allow modification.
  features = {
      k: np.array(v) if isinstance(v, (np.ndarray, jax.Array)) else v
      for k, v in dataclasses.asdict(diffusion_input).items()
  }
  # Protein already becomes dict above but we have to do it again to get mutable
  # arrays and handle irregular field names.
  features['protein'] = diffusion_input.protein.to_feature_dict()
  return features


def _feature_dict_to_diffusion_input(
    features: dict[str, Any],
) -> types_lib.DiffusionInput:
  """Convert feature dict to DiffusionInput."""
  features['protein'] = types_lib.Protein.from_feature_dict(features['protein'])
  return types_lib.DiffusionInput(**features)


def _create_diffusion_protein_features_from_protein(
    protein: types_lib.Protein,
    *,
    num_res_unindexed: int = 128,
) -> dict[str, Any]:
  """Create diffusion features from a protein object, via DiffusionInput."""
  diff_input = types_lib.DiffusionInput(
      protein=protein,
      fixed_atom_mask=np.zeros_like(protein.atom_mask, dtype=int),  # pyrefly: ignore[bad-argument-type]
      fixed_seq_mask=np.zeros_like(protein.sequence_mask, dtype=int),  # pyrefly: ignore[bad-argument-type]
      unindexed_motif_atom_positions=np.zeros_like(
          protein.atom_positions[:num_res_unindexed]
      ),
      unindexed_motif_atom_mask=np.zeros_like(  # pyrefly: ignore[bad-argument-type]
          protein.atom_mask[:num_res_unindexed], dtype=int
      ),
      unindexed_motif_aatype=np.zeros_like(protein.aatype[:num_res_unindexed]),
      unindexed_motif_aatype_mask=np.zeros_like(  # pyrefly: ignore[bad-argument-type]
          protein.sequence_mask[:num_res_unindexed], dtype=int
      ),
      unindexed_motif_orig_res_id=np.zeros_like(
          protein.aatype[:num_res_unindexed]
      ),
      unindexed_motif_orig_chain_index=np.zeros_like(
          protein.aatype[:num_res_unindexed]
      ),
      t_struct=np.float32(0.01),  # pyrefly: ignore[bad-argument-type]
      t_seq=np.float32(0.01),  # pyrefly: ignore[bad-argument-type]
  )
  return _diffusion_input_to_feature_dict(diff_input)


def get_processed_motif_str(motif_str: str) -> str:
  """Formats motif strings containing '|' into standard segment notation."""
  if '|' in motif_str:
    motifs, motif_str_to_format = motif_str.split('|')
    return motif_str_to_format.format(*motifs.split(','))
  return motif_str


def create_designable_protein_features(
    res_map: residue_mapping.ResidueMap,
    *,
    max_num_res: int,
    max_num_ligands: int = 0,
    max_num_res_unindexed: int = 128,
) -> dict[str, Any]:
  """Create protein features representing a fully designable protein."""
  if len(res_map.source_residue_ids) > max_num_res:
    raise ValueError(
        f'Number of residues in motif_spec > {max_num_res}. Set max_num_res to'
        ' a higher value.'
    )
  actual_num_res = len(res_map.source_residue_ids)

  protein = types_lib.pad_protein(
      types_lib.make_blank_protein(actual_num_res),
      num_residues=max_num_res,
      num_ligands=max_num_ligands,
  )
  protein.sequence_mask[:actual_num_res] = 1
  protein.residue_index[:actual_num_res] = np.array(res_map.target_residue_ids)
  protein.atom_mask[:actual_num_res] = 1
  protein.chain_index[:actual_num_res] = np.array(
      [mmcif.str_id_to_int_id(i) for i in res_map.target_chain_ids]
  )
  protein.entity_id[:actual_num_res] = np.array(  # pyrefly: ignore[unsupported-operation]
      [mmcif.str_id_to_int_id(i) for i in res_map.target_chain_ids]
  )
  return _create_diffusion_protein_features_from_protein(
      protein,
      num_res_unindexed=max_num_res_unindexed,
  )


def add_fixed_residue_features(
    input_features: dict[str, Any],
    input_struct: structure.Structure,
    res_map: residue_mapping.ResidueMap,
    max_num_res: int,
) -> dict[str, Any]:
  """Add fixed protein residues to DiffusionInput.

  Adds features & coordinates for all atoms of residues with any fixed atoms.
  For tip atom residues and reseq residues, some of the atoms will be removed
  later by add_motif_atom_features and add_reseq_residue_features.

  Args:
    input_features: All input features.
    input_struct: Input structure.
    res_map: Residue map.
    max_num_res: Maximum number of residues.

  Returns:
    Updated features.
  """
  is_fixed_seq_protein = res_map.fixed_seq_mask & ~res_map.is_ligand_mask
  fixed_seq_mask = is_fixed_seq_protein | res_map.is_reseq_residue
  res_arrays = input_struct.to_res_arrays(include_missing_residues=False)
  source_atom_positions = res_arrays.atom_positions
  source_atom_mask = res_arrays.atom_mask
  res_names = input_struct.present_residues.name
  index_by_res_name = {r: i for i, r in enumerate(residue_names.PROTEIN_TYPES)}
  unk_index = residue_names.PROTEIN_TYPES_WITH_UNKNOWN.index(residue_names.UNK)
  source_aatype = np.array(
      [index_by_res_name.get(r, unk_index) for r in res_names]
  )
  residues = input_struct.iter_residues(include_unresolved=False)
  index_by_chain_res = {
      (r['chain_id'], int(r['res_id'])): i for i, r in enumerate(residues)
  }
  source_idx = []
  for chain_res in zip(
      res_map.source_chain_ids[fixed_seq_mask],
      res_map.source_residue_ids[fixed_seq_mask],
  ):
    if chain_res not in index_by_chain_res:
      raise ValueError(
          f'Motif residue {chain_res} not found in input structure.'
      )
    source_idx.append(index_by_chain_res[chain_res])

  padded_fixed_seq_mask = pipeline_utils.pad_to_size(
      fixed_seq_mask, max_num_res
  )
  input_features['fixed_seq_mask'][padded_fixed_seq_mask] = 1
  input_features['fixed_atom_mask'][padded_fixed_seq_mask] = source_atom_mask[
      source_idx
  ]
  input_features['crop_cond_atom_positions'][padded_fixed_seq_mask] = (
      source_atom_positions[source_idx]
  )
  input_features['crop_cond_aatype'][padded_fixed_seq_mask] = source_aatype[
      source_idx
  ]

  protein_features = input_features['protein']
  protein_features['all_atom_positions'][padded_fixed_seq_mask] = (
      source_atom_positions[source_idx]
  )
  protein_features['all_atom_mask'][padded_fixed_seq_mask] = source_atom_mask[
      source_idx
  ]
  protein_features['aatype'][padded_fixed_seq_mask] = source_aatype[source_idx]
  input_features['protein'] = protein_features

  return input_features


def add_ligand_features(
    input_features: dict[str, Any],
    input_struct: structure.Structure,
    res_map: residue_mapping.ResidueMap,
    max_num_res: int,
    ligand_mobile_atoms: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
  """Add ligand atom features and apply optional mobile-coordinate masks."""
  protein_features = input_features['protein']
  mobile_atoms_by_source_residue = {}
  for residue, atom_names in (ligand_mobile_atoms or {}).items():
    chain_id, residue_id = residue_mapping.separate_chain_id_from_res_ids(
        residue
    )
    key = (chain_id, int(residue_id))
    if key in mobile_atoms_by_source_residue:
      raise ValueError(f'Duplicate ligand_mobile_atoms residue reference: {residue}')
    if not atom_names or any(not atom for atom in atom_names):
      raise ValueError(
          f'ligand_mobile_atoms[{residue!r}] must contain at least one atom name.'
      )
    mobile_atoms_by_source_residue[key] = set(atom_names)
  used_mobile_residues = set()

  unique_chain_ids = []
  for i in res_map.source_chain_ids[res_map.is_ligand_mask]:
    if i not in unique_chain_ids:
      unique_chain_ids.append(i)

  for i_ligand, source_chain_id in enumerate(unique_chain_ids):
    target_mask = res_map.source_chain_ids == source_chain_id
    num_ligand_atoms = sum(target_mask)

    target_chain_id = np.unique(res_map.target_chain_ids[target_mask])
    if len(target_chain_id) > 1:
      raise ValueError(
          f'Multiple target chains {target_chain_id} found for source ligand'
          f' {source_chain_id}.'
      )
    target_chain_id = target_chain_id[0]

    ligand_struct = input_struct.filter(
        chain_id=source_chain_id
    ).without_hydrogen()
    source_residue_ids = set(
        int(res_id) for res_id in res_map.source_residue_ids[target_mask]
    )
    if len(source_residue_ids) != 1:
      raise ValueError(
          f'Ligand chain {source_chain_id} maps to source residues '
          f'{sorted(source_residue_ids)}; one residue per ligand chain is supported.'
      )
    source_residue_key = (str(source_chain_id), next(iter(source_residue_ids)))
    mobile_atom_names = mobile_atoms_by_source_residue.get(
        source_residue_key, set()
    )
    if mobile_atom_names:
      available_atom_names = set(map(str, ligand_struct.atom_name))
      unknown_atoms = mobile_atom_names - available_atom_names
      if unknown_atoms:
        raise ValueError(
            f'ligand_mobile_atoms for {source_residue_key} references absent '
            f'atom(s): {sorted(unknown_atoms)}.'
        )
      used_mobile_residues.add(source_residue_key)
    ligand_features = structure_utils.features_from_structure(
        struct=ligand_struct,  # pyrefly: ignore[bad-argument-type]
        num_residues=num_ligand_atoms,
        num_ligands=1,
    )

    padded_target_mask = pipeline_utils.pad_to_size(target_mask, max_num_res)

    protein_features['all_atom_positions'][padded_target_mask] = (
        ligand_features['all_atom_positions']
    )
    protein_features['all_atom_mask'][padded_target_mask] = ligand_features[
        'all_atom_mask'
    ]
    protein_features['aatype'][padded_target_mask] = ligand_features['aatype']
    protein_features['is_ligand_mask'][padded_target_mask] = 1

    if not np.all(ligand_features['component_entity_id'] == 1):
      raise ValueError(
          'Expected source ligand structure to have only 1 ligand.'
      )

    # Assign new entity id to this ligand. Assumes that protein.entity_id has
    # been assigned using the target_chain_ids from the same ResidueMap.
    protein_features['component_entity_id'][i_ligand] = mmcif.str_id_to_int_id(
        target_chain_id
    )  # (num_ligands,)

    for feat_name in [
        'component_id',  # (num_ligands,)
        'component_atom_names',  # (num_ligands, max_num_ligand_tokens)
        'component_atomic_numbers',  # (num_ligands, max_num_ligand_tokens)
        'component_fragment_indices',  # (num_ligands, max_num_ligand_tokens)
        'component_padding_mask_1d',  # (num_ligands, max_num_ligand_tokens)
    ]:
      protein_features[feat_name][i_ligand] = ligand_features[feat_name][0]

    ligand_fixed_atom_mask = np.array(
        ligand_features['all_atom_mask'], copy=True
    )
    if mobile_atom_names:
      ligand_token_atom_names = ligand_features['ligand_atom_names']
      mobile_rows = np.asarray(
          [str(name) in mobile_atom_names for name in ligand_token_atom_names]
      )
      mapped_mobile_names = {
          str(name) for name in ligand_token_atom_names[mobile_rows]
      }
      if mapped_mobile_names != mobile_atom_names:
        raise ValueError(
            f'Could not map every ligand_mobile_atoms name for '
            f'{source_residue_key} to Novo ligand atom order.'
        )
      ligand_fixed_atom_mask[mobile_rows, 0] = 0
    input_features['fixed_atom_mask'][padded_target_mask] = (
        ligand_fixed_atom_mask
    )
    input_features['fixed_seq_mask'][padded_target_mask] = 1
    input_features['ligand_charge'][padded_target_mask] = ligand_features[
        'ligand_charge'
    ]
    input_features['ligand_hybridization'][padded_target_mask] = (
        ligand_features['ligand_hybridization']
    )
    input_features['crop_cond_atom_positions'][padded_target_mask] = (
        ligand_features['all_atom_positions']
    )
    input_features['crop_cond_aatype'][padded_target_mask] = ligand_features[
        'aatype'
    ]

  input_features['protein'] = protein_features
  unused_mobile_residues = set(mobile_atoms_by_source_residue) - used_mobile_residues
  if unused_mobile_residues:
    raise ValueError(
        'ligand_mobile_atoms contains residues not present as ligand motifs: '
        f'{sorted(unused_mobile_residues)}.'
    )
  return input_features


def _residue_atom_masks(
    motif_atoms: str,
    source_chain_res: list[residue_mapping.ResTuple],
) -> dict[int, np.ndarray]:
  """Builds unioned atom37 masks per residue index from a motif_atoms string."""
  masks_by_res_idx: dict[int, np.ndarray] = {}
  for res_atoms_str in motif_atoms.split(' '):
    if not res_atoms_str:
      continue
    res, atom_str = res_atoms_str.split(':')
    ch, r = residue_mapping.separate_chain_id_from_res_ids(res)
    res_idx = source_chain_res.index((ch, int(r)))
    if res_idx not in masks_by_res_idx:
      masks_by_res_idx[res_idx] = np.zeros(37, dtype=int)
    for atom in filter(None, atom_str.split(',')):
      masks_by_res_idx[res_idx][atom_types.ATOM37_ORDER[atom]] = 1
  return masks_by_res_idx


def add_motif_atom_features(
    input_features: dict[str, Any],
    res_map: residue_mapping.ResidueMap,
    motif_atoms: str,
    is_partial_diffusion: bool = False,
) -> dict[str, Any]:
  """Update input feature dict with motif atom features."""
  protein_features = input_features['protein']
  source_chain_res = list(
      zip(res_map.source_chain_ids, res_map.source_residue_ids)
  )

  for res_idx, mask in _residue_atom_masks(
      motif_atoms, source_chain_res
  ).items():
    input_features['fixed_atom_mask'][res_idx] *= mask
    input_features['crop_cond_atom_positions'][res_idx] *= mask[..., None]
    # Mask out the non-fixed atoms, unless this is partial diffusion, in which
    # case we keep all the atoms.
    if not is_partial_diffusion:
      protein_features['all_atom_positions'][res_idx] *= mask[..., None]

  return input_features


def add_reseq_residue_features(
    input_features: dict[str, Any],
    res_map: residue_mapping.ResidueMap,
    reseq_residues: str,
    is_partial_diffusion: bool = False,
) -> dict[str, Any]:
  """Update input feature dict with resequenced residues."""
  protein_features = input_features['protein']
  source_chain_res = list(
      zip(res_map.source_chain_ids, res_map.source_residue_ids)
  )

  bb_mask = np.zeros(37, dtype=int)
  for atom in data_constants.BACKBONE_ATOMS_WITH_O:
    bb_mask[atom_types.ATOM37_ORDER[atom]] = 1

  for ch_res in residue_mapping.chain_and_residue_pairs(reseq_residues):
    res_idx = source_chain_res.index(ch_res)
    input_features['fixed_seq_mask'][res_idx] = 0
    input_features['fixed_atom_mask'][res_idx] *= bb_mask
    input_features['crop_cond_atom_positions'][res_idx] *= bb_mask[..., None]
    input_features['crop_cond_aatype'][res_idx] = 0
    # Mask out the aatype and sidechain atoms, unless this is partial
    # diffusion, in which case we keep the sequence and all the atoms.
    if not is_partial_diffusion:
      protein_features['aatype'][res_idx] = 0
      protein_features['all_atom_positions'][res_idx] *= bb_mask[..., None]

  input_features['protein'] = protein_features
  return input_features


def _apply_unindexed_motif_atoms(
    input_features: dict[str, Any],
    motif_atoms: str,
    protein_source_chain_res: list[residue_mapping.ResTuple],
) -> None:
  """Applies atom mask from motif_atoms to unindexed motif input_features."""
  for res_idx, mask in _residue_atom_masks(
      motif_atoms, protein_source_chain_res
  ).items():
    input_features['unindexed_motif_atom_positions'][res_idx] *= mask[..., None]
    input_features['unindexed_motif_atom_mask'][res_idx] = mask


def _add_unindexed_motif_protein_features(
    input_features: dict[str, Any],
    *,
    max_num_res: int,
    protein_fixed_seq_mask: np.ndarray,
    unindexed_motif_res_map: residue_mapping.ResidueMap,
    source_atom_positions: np.ndarray,
    source_atom_mask: np.ndarray,
    source_aatype: np.ndarray,
    source_idx: list[int],
) -> None:
  """Adds protein features to unindexed motif features."""
  padded_protein_fixed_seq_mask = pipeline_utils.pad_to_size(
      protein_fixed_seq_mask, max_num_res
  )
  input_features['unindexed_motif_atom_positions'] = np.zeros_like(
      input_features['protein']['all_atom_positions'][:max_num_res]
  )
  input_features['unindexed_motif_atom_mask'] = np.zeros_like(
      input_features['protein']['all_atom_mask'][:max_num_res]
  )
  input_features['unindexed_motif_aatype'] = np.zeros_like(
      input_features['protein']['aatype'][:max_num_res]
  )
  input_features['unindexed_motif_aatype_mask'] = np.zeros_like(
      input_features['protein']['aatype'][:max_num_res], dtype=int
  )
  input_features['unindexed_motif_atom_positions'][
      padded_protein_fixed_seq_mask
  ] = source_atom_positions[source_idx]
  input_features['unindexed_motif_atom_mask'][padded_protein_fixed_seq_mask] = (
      source_atom_mask[source_idx]
  )
  input_features['unindexed_motif_aatype'][padded_protein_fixed_seq_mask] = (
      source_aatype[source_idx]
  )
  input_features['unindexed_motif_aatype_mask'][
      padded_protein_fixed_seq_mask
  ] = 1
  input_features['unindexed_motif_orig_res_id'] = np.zeros_like(
      input_features['protein']['aatype'][:max_num_res], dtype=int
  )
  input_features['unindexed_motif_orig_res_id'][
      padded_protein_fixed_seq_mask
  ] = unindexed_motif_res_map.source_residue_ids[protein_fixed_seq_mask]
  input_features['unindexed_motif_orig_chain_index'] = np.zeros_like(
      input_features['protein']['aatype'][:max_num_res], dtype=int
  )
  orig_chain_ids = unindexed_motif_res_map.source_chain_ids[
      protein_fixed_seq_mask
  ]
  chain_id_to_idx = {
      chain_id: i + 1 for i, chain_id in enumerate(set(orig_chain_ids))
  }
  orig_chain_indices = [
      chain_id_to_idx[chain_id] for chain_id in orig_chain_ids
  ]
  input_features['unindexed_motif_orig_chain_index'][
      padded_protein_fixed_seq_mask
  ] = orig_chain_indices


def _apply_unindexed_motif_reseq_residues(
    input_features: dict[str, Any],
    reseq_residues: str,
    protein_source_chain_res: list[residue_mapping.ResTuple],
) -> None:
  """Applies reseq mask to unindexed motif features."""
  bb_mask = np.zeros(37, dtype=int)
  for atom in data_constants.BACKBONE_ATOMS_WITH_O:
    bb_mask[atom_types.ATOM37_ORDER[atom]] = 1

  for ch_res in residue_mapping.chain_and_residue_pairs(reseq_residues):
    res_idx = protein_source_chain_res.index(ch_res)
    input_features['unindexed_motif_atom_mask'][res_idx] *= bb_mask
    input_features['unindexed_motif_atom_positions'][res_idx] *= bb_mask[
        ..., None
    ]
    input_features['unindexed_motif_aatype'][res_idx] = 0
    input_features['unindexed_motif_aatype_mask'][res_idx] = 0


def _add_unindexed_motif_ligand_features(
    input_features: dict[str, Any],
    ligand_indices: np.ndarray,
    protein_fixed_seq_mask: np.ndarray,
    fixed_atom_mask: np.ndarray,
) -> None:
  """Adds ligand features to unindexed motif features."""
  num_ligands = len(ligand_indices)
  unindexed_protein_motif_indices = np.nonzero(protein_fixed_seq_mask)[0]
  start_idx = np.max(unindexed_protein_motif_indices) + 1
  end_idx = start_idx + num_ligands
  unindexed_ligand_indices = np.arange(start_idx, end_idx, dtype=int)
  input_features['unindexed_motif_atom_positions'][unindexed_ligand_indices] = (
      input_features['protein']['all_atom_positions'][ligand_indices]
      * fixed_atom_mask[ligand_indices, :, None]
  )
  input_features['unindexed_motif_atom_mask'][unindexed_ligand_indices] = (
      fixed_atom_mask[ligand_indices]
  )
  input_features['unindexed_motif_aatype'][unindexed_ligand_indices] = (
      input_features['protein']['aatype'][ligand_indices]
  )
  input_features['unindexed_motif_aatype_mask'][unindexed_ligand_indices] = (
      np.ones_like(
          input_features['protein']['aatype'][ligand_indices], dtype=int
      )
  )


def add_unindexed_motif_features(
    *,
    input_features: dict[str, Any],
    input_struct: structure.Structure,
    unindexed_motif_residues: str,
    reseq_residues: str,
    motif_atoms: str,
    max_num_res: int,
    fixed_atom_mask: np.ndarray | None = None,
) -> dict[str, Any]:
  """Update input feature dict with unindexed motif atom features."""
  unindexed_motif_res_map = residue_mapping.motif_str_to_residue_map(
      motif_str=unindexed_motif_residues,
      input_struct=input_struct,
      reseq_residues=reseq_residues,
  )
  protein_fixed_seq_mask = (
      unindexed_motif_res_map.fixed_seq_mask
      | unindexed_motif_res_map.is_reseq_residue
  ) & (~unindexed_motif_res_map.is_ligand_mask)
  protein_fixed_seq_mask = protein_fixed_seq_mask[:max_num_res]
  source_res_arrays = input_struct.to_res_arrays(include_missing_residues=False)
  source_atom_positions = source_res_arrays.atom_positions
  source_atom_mask = source_res_arrays.atom_mask
  res_names = input_struct.present_residues.name
  index_by_res_name = {r: i for i, r in enumerate(residue_names.PROTEIN_TYPES)}
  unk_index = residue_names.PROTEIN_TYPES_WITH_UNKNOWN.index(residue_names.UNK)
  source_aatype = np.array(
      [index_by_res_name.get(r, unk_index) for r in res_names]
  )
  residues = input_struct.iter_residues(include_unresolved=False)
  index_by_chain_res = {
      (r['chain_id'], int(r['res_id'])): i for i, r in enumerate(residues)
  }
  protein_source_chain_res = list(
      zip(
          unindexed_motif_res_map.source_chain_ids[protein_fixed_seq_mask],
          unindexed_motif_res_map.source_residue_ids[protein_fixed_seq_mask],
      )
  )
  source_idx = []
  for chain_res in protein_source_chain_res:
    if chain_res not in index_by_chain_res:
      raise ValueError(
          f'Unindexed motif residue {chain_res} not found in input structure.'
      )
    source_idx.append(index_by_chain_res[chain_res])

  ligand_mask = input_features['protein']['is_ligand_mask']
  ligand_indices = np.nonzero(ligand_mask)[0]
  if fixed_atom_mask is None:
    fixed_atom_mask = input_features['fixed_atom_mask']
  num_source_residues = len(source_idx) + len(ligand_indices)
  if num_source_residues > max_num_res:
    raise ValueError(
        f'Number of source residues {num_source_residues} is larger than the'
        f' maximum number of residues for unindexed motif {max_num_res}.'
    )

  _add_unindexed_motif_protein_features(
      input_features,
      max_num_res=max_num_res,
      protein_fixed_seq_mask=protein_fixed_seq_mask,
      unindexed_motif_res_map=unindexed_motif_res_map,
      source_atom_positions=source_atom_positions,
      source_atom_mask=source_atom_mask,
      source_aatype=source_aatype,
      source_idx=source_idx,
  )

  if motif_atoms:
    _apply_unindexed_motif_atoms(
        input_features, motif_atoms, protein_source_chain_res
    )

  if reseq_residues:
    _apply_unindexed_motif_reseq_residues(
        input_features, reseq_residues, protein_source_chain_res
    )

  # Add ligand features to the unindexed motif features as well,
  # placing them after the protein motif features.
  _add_unindexed_motif_ligand_features(
      input_features, ligand_indices, protein_fixed_seq_mask, fixed_atom_mask
  )

  return input_features


@dataclasses.dataclass(frozen=True, kw_only=True)
class MotifSpec:
  """Specification of a motif-conditioned protein design problem.

  Can represent unconditional design, (binding interface or active site) motif
  scaffolding, target-conditioned binder design, ligand-binding protein design,
  and enzyme design (fixed ligand + active site motif, or fixed catalytic tip
  atoms only).

  `motif_str` specifies which parts of the input structure represent motifs to
  condition on. A motif can be a structural fragment being scaffolded, an entire
  separate chain (i.e. binding target), or a non-protein chain (ligand). For
  example:

    "0-20,C2-6,2-30,C9-22,0-20" (motif scaffolding): Design a protein with 0-20
      generated residues, followed by chain C residues 2-6 from the input
      structure, then generate 2-30 residues, then chain C residues 9-22 from
      the input, then 0-20 generated residues. Length ranges indicate that on
      each sampling trajectory, a length is uniformly sampled from that closed
      interval.

    "40-120/R13-107/S13-107" (protein binder design): Design a protein with
      40-120 residues (sampling length uniformly from this closed interval on
      every trajectory), conditioning on chain R residues 13-107 and chain S
      residues 13-107 of the input (target proteins). Note that "/" indicates a
      chain break and calls for the corresponding input features.

    "150-230/A154" (ligand binder design): Design a protein with 150-230
      residues, conditioning on a 2nd chain consisting of chain A residue 154 of
      the input (a small molecule).

    "20-100,A56,20-70,A100,20-70,A125,20-100/A2001/A3001" (enzyme design):
      Scaffold the indicated (catalytic) residues, conditioning on the 2
      indicated ligands.

    "A56,A100,A125|20-100,{},20-70,{},20-70,{},20-100/A2001/A3001"
    (enzyme design):
      Scaffold the residues left of "|", placing them in random order where the
      "{}" placeholders are, conditioning on the 2 indicated ligands.

    "100-250/A2001/A3001" (unindexed motif enzyme design):
      Add the motif residues and motif atoms in the unindexed_motif_residues and
      motif_atoms fields. Eg "A56,A100,A125" for
      unindexed_motif_residues and "A56:NE2 A125:OD2" for motif_atoms.

  Attributes:
    name: Name of design problem.
    description: Description of design problem.
    input_file: Path to input containing motifs & structures to condition on.
    input_struct: structure.Structure containing structs to condition on.
    motif_str: String representing motif residues and linker lengths. May
      contain length ranges for designable regions.
    sampled_motif_str: Like `motif_str`, but with specific lengths sampled from
      ranges.
    is_author_naming: Whether to use author residue naming.
    motif_atoms: String representing subset of atoms in each residue to fix.
      Residues not specified will have all atoms fixed. Used for "flexible
      rotamer" catalytic site scaffolding (in both indexed and unindexed modes).
    seq_length: If set, then on each sampling trajectory, lengths are sampled
      according to `motif_str` repeatedly until a total design length falls
      within `seq_length`.
    reseq_residues: Residues on motifs allowed to change AA type (in both
      indexed and unindexed modes).
    unindexed_motif_residues: Motif residues to be used in unindexed motif
      conditioning.
    ligand_mobile_atoms: Source ligand residue-to-atom-name selections whose
      observed coordinates are not fixed during diffusion.
    residue_map: Map from input structure to design residue/ligand tokens.
    is_parsed: Whether the MotifSpec has been parsed and is ready to be
      converted to DiffusionInput.
    partial_diffusion_input_struct: The input structure to use for partial
      diffusion.
  """

  name: str
  input_file: str | None
  input_struct: structure.Structure | None = None
  partial_diffusion_input_struct: structure.Structure | None = None
  motif_str: str
  sampled_motif_str: str | None = None
  is_author_naming: bool | None
  description: str = ''
  motif_atoms: str = ''
  reseq_residues: str | None = None
  seq_length: str | None = None
  unindexed_motif_residues: str | None = None
  ligand_mobile_atoms: Mapping[str, Sequence[str]] | None = None
  residue_map: residue_mapping.ResidueMap | None = None
  is_parsed: bool = False

  def with_internal_naming(self) -> Self:
    """Return copy of self with author residue numbering mapped to internal."""
    if not self.is_author_naming:
      return self

    internal_named_fields = {}
    if self.input_struct is not None or self.input_file:
      if self.input_struct is not None:
        input_struct = self.input_struct
      elif self.input_file:
        with open(self.input_file, 'r') as f:
          input_struct = structure.from_mmcif(f.read())
        # Clear bioassembly data, which matches the default loaded in PyMol.
        input_struct = input_struct.copy_and_update_globals(
            bioassembly_data=None
        )
        input_struct = input_struct.generate_bioassembly()
      else:
        raise ValueError('No input_file or input_struct set.')

      auth2internal, _ = get_author_chain_residue_index_map(input_struct)

      internal_named_fields['motif_str'] = renumber_motif_string(
          self.motif_str, auth2internal
      )
      internal_named_fields['sampled_motif_str'] = renumber_motif_string(
          self.sampled_motif_str,  # pyrefly: ignore[bad-argument-type]
          auth2internal,
      )
      internal_motif_atom_groups = []
      for group in self.motif_atoms.split():
        source_residue, atom_names = group.split(':', maxsplit=1)
        source_chain, source_res_id = (
            residue_mapping.separate_chain_id_from_res_ids(source_residue)
        )
        target_chain, target_res_id = auth2internal[
            (source_chain, int(source_res_id))
        ]
        internal_motif_atom_groups.append(
            f'{target_chain}{target_res_id}:{atom_names}'
        )
      internal_named_fields['motif_atoms'] = ' '.join(
          internal_motif_atom_groups
      )
      internal_named_fields['reseq_residues'] = renumber_motif_string(
          self.reseq_residues,  # pyrefly: ignore[bad-argument-type]
          auth2internal,
      )
      internal_named_fields['unindexed_motif_residues'] = renumber_motif_string(
          self.unindexed_motif_residues,  # pyrefly: ignore[bad-argument-type]
          auth2internal,
      )
      if self.ligand_mobile_atoms:
        renumbered = renumber_motif_string(
            ','.join(self.ligand_mobile_atoms), auth2internal
        ).split(',')
        internal_named_fields['ligand_mobile_atoms'] = dict(
            zip(renumbered, self.ligand_mobile_atoms.values(), strict=True)
        )

    return dataclasses.replace(
        self,
        is_author_naming=False,
        **internal_named_fields,
    )

  def with_input_struct(self) -> Self:
    """Return a copy of self with input_struct set."""
    if not self.input_file:
      return self
    with open(self.input_file, 'r') as f:
      input_struct = structure.from_mmcif(f.read()).without_hydrogen()

    # Clear bioassembly data, which matches the default loaded in PyMol.
    input_struct = input_struct.copy_and_update_globals(bioassembly_data=None)
    input_struct = input_struct.generate_bioassembly()

    return dataclasses.replace(self, input_struct=input_struct)

  def parsed(self, seed: int = 0) -> Self:
    """Returns a copy of self that can be converted to DiffusionInput."""
    if '|' in self.motif_str:
      prefix, suffix = self.motif_str.split('|')
      if len(prefix.split(',')) != suffix.count('{}'):
        raise ValueError(
            f'Motif string {self.motif_str} contains different number of'
            f' motif islands in prefix ({len(prefix.split(","))}) and suffix'
            f' ({suffix.count("{}")}).'
        )

    motif_spec = self
    if self.input_struct is None:
      motif_spec = motif_spec.with_input_struct()
    if self.is_author_naming:
      motif_spec = motif_spec.with_internal_naming()

    if self.sampled_motif_str is not None:
      logging.warning(
          'MotifSpec.sampled_motif_str will be overwritten by a newly'
          ' sampled motif string.'
      )
    rng = np.random.default_rng(seed)
    sampled_motif_str = residue_mapping.sample_designable_lengths(
        motif_spec.motif_str,
        rng,
        motif_spec.seq_length,
    )
    sampled_motif_str = (
        residue_mapping.remove_missing_residues_from_motif_string(
            sampled_motif_str,
            motif_spec.input_struct,  # pyrefly: ignore[bad-argument-type]
        )
    )
    return dataclasses.replace(
        motif_spec,
        sampled_motif_str=sampled_motif_str,
        residue_map=residue_mapping.motif_str_to_residue_map(
            motif_str=sampled_motif_str,
            input_struct=motif_spec.input_struct,
            reseq_residues=motif_spec.reseq_residues,
        ),
        is_parsed=True,
    )

  @property
  def is_unindexed(self) -> bool:
    """Returns True if the spec specifies an unindexed motif."""
    return bool(self.unindexed_motif_residues)

  @property
  def is_partial_diffusion(self) -> bool:
    """Returns True if the spec specifies a partial diffusion input structure."""
    return self.partial_diffusion_input_struct is not None

  def add_protein_ligand_features(
      self,
      input_features: dict[str, Any],
      input_struct: structure.Structure,
      *,
      max_num_res: int,
      is_partial_diffusion: bool = False,
  ) -> dict[str, Any]:
    """Adds protein and ligand features to input_features in-place.

    Note that because this function modifies the features in-place, we require
    a dict annotation for the type, instead of a Mapping.

    Args:
      input_features: The input features to add features to.
      input_struct: The input structure to add features from.
      max_num_res: The maximum number of residues to add features for.
      is_partial_diffusion: Whether the features are for partial diffusion.

    Returns:
      The input features with the added features.
    """
    if input_struct:
      input_features = add_fixed_residue_features(
          input_features,
          input_struct,
          self.residue_map,  # pyrefly: ignore[bad-argument-type]
          max_num_res,
      )
    if self.residue_map and self.residue_map.is_ligand_mask.any():
      input_features = add_ligand_features(
          input_features,
          input_struct,
          self.residue_map,
          max_num_res,
          self.ligand_mobile_atoms,
      )
    if self.motif_atoms and not self.is_unindexed:
      input_features = add_motif_atom_features(
          input_features,
          self.residue_map,  # pyrefly: ignore[bad-argument-type]
          self.motif_atoms,
          is_partial_diffusion=is_partial_diffusion,
      )
    if self.reseq_residues and not self.is_unindexed:
      input_features = add_reseq_residue_features(
          input_features,
          self.residue_map,  # pyrefly: ignore[bad-argument-type]
          self.reseq_residues,
          is_partial_diffusion=is_partial_diffusion,
      )

    return input_features

  def to_diffusion_input(
      self,
      *,
      max_num_res: int,
      max_num_ligands: int = types_lib.MAX_NUM_LIGANDS,
      max_num_res_unindexed: int = 128,
  ) -> types_lib.DiffusionInput:
    """Returns a DiffusionInput with input features for sampling."""
    if not self.is_parsed:
      raise ValueError(
          'MotifSpec must be parsed before converting to DiffusionInput.'
      )

    input_features = create_designable_protein_features(
        self.residue_map,  # pyrefly: ignore[bad-argument-type]
        max_num_res=max_num_res,
        max_num_ligands=max_num_ligands,
        max_num_res_unindexed=max_num_res_unindexed,
    )
    input_features = self.add_protein_ligand_features(
        input_features,
        self.input_struct,  # pyrefly: ignore[bad-argument-type]
        max_num_res=max_num_res,
        is_partial_diffusion=False,
    )

    if self.unindexed_motif_residues:
      unindexed_motif_residues = self.unindexed_motif_residues
    else:
      unindexed_motif_residues = (
          residue_mapping.extract_motifs_from_designable_chains(
              get_processed_motif_str(self.motif_str)
          )
      )

    if unindexed_motif_residues:  # To exclude binder design edge case.
      input_features = add_unindexed_motif_features(
          input_features=input_features,
          input_struct=self.input_struct,  # pyrefly: ignore[bad-argument-type]
          unindexed_motif_residues=unindexed_motif_residues,
          motif_atoms=self.motif_atoms or '',
          reseq_residues=self.reseq_residues or '',
          max_num_res=max_num_res_unindexed,
          fixed_atom_mask=input_features['fixed_atom_mask'],
      )

    if self.partial_diffusion_input_struct is None:
      partial_diffusion_input_features = None
    else:
      # Construct features from partial diffusion input struct.
      partial_diffusion_input_features = (
          _create_diffusion_protein_features_from_protein(
              types_lib.pad_protein(
                  types_lib.Protein.from_structure(
                      self.partial_diffusion_input_struct
                  ),
                  num_residues=max_num_res,
                  num_ligands=max_num_ligands,
              ),
              num_res_unindexed=max_num_res_unindexed,
          )
      )
      # Make new spec, in design residue numbering, with only the features that
      # are relevant for partial diffusion.
      partial_diffusion_spec = make_partial_diffusion_spec(
          self, self.partial_diffusion_input_struct
      )
      # We only want to extract positions and aatype from the partial diffusion
      # input struct. We need to do necessary feature extraction on it first.
      partial_diffusion_input_features = (
          partial_diffusion_spec.add_protein_ligand_features(
              partial_diffusion_input_features,
              self.partial_diffusion_input_struct,
              max_num_res=max_num_res,
              is_partial_diffusion=True,
          )
      )

    if partial_diffusion_input_features is not None:
      # Read the target mask before it is overwritten below; the alignment
      # needs to see it in its pre-overwrite state.
      target_atom_mask = np.asarray(input_features['protein']['all_atom_mask'])
      partial_diffusion_atom_mask = np.asarray(
          partial_diffusion_input_features['protein']['all_atom_mask']
      )
      fixed_mask = (
          np.asarray(input_features['fixed_atom_mask'])
          * target_atom_mask
          * partial_diffusion_atom_mask
      ).astype(bool)
      input_features['protein']['all_atom_positions'] = (
          _align_partial_diffusion_positions(
              source_positions=np.asarray(
                  partial_diffusion_input_features['protein'][
                      'all_atom_positions'
                  ]
              ),
              target_positions=np.asarray(
                  input_features['protein']['all_atom_positions']
              ),
              source_atom_mask=partial_diffusion_atom_mask.astype(bool),
              fixed_atom_mask=fixed_mask,
          )
      )
      # `all_atom_positions` and `aatype` now describe the partial diffusion
      # struct, so `all_atom_mask` must describe it too.
      # `create_designable_protein_features` marks all 37 atom37 slots present
      # as a placeholder for designable residues; keeping that placeholder here
      # would mark atoms as present that the partial diffusion struct does not
      # have, leaving them at the origin and corrupting reconstruction.
      input_features['protein']['aatype'] = partial_diffusion_input_features[
          'protein'
      ]['aatype']
      input_features['protein']['all_atom_mask'] = (
          partial_diffusion_input_features['protein']['all_atom_mask']
      )

    return _feature_dict_to_diffusion_input(input_features)


def _align_partial_diffusion_positions(
    *,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    source_atom_mask: np.ndarray,
    fixed_atom_mask: np.ndarray,
) -> np.ndarray:
  """Rigidly aligns partial diffusion atom positions onto target motif positions."""
  num_fixed = int(np.sum(fixed_atom_mask))
  if num_fixed == 0 or not np.any(source_atom_mask):
    return source_positions

  src_f64 = source_positions.astype(np.float64)
  tgt_f64 = target_positions.astype(np.float64)
  src_fixed = src_f64[fixed_atom_mask]
  tgt_fixed = tgt_f64[fixed_atom_mask]
  src_centroid = np.mean(src_fixed, axis=0)
  tgt_centroid = np.mean(tgt_fixed, axis=0)

  aligned = np.zeros_like(src_f64)
  if num_fixed >= 3:
    cov = (src_fixed - src_centroid).T @ (tgt_fixed - tgt_centroid)
    u, _, vt = np.linalg.svd(cov)
    det_sign = float(np.sign(np.linalg.det(u @ vt)))
    rot = u @ np.diag([1.0, 1.0, det_sign]) @ vt
    aligned[source_atom_mask] = (
        src_f64[source_atom_mask] - src_centroid
    ) @ rot + tgt_centroid
  else:
    aligned[source_atom_mask] = src_f64[source_atom_mask] + (
        tgt_centroid - src_centroid
    )
  return aligned.astype(source_positions.dtype)


def make_partial_diffusion_spec(
    spec: MotifSpec,
    partial_diffusion_input_struct: structure.Structure | None = None,
) -> MotifSpec:
  """Make a MotifSpec representing a partial diffusion input.

  This function creates a new MotifSpec that can be used to extract features
  from `spec.partial_diffusion_input_struct`. This structure contains atom
  coordinates and amino acid types that will be used to initialize the
  diffusion process at a given timestep.

  The residue numbering in `partial_diffusion_input_struct` matches the
  numbering of the *designed* protein, i.e. target residue numbering,
  therefore we need to update fields in MotifSpec to use this numbering instead
  of source structure numbering, so that we can reuse
  `add_protein_ligand_features` for feature extraction.

  Args:
    spec: An already-parsed MotifSpec with `partial_diffusion_input_struct` set.
    partial_diffusion_input_struct: The input structure to use for partial
      diffusion. If None, `spec.partial_diffusion_input_struct` is used.

  Returns:
    A new MotifSpec for featurizing `partial_diffusion_input_struct`.
  """
  if not spec.is_parsed:
    raise ValueError(
        'MotifSpec must be parsed before featurizing for partial diffusion.'
    )
  if spec.residue_map is None:
    raise ValueError('MotifSpec must have a residue map.')

  source2target = spec.residue_map.get_source_resid_to_target_resid_dict()
  new_source_chain_res_ids = [
      source2target.get(ch_res, ch_res)
      for ch_res in zip(
          spec.residue_map.source_chain_ids, spec.residue_map.source_residue_ids
      )
  ]
  new_source_chain_ids, new_source_residue_ids = list(
      zip(*new_source_chain_res_ids)
  )

  partial_diffusion_residue_map = dataclasses.replace(
      spec.residue_map,
      source_chain_ids=np.array(new_source_chain_ids),
      source_residue_ids=np.array(new_source_residue_ids),
  )
  input_struct = (
      partial_diffusion_input_struct or spec.partial_diffusion_input_struct
  )
  return MotifSpec(
      name=spec.name,
      description=spec.description,
      input_file=None,
      input_struct=input_struct,
      is_author_naming=spec.is_author_naming,
      residue_map=partial_diffusion_residue_map,
      motif_str=renumber_motif_string(spec.motif_str, source2target),
      sampled_motif_str=renumber_motif_string(
          spec.sampled_motif_str, source2target  # pyrefly: ignore[bad-argument-type]
      ),
      motif_atoms=renumber_motif_atoms(spec.motif_atoms, source2target),
      reseq_residues=renumber_motif_string(
          spec.reseq_residues, source2target  # pyrefly: ignore[bad-argument-type]
      ),
  )


def get_motif_gtstruct(
    dinput: types_lib.DiffusionInput,
) -> structure.Structure:
  """Constructs the ground-truth motif structure in target residue indexing."""
  if dinput.crop_cond_atom_positions is None or dinput.crop_cond_aatype is None:
    raise ValueError(
        'DiffusionInput must have crop_cond_atom_positions and'
        ' crop_cond_aatype to construct ground-truth motif structure.'
    )
  protein_for_gt = dataclasses.replace(
      dinput.protein,
      atom_positions=dinput.crop_cond_atom_positions,
      aatype=dinput.crop_cond_aatype,
  )
  gt_protein, _ = denoiser.parse_super_all_atom_prot(
      protein_for_gt,
      enforce_input_aatype_for_fixed_residues=True,
      fixed_aatype_mask=dinput.fixed_seq_mask,
      diffusion_input_aatype=dinput.crop_cond_aatype,
  )
  return gt_protein.to_structure(b_factors=dinput.fixed_atom_mask)
