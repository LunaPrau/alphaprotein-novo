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

"""Helper functions for the data pipeline."""

import collections
from collections.abc import Collection, Mapping, MutableMapping, Sequence
from typing import Any

from alphafold3 import structure
from alphafold3.constants import mmcif_names
from alphafold3.constants import residue_names
from alphaprotein_novo.data import data_constants
from alphaprotein_novo.data import structure_features
import numpy as np

Batch = MutableMapping[str, np.ndarray]
FeatureDict = MutableMapping[str, Any]
FeatureShapes = Mapping[str, Sequence[str | None]]


class InvalidResidueTypeError(Exception):
  """Raised when the example contains an invalid amino acid type."""


class UnsupportedChainTypeError(Exception):
  """Raised when a chain type is not supported."""


def pad_to_size(
    arr: np.ndarray,
    size: int,
    dim: int = 0,
    padding_token: Any = 0,
) -> np.ndarray:
  """Pad an array to the specified size in dim."""
  # Return if not an array or is a scalar.
  if not hasattr(arr, 'shape') or not arr.shape:
    return arr
  else:
    arr_shape = arr.shape

  if arr_shape[dim] > size:
    raise RuntimeError(
        f'Array is too large in dim {dim} to pad. {arr.shape[dim]} vs {size}'
    )
  else:
    arr_shape = list(arr.shape)
    pad_size = size - arr_shape[dim]
    arr_shape[dim] = pad_size
    pad_arr = np.full(
        shape=arr_shape,
        fill_value=padding_token,
        dtype=arr.dtype,
    )
    return np.concatenate([arr, pad_arr], axis=dim)


def _get_single_chain_structure_feats(
    struct: structure.Structure,
    include_missing_residues: bool,
    fix_nonstandard: bool = True,
) -> FeatureDict:
  """Unpack a single chain structure.Structure into feature arrays."""

  if len(struct.chains) != 1:
    raise RuntimeError('Structures unpacked must contain a single chain only.')

  chain_id = struct.chains[0]
  res_arrays = struct.to_res_arrays(
      include_missing_residues=include_missing_residues,
  )
  sequence = struct.chain_single_letter_sequence(
      include_missing_residues=include_missing_residues,
  )[chain_id]

  if not set(sequence).issubset(
      residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN
  ):
    raise InvalidResidueTypeError(
        f'Invalid residue found in sequence: {sequence}'
    )

  aatype = np.asarray([
      residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN.index(x)
      for x in sequence
  ])
  sequence_mask = np.ones_like(aatype)
  if include_missing_residues:
    residue_index = np.asarray(
        [res_id for (_, res_id) in struct.all_residues[chain_id]],
        dtype=np.int32,
    )
  else:
    residue_index = np.asarray(
        [
            res['res_id']
            # Only iterates over resolved residues.
            for res in struct.iter_residues()
            if res['chain_id'] == chain_id
        ],
        dtype=np.int32,
    )

  # CCD codes for every residue as a tuple.
  # If fix_nonstandard, non standard residues are mapped to standard res names.
  # If unknown, gives UNK for protein and N for RNA / DNA.
  component_ids = struct.chain_res_name_sequence(
      include_missing_residues=include_missing_residues,
      fix_non_standard_polymer_res=fix_nonstandard,
  )[chain_id]

  is_protein = np.ones_like(aatype, dtype=np.int32)

  # Padding for ligand features.
  ligand_atomic_number = np.zeros_like(aatype, dtype=int)
  ligand_charge = np.zeros_like(aatype, dtype=int)
  # -1 in unused for hybridization constants: see bond_constants.py.
  ligand_hybridization = np.ones_like(aatype, dtype=int) * -1
  ligand_atom_names = np.full(
      shape=(len(aatype),),
      fill_value='',
      dtype=object,
  )

  features_dict = {
      'chain_id': np.array([chain_id] * len(sequence), dtype=object),
      'component_ids': np.array(component_ids, dtype=object),
      'all_atom_positions': res_arrays.atom_positions.astype(float),
      'all_atom_mask': res_arrays.atom_mask.astype(int),
      'b_factors': res_arrays.atom_b_factor.astype(float),
      'residue_index': residue_index.astype(int),
      'aatype': aatype.astype(int),
      'seq_mask': sequence_mask.astype(int),
      'seq_length': len(sequence),
      'is_protein': is_protein.astype(int),
      'ligand_atomic_number': ligand_atomic_number,
      'ligand_charge': ligand_charge,
      'ligand_hybridization': ligand_hybridization,
      'ligand_atom_names': ligand_atom_names,
  }

  return features_dict


def get_global_features(struct: structure.Structure):
  return {'resolution': struct.resolution or 0.0}


def add_symmetry_group_identifiers(
    polymers: list[MutableMapping[str, np.ndarray]],
    ligands: list[MutableMapping[str, np.ndarray]],
) -> tuple[
    list[MutableMapping[str, np.ndarray]], list[MutableMapping[str, np.ndarray]]
]:
  """Add symmetry group identifiers to chains."""
  seq_to_entity_id = {}
  grouped = {
      'polymers': collections.defaultdict(list),
      'ligands': collections.defaultdict(list),
  }

  chain_id = 1
  for chain in polymers:
    chain['chain_counter'] = chain_id  # pyrefly: ignore[unsupported-operation]
    seq = ','.join(chain['component_ids'].tolist())
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped['polymers'][seq_to_entity_id[seq]].append(chain.copy())  # pyrefly: ignore[missing-attribute]
    chain_id += 1
  for chain in ligands:
    chain['chain_counter'] = chain_id  # pyrefly: ignore[unsupported-operation]
    seq = chain['component_id'].item()
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped['ligands'][seq_to_entity_id[seq]].append(chain.copy())  # pyrefly: ignore[missing-attribute]
    chain_id += 1

  new_polymers = []
  for entity_id, chains in grouped['polymers'].items():
    for sym_id, chain in enumerate(chains, start=1):
      seq_length = len(chain['component_ids'])
      chain['asym_id'] = chain['chain_counter'] * np.ones(
          seq_length, dtype=np.int32
      )
      chain['sym_id'] = sym_id * np.ones(seq_length)
      chain['entity_id'] = entity_id * np.ones(seq_length, dtype=np.int32)
      new_polymers.append(chain)

  new_ligands = []
  for entity_id, chains in grouped['ligands'].items():
    for sym_id, chain in enumerate(chains, start=1):
      chain['component_asym_id'] = np.array(
          chain['chain_counter'], dtype=np.int32
      )
      chain['component_sym_id'] = np.array(sym_id)
      chain['component_entity_id'] = np.array(entity_id, dtype=np.int32)
      new_ligands.append(chain)

  # Sort chains by chain_id.
  new_polymers.sort(key=lambda x: x['chain_counter'])
  new_ligands.sort(key=lambda x: x['chain_counter'])

  # Remove temporary `chain_counter` field.
  for polymer in new_polymers:
    del polymer['chain_counter']
  for ligand in new_ligands:
    del ligand['chain_counter']

  return new_polymers, new_ligands


def extract_chain_features(
    struct: structure.Structure,
    *,
    include_missing_residues: bool,
    ligands_to_skip: Collection[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Extract features for polymer and ligand chains.

  Args:
    struct: A Structure to extract from.
    include_missing_residues: Whether to include unresolved residues (True) or
      drop them (False).
    ligands_to_skip: tuple of CCD codes for which, if any residue of a chain has
      this code, the entire chain is skipped.

  Returns:
    A tuple of polymer and ligand feature chains.

  Raises:
    UnsupportedChainTypeError: If the Structure contains an unknown chain type.
  """
  polymer_chains = {}
  ligand_chains = {}
  ligands_to_skip = set(ligands_to_skip or ())
  for chain_struct in struct.split_by_chain():
    chain_type = chain_struct.chains_table.type
    assert len(chain_type) == 1
    if (chain_type == mmcif_names.PROTEIN_CHAIN).all():
      polymer_chains[chain_struct.chain_id[0]] = (
          _get_single_chain_structure_feats(
              struct=chain_struct,
              include_missing_residues=include_missing_residues,
          )
      )
    elif (chain_type == mmcif_names.NON_POLYMER_CHAIN).all():
      if any(r in ligands_to_skip for r in chain_struct.res_name):
        continue
      ligand_chains[
          chain_struct.chain_id[0]
      ] = structure_features.get_ligand_features(
          chain_struct,  # pyrefly: ignore[bad-argument-type]
      )
    else:
      raise UnsupportedChainTypeError(
          f'Chain type {chain_type[0]} not supported.'
      )
  return polymer_chains, ligand_chains


def chain_dict_to_list(
    chain_dict: Mapping[str, FeatureDict],
) -> list[FeatureDict]:
  return [
      feats
      for _, feats in sorted(
          [(chain_id, f) for chain_id, f in chain_dict.items()]
      )
  ]


def make_fixed_size(
    *,
    np_example: Batch,
    shape_schema: FeatureShapes,
    num_res: int,
    num_res_unindexed: int,
) -> Batch:
  """Guess at the sequence dimensions to make fixed size."""

  pad_size_map = {
      data_constants.NUM_RES: num_res,
      data_constants.NUM_RES_UNINDEXED: num_res_unindexed,
  }

  features_to_skip = (
      data_constants.CHAIN_FEATURES + data_constants.HOST_ONLY_FEATURES
  )
  for k, v in np_example.items():
    if k in features_to_skip:
      continue
    if k not in shape_schema:
      continue
    shape = v.shape
    schema = shape_schema[k]
    if len(shape) != len(schema):
      raise ValueError(
          f'{k}: Rank mismatch between shape and '
          f'shape schema: {shape} vs {schema}'
      )

    pad_size = (
        pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)  # pyrefly: ignore[no-matching-overload]
    )
    padding = []
    for i, p in enumerate(pad_size):
      if p - v.shape[i] < 0:
        raise ValueError(
            f'{k}: Negative padding {p - v.shape[i]} '
            f'for seq length {v.shape[i]}'
        )
      padding.append((0, p - v.shape[i]))
    if padding:
      constant_values = 0
      if v.dtype == object or v.dtype.kind in ('U', 'S'):
        constant_values = ''
      np_example[k] = np.pad(
          v,
          padding,
          mode='constant',
          constant_values=constant_values,
      )

  return np_example


def populate_required_fields(
    examples: Sequence[FeatureDict],
    source_file_name: str,
) -> list[FeatureDict]:
  return [
      dict(source_file_name=source_file_name, domain_name='none', **example)
      for example in examples
  ]


def merge_chains(
    polymers: list[FeatureDict],
    ligands: list[FeatureDict],
) -> FeatureDict:
  """Merges polymer and ligand features into a single feature dictionary."""
  feature_names = set()
  if polymers:
    feature_names = feature_names.union(polymers[0].keys())
  if ligands:
    feature_names = feature_names.union(ligands[0].keys())

  all_chains = polymers + ligands

  merged_features = {}
  for feature_name in feature_names:
    features = [c[feature_name] for c in all_chains if feature_name in c]
    if feature_name in data_constants.SEQ_FEATURES:
      merged_features[feature_name] = np.concatenate(features, axis=0)
    elif feature_name in data_constants.CHAIN_FEATURES:
      merged_features[feature_name] = sum(x for x in features)
    else:
      merged_features[feature_name] = features[0]
  return merged_features


def make_seq_mask(np_example: Batch) -> Batch:
  if 'entity_id' in np_example:
    np_example['seq_mask'] = (np_example['entity_id'] > 0).astype(np.int32)
  else:
    np_example['seq_mask'] = np.ones(np_example['aatype'].shape, dtype=np.int32)
  return np_example


def set_ligand_aatype(
    features: Batch,
    ligand_aatype: dict[int, int],
) -> Batch:
  """Sets the aatype for ligand atoms according to their atomic number."""
  lig_aatype = [
      ligand_aatype.get(a, 0) for a in features['ligand_atomic_number']
  ]
  features['aatype'] = np.where(
      features['is_ligand_mask'],
      lig_aatype,
      features['aatype'],
  )
  return features
