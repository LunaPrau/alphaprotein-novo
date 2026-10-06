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

"""Methods for constructing Structure instances and features."""

from collections.abc import Mapping, MutableMapping, Sequence, Set
import dataclasses
import logging
import re
from typing import Any, Final, cast

from alphafold3 import structure
from alphafold3.constants import mmcif_names
from alphafold3.cpp import membership
from alphafold3.cpp import string_array
from alphafold3.structure import mmcif
from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.constants import ligand_constants
from alphaprotein_novo.constants import periodic_table
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.constants import terms_of_use
from alphaprotein_novo.data import data_constants
from alphaprotein_novo.data import ligand_data
from alphaprotein_novo.data import pipeline_utils
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial


class TooManyAtomsForLigandResidueError(ValueError):
  """Too many atoms in ligand residue."""


@dataclasses.dataclass(frozen=True, slots=True)
class LigandStructureMetadata:
  """Metadata for ligand atoms in a structure.

  The arrays are ragged. Each outer index corresponds to a ligand molecule.
  Outer lists must match length of True values in is_ligand_mask.

  Attributes:
    is_ligand_mask: A boolean array, shape (num_res,) indicating if each residue
      is a ligand or not.
    ligand_atom_names: A ragged list of lists of atom names. The outer list runs
      over the ordinal of the ligands in the dense (num_res,) arrays, ignoring
      protein residues. The inner lists run over the atom count for each ligand
      residue. The lists of atoms names can not be longer than
      `atom_types.ATOM37_NUM`.
    ligand_atom_elements: A list of lists of element strings. Indexing is the
      same as ligand_atom_names, values defined in constants/atom_types.py.
    ligand_resnames: A list of residue names, one for each _ligand_ residue.
    ligand_smiles_descriptors: A list of optional SMILES descriptors, one for
      each _ligand_ residue.
  """

  is_ligand_mask: np.ndarray
  ligand_atom_names: list[list[str]]
  ligand_atom_elements: list[list[str]]
  ligand_resnames: list[str]
  ligand_smiles_descriptors: list[str | None]

  def __post_init__(self):
    if not (
        self.is_ligand_mask.sum()
        == len(self.ligand_resnames)
        == len(self.ligand_atom_names)
        == len(self.ligand_atom_elements)
        == len(self.ligand_smiles_descriptors)
    ):
      raise ValueError(
          "Fields don't agree on the number of ligand residues!"
          f' Mask: {self.is_ligand_mask.sum()}, '
          f' Resnames: {len(self.ligand_resnames)}'
          f' Atom names: {len(self.ligand_atom_names)}'
          f' Atom elements: {len(self.ligand_atom_elements)}'
          f' SMILES descriptors: {len(self.ligand_smiles_descriptors)}'
      )
    for ires in range(self.is_ligand_mask.sum()):
      num_names = len(self.ligand_atom_names[ires])
      num_elements = len(self.ligand_atom_elements[ires])
      if num_names != num_elements:
        raise ValueError(
            f'The number of atom names ({num_names})'
            f" and atom elements ({num_elements}) don't match"
            f' for residue {ires} ({self.ligand_resnames[ires]})'
        )
      if num_names > atom_types.ATOM37_NUM:
        raise TooManyAtomsForLigandResidueError(
            f'Too many atoms ({num_names}) for residue {ires}'
            f' ({self.ligand_resnames[ires]})'
            f' the maximum is {atom_types.ATOM37_NUM}'
        )


def _insert_ligand_renames(
    res_names: Sequence[str],
    ligand_structure_data: LigandStructureMetadata,
) -> Sequence[str]:
  """Inserts three letter CCD codes into a list of resnames."""
  if len(ligand_structure_data.is_ligand_mask) != len(res_names):
    raise ValueError(
        f'{len(ligand_structure_data.is_ligand_mask)} != {len(res_names)}'
    )

  res_names_with_ligands = []
  ligand_index = 0
  for ires, res_name in enumerate(res_names):
    if ligand_structure_data.is_ligand_mask[ires]:
      ligand_name = ligand_structure_data.ligand_resnames[ligand_index]
      res_names_with_ligands.append(ligand_name)
      ligand_index += 1
    else:
      res_names_with_ligands.append(res_name)
  return res_names_with_ligands


def _populate_chemical_components_data(
    ligand_structure_data: LigandStructureMetadata,
) -> structure.ChemicalComponentsData:
  """Populates ChemicalComponentsData from LigandStructureMetadata."""
  chem_comp = {}
  for component_id, smiles_descriptor in zip(
      ligand_structure_data.ligand_resnames,
      ligand_structure_data.ligand_smiles_descriptors,
      strict=True,
  ):
    if not smiles_descriptor:
      continue
    chem_comp_entry = structure.ChemCompEntry(
        type='non-polymer',
        pdbx_smiles=smiles_descriptor,
    )
    if component_id not in chem_comp:
      chem_comp[component_id] = chem_comp_entry
    elif chem_comp[component_id] != chem_comp_entry:
      raise ValueError(
          f'Mismatching data for ligand {component_id}: '
          f'{chem_comp_entry} != {chem_comp[component_id]}'
      )
  return structure.ChemicalComponentsData(chem_comp=chem_comp)


def make_chain_type_and_atom_name(
    restypes: Sequence[str],
    ligand_structure_data: LigandStructureMetadata | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds chain_type and atom_name fields based on a sequence of restypes."""
  protein_atom_elements = np.array(
      [atom_type[:1] for atom_type in atom_types.ATOM37], dtype=object
  )
  protein_restypes = set(residue_names.PROTEIN_TYPES_WITH_UNKNOWN)

  chain_type = []
  atom_name = []
  atom_element = []
  ligand_index = 0
  for resindex, restype in enumerate(restypes):
    if (
        ligand_structure_data is not None
        and ligand_structure_data.is_ligand_mask[resindex]
    ):
      chain_type.append(mmcif_names.NON_POLYMER_CHAIN)
      num_atoms_for_ligand_res = len(
          ligand_structure_data.ligand_atom_names[ligand_index]
      )
      ligand_pad_len = atom_types.ATOM37_NUM - num_atoms_for_ligand_res

      if ligand_pad_len < 0:
        raise ValueError(
            'Residue for ligand '
            f'{ligand_structure_data.ligand_resnames[ligand_index]} '
            f'has too many atoms ({num_atoms_for_ligand_res})!'
        )

      atom_name.append(
          ligand_structure_data.ligand_atom_names[ligand_index]
          + [''] * ligand_pad_len
      )
      atom_element.append(
          ligand_structure_data.ligand_atom_elements[ligand_index]
          + [''] * ligand_pad_len
      )
      ligand_index += 1
    elif restype in protein_restypes:
      chain_type.append(mmcif_names.PROTEIN_CHAIN)
      atom_name.append(atom_types.ATOM37)
      atom_element.append(protein_atom_elements)
    else:
      raise ValueError(f'Unsupported restype: {restype}')

  chain_type = np.array(chain_type, dtype=object)
  atom_name = np.stack(atom_name, axis=0, dtype=object)
  atom_element = np.stack(atom_element, axis=0)
  return chain_type, atom_name, atom_element


def from_index_arrays(
    name: str,
    sequence_mask: np.ndarray,
    chain_index: np.ndarray,
    aatype: np.ndarray,
    residue_index: np.ndarray,
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    b_factors: np.ndarray,
    occupancies: np.ndarray | None = None,
    ligand_structure_data: LigandStructureMetadata | None = None,
    aatype_vocab: tuple[str, ...] = residue_names.POLYMER_TYPES_WITH_UNKNOWN,
) -> structure.Structure:
  """Builds a structure from arrays of indices (e.g. a model output batch).

  Index arrays in this context mean that all input arrays are number arrays. All
  string arrays like chain names, residue names, or atom names are inferred from
  positions or integer encodings within those arrays.

  Args:
    name: The name of the structure. E.g. a PDB ID.
    sequence_mask: A sequence mask with shape (num_res,). This is typically the
      mask used when handling ragged batches.
    chain_index: An integer array with shape (num_res,). The lowest (unmasked)
      value will be assigned chain ID "A".
    aatype: An integer array with shape (num_res,). The integer values
      correspond to what specified in the aatype_vocab.
    residue_index: The integer residue IDs with shape (num_res,).
    atom_positions: A float array of atom positions with shape (..., num_res,
      num_atom, 3). The num_atom dimension corresponds to the atoms present in a
      given residue type as described in constants/atom_types.py. Leading
      dimensions are interpreted as extra models by Structure.
    atom_mask: A (num_res, num_atom) shaped mask. This should mask out atoms
      that aren't present in the structure, either because they aren't used in
      the given residue, or because they were not resolved/predicted.
    b_factors: A float array of shape (num_res, num_atom) representing the
      B-factor (or other arbitrary value e.g. pLDDT) for each atom.
    occupancies: A float array with values [0, 1] of shape (num_res, num_atom)
      representing the occupancy for each atom.
    ligand_structure_data: An optional instance of LigandStructureMetadata that
      will be used to create ligands.
    aatype_vocab: A tuple of residue types to decode the aatype from integers
      into residue strings.

  Returns:
    A Structure instance that represents the same structure as the input arrays.
  """
  res_indices = np.where(sequence_mask)[0]

  if not res_indices.size:
    return structure.from_res_arrays(
        atom_mask=np.zeros((0, atom_types.ATOM37_NUM), dtype=bool),
        name=name,
    )

  chain_id_arr = chain_index[res_indices].astype(np.int32)
  chain_id = np.vectorize(mmcif.int_id_to_str_id, otypes=[object])(
      chain_id_arr + (1 - chain_id_arr.min())
  )

  aatype = aatype[res_indices]
  restypes = [aatype_vocab[res] for res in aatype]

  chemical_components_data = None
  if ligand_structure_data is not None:
    restypes = _insert_ligand_renames(restypes, ligand_structure_data)
    chemical_components_data = _populate_chemical_components_data(
        ligand_structure_data
    )

  res_name = np.array(restypes, dtype=object)

  chain_type, atom_name, atom_element = make_chain_type_and_atom_name(
      restypes, ligand_structure_data=ligand_structure_data
  )

  # Normal atom_positions have three dimensions, find out if we have extra.
  num_leading = len(atom_positions.shape) - 3
  atom_coords = np.take(atom_positions, res_indices, axis=num_leading)
  res_id = residue_index[res_indices]

  from_res_arrays_kwargs = dict(
      name=name,
      atom_mask=atom_mask[res_indices],
      chain_id=chain_id,
      chain_type=chain_type,
      res_name=res_name,
      res_id=res_id,
      atom_name=atom_name,
      atom_element=atom_element,
      atom_x=atom_coords[..., 0],
      atom_y=atom_coords[..., 1],
      atom_z=atom_coords[..., 2],
      atom_b_factor=np.take(b_factors, res_indices, axis=num_leading),
      chemical_components_data=chemical_components_data,
  )
  if occupancies is not None:
    from_res_arrays_kwargs['atom_occupancy'] = np.take(
        occupancies, res_indices, axis=num_leading
    )

  return structure.from_res_arrays(**from_res_arrays_kwargs)


def _get_ligand_structure_metadata(
    batch: Mapping[str, Any],
    residues_to_keep: np.ndarray,
) -> LigandStructureMetadata:
  """Creates LigandStructureMetadata for a batch."""
  element_by_atomic_num = {
      v: k.upper() for k, v in periodic_table.ATOMIC_NUMBER.items()
  }
  element_by_atomic_num[1] = 'H'

  entity_ids = batch['entity_id']
  residue_index = batch['residue_index']
  is_ligand_mask = batch['is_ligand_mask']

  component_ids = batch['component_id']
  component_entity_ids = batch['component_entity_id']
  component_smiles_descriptors = batch['component_smiles_descriptor']

  component_atom_names = batch['component_atom_names']
  component_fragment_indices = batch['component_fragment_indices']
  component_atomic_numbers = batch['component_atomic_numbers']
  component_padding_mask = batch['component_padding_mask_1d']

  atom_names = []
  res_names = []
  element_names = []
  smiles_descriptors = []
  for ires, (is_ligand, per_chain_res_idx, entity_id) in enumerate(
      zip(is_ligand_mask, residue_index, entity_ids)
  ):
    if is_ligand and ires in residues_to_keep:
      ligand_index = np.where(component_entity_ids == entity_id)[0][0]
      keep_mask = (
          component_fragment_indices[ligand_index] == per_chain_res_idx
      ) & (component_padding_mask[ligand_index].astype(bool))
      atomic_numbers = component_atomic_numbers[ligand_index][
          keep_mask
      ].tolist()
      atoms_for_res = component_atom_names[ligand_index][keep_mask].tolist()
      atom_names.append(atoms_for_res)
      element_names.append([element_by_atomic_num[x] for x in atomic_numbers])
      res_names.append(component_ids[ligand_index])
      smiles_descriptors.append(
          component_smiles_descriptors[ligand_index] or None
      )

  return LigandStructureMetadata(
      is_ligand_mask=is_ligand_mask[residues_to_keep],
      ligand_atom_names=atom_names,
      ligand_resnames=res_names,
      ligand_atom_elements=element_names,
      ligand_smiles_descriptors=smiles_descriptors,
  )


def structure_from_features(
    features: MutableMapping[str, Any],
    name: str = 'DefaultName',
    b_factors: np.ndarray | jax.Array | None = None,
) -> structure.Structure:
  """Constructs a Structure object from a feature dictionary."""
  if isinstance(b_factors, jax.Array):
    b_factors = np.array(b_factors)

  required_keys = {
      'seq_mask',
      'asym_id',
      'aatype',
      'residue_index',
      'all_atom_positions',
      'all_atom_mask',
  }
  required_ligand_keys = {
      'is_ligand_mask',
      'entity_id',
      'component_id',
      'component_entity_id',
      'component_atom_names',
      'component_fragment_indices',
      'component_atomic_numbers',
      'component_padding_mask_1d',
  }

  if missing_features := required_keys - set(features):
    raise ValueError(
        f'Minimal required features not provided. {missing_features=}.'
    )
  if features['aatype'].ndim != 1:
    raise ValueError('Requiring unbatched input.')

  if 'is_ligand_mask' in features and features['is_ligand_mask'].sum() > 0:
    if missing_ligand_features := required_ligand_keys - set(features):
      raise ValueError(
          f'Required ligand features not provided. {missing_ligand_features=}.'
      )

    # Apply PDB residue counting convention that all ligand atoms are counted
    # as a single residue with index 1.
    residue_index = np.copy(features['residue_index'])
    residue_index[features['is_ligand_mask'] == 1] = 1

    # Add placeholder entries for features that are required by the conversion
    # function, but not to specify a Structure.
    num_ligands, _ = features['component_atomic_numbers'].shape
    if 'component_smiles_descriptor' not in features:
      features['component_smiles_descriptor'] = np.asarray(
          [''] * num_ligands,
          dtype=object,
      )

    # Construct ligand meta data.
    res_indices = np.where(features['seq_mask'])[0]
    ligand_structure_data = _get_ligand_structure_metadata(
        batch=features,
        residues_to_keep=res_indices,
    )

    # Filter to protein aatypes.
    aatype = np.where(
        features['is_ligand_mask'],
        residue_names.PROTEIN_TYPES_WITH_UNKNOWN.index('UNK'),
        features['aatype'],
    )
  else:
    residue_index = features['residue_index']
    aatype = features['aatype']
    ligand_structure_data = None

  if b_factors is None:
    b_factors = features['all_atom_mask']
  if b_factors.shape != features['all_atom_mask'].shape:
    raise ValueError(
        'b_factors and all_atom_mask must have the same shape, but got:'
        f' {b_factors.shape} and {features["all_atom_mask"].shape}.'
    )

  struct = from_index_arrays(
      name=name,
      sequence_mask=features['seq_mask'],
      chain_index=features['asym_id'].astype(np.int32),
      aatype=aatype.astype(int),
      residue_index=residue_index,
      atom_positions=features['all_atom_positions'],
      atom_mask=features['all_atom_mask'],
      b_factors=b_factors.astype(np.float32),
      ligand_structure_data=ligand_structure_data,
  )

  # Remove duplicates in the residues table. This is relevant for ligands since
  # each atom is at this point encoded by a single residue with residue index 1.
  old_res = struct.residues_table

  unique_residues = {}
  res_key_mapping = {}
  deduplicated_mask = np.zeros(len(old_res), dtype=bool)
  for i, res in enumerate(zip(old_res.chain_key, old_res.id, old_res.name)):
    res_key = old_res.key[i]
    if res not in unique_residues:
      deduplicated_mask[i] = True
      unique_residues[res] = res_key
      res_key_mapping[res_key] = res_key
    else:
      res_key_mapping[res_key] = unique_residues[res]

  # Mask out duplicate residues within each chain.
  new_residues = old_res.apply_array(deduplicated_mask)
  # Remap atoms to belong only in the deduplicated residues.
  new_atoms = struct.atoms_table.copy_and_remap(res_key=res_key_mapping)
  return struct.copy_and_update(residues=new_residues, atoms=new_atoms)


def _create_shape_dtype_spec_dict(
    shape_dict: Mapping[str, tuple[int, ...]],
    dtype_dict: Mapping[str, np.typing.DTypeLike],
) -> dict[str, ligand_data.ShapeDtypeSpec]:
  """Creates a dictionary of `ShapeDtypeSpec` objects from shapes and dtypes."""
  common_keys = set(shape_dict) & set(dtype_dict)
  return {
      key: ligand_data.ShapeDtypeSpec(shape_dict[key], dtype_dict[key])  # pyrefly: ignore[bad-argument-type]
      for key in common_keys
  }


def _get_concrete_shape_spec(
    shape_schema: Mapping[str, list[str | None]],
    num_res: int | None = None,
    num_atoms: int = atom_types.ATOM37_NUM,
    spatial_dim: int = 3,
    num_ligands: int | None = None,
    num_ligand_atoms: int | None = None,
    num_res_unindexed: int | None = None,
) -> dict[str, tuple[int, ...]]:
  """Populates a concrete shape specification for a given shape schema."""
  shape_map = {
      data_constants.NUM_RES: num_res,
      data_constants.NUM_ATOMS: num_atoms,
      data_constants.SPATIAL_DIM: spatial_dim,
      data_constants.NUM_LIGANDS: num_ligands,
      data_constants.NUM_LIGAND_ATOMS: num_ligand_atoms,
      data_constants.NUM_RES_UNINDEXED: num_res_unindexed,
  }
  provided_dims = set(k for k, v in shape_map.items() if v is not None)
  concrete_shape_spec = {}
  for feature_name, shape_descriptor in shape_schema.items():
    if set(shape_descriptor).issubset(provided_dims):
      concrete_shape_spec[feature_name] = tuple(
          [shape_map[s] for s in shape_descriptor]  # pyrefly: ignore[bad-index]
      )
  return cast(dict[str, tuple[int, ...]], concrete_shape_spec)


def _stack_tree_list(trees: Sequence[Any]) -> Any:
  """Stack a list of trees into a single tree with stacked leading dimension."""

  def stack_if_not_nones_else_none(*inputs):
    non_none_inputs = [x for x in inputs if x is not None]
    if not non_none_inputs:
      return None
    elif len(non_none_inputs) == len(inputs):
      if isinstance(non_none_inputs[0], jax.Array):
        return jnp.stack(inputs)
      return np.stack(inputs)
    else:
      raise ValueError(
          'Tree leafs contained an inconsistent mix of arrays and Nones'
      )

  if not trees:
    raise ValueError('Must provide at least one structure to stack_tree_list.')

  return jax.tree.map(stack_if_not_nones_else_none, *trees)


def features_from_structure(
    struct: structure.Structure,
    num_residues: int,
    num_ligands: int,
    num_ligand_atoms: int = 140,
    num_ligand_atom_types: int = ligand_constants.NUM_LIGAND_ELEMS,
    num_prot_aatypes: int = ligand_constants.NUM_PROT_AATYPES,
    include_missing_residues: bool = True,
    ligands_to_skip: tuple[str, ...] = ('',),
    num_res_unindexed: int = 128,
) -> dict[str, Any]:
  """Generates a feature dictionary from a Structure."""
  struct_num_ligands = int(
      np.sum(struct.chains_table.type == mmcif_names.NON_POLYMER_CHAIN)
  )
  if num_ligands < struct_num_ligands:
    raise ValueError(
        f'Number of ligands in structure ({struct_num_ligands}) is larger than'
        f' available array size for ligands ({num_ligands}).'
    )

  ligand_aatype = {
      k: v + num_prot_aatypes
      for k, v in ligand_constants.ATOMIC_NUMBER_TO_IDX.items()
      if v < num_ligand_atom_types
  }

  polymer_chains, ligand_chains = pipeline_utils.extract_chain_features(
      struct,  # pyrefly: ignore[bad-argument-type]
      include_missing_residues=include_missing_residues,
      ligands_to_skip=ligands_to_skip,
  )

  polymer_chains = pipeline_utils.chain_dict_to_list(polymer_chains)
  ligand_chains = pipeline_utils.chain_dict_to_list(ligand_chains)

  polymer_chains = pipeline_utils.populate_required_fields(
      polymer_chains,
      source_file_name=struct.name,
  )

  (
      polymer_chains,
      ligand_chains,
  ) = pipeline_utils.add_symmetry_group_identifiers(
      polymer_chains,
      ligand_chains,
  )

  feature_shapes = data_constants.FEATURE_SIZES_TPU_SUPPORTED
  feature_shapes_concrete = _get_concrete_shape_spec(
      feature_shapes,  # pyrefly: ignore[bad-argument-type]
      num_res=num_residues,
      num_ligands=num_ligands,
      num_ligand_atoms=num_ligand_atoms,
      num_res_unindexed=num_res_unindexed,
  )
  feature_dtypes = data_constants.FEATURE_DTYPES
  shape_dtype_spec_dict = _create_shape_dtype_spec_dict(
      shape_dict=feature_shapes_concrete,
      dtype_dict=feature_dtypes,
  )

  ligand_chains = ligand_data.add_chain_representation_to_ligands(
      ligand_chains,
      protein_array_spec_dict=shape_dtype_spec_dict,
  )

  polymer_chains, ligand_chains = ligand_data.add_is_ligand_mask(
      polymer_chains,
      ligand_chains,
  )

  features = pipeline_utils.merge_chains(
      cast(list[Any], polymer_chains),
      [l['chain'] for l in ligand_chains],
  )

  features = pipeline_utils.make_fixed_size(
      np_example=features,
      shape_schema=feature_shapes,
      num_res=num_residues,
      num_res_unindexed=num_res_unindexed,
  )

  features = pipeline_utils.make_seq_mask(features)

  if num_ligands > 0:
    padded_ligand_features = ligand_data.pad_ligand_features(
        ligands=ligand_chains,
        feature_shapes=feature_shapes,
        num_ligands=num_ligands,
        num_atoms_per_ligand=num_ligand_atoms,
    )
    features.update(_stack_tree_list(padded_ligand_features))
    # Update aatype for ligands after merging chains.
    features = pipeline_utils.set_ligand_aatype(features, ligand_aatype)

  global_features = pipeline_utils.get_global_features(
      struct  # pyrefly: ignore[bad-argument-type]
  )
  features.update(global_features)

  features = data_constants.filter_features(features)

  features = cast(dict[str, Any], features)
  return features


def _extract_ligand_component_ids(
    struct: structure.Structure,
) -> Sequence[str]:
  """Extracts ligand component ids, i.e., ligand CCD codes from a structure."""
  ligand_part = struct.filter_to_entity_type(ligand=True)
  return [x['res_name'] for x in ligand_part.iter_residues()]


def count_all_residues(
    struct: structure.Structure,
    use_ccd_for_ligands: bool = False,
) -> int:
  """Counts all residues in the Structure."""
  chain_types = set(struct.chains_table.type)
  implemented_chain_types = {
      mmcif_names.PROTEIN_CHAIN,
      mmcif_names.NON_POLYMER_CHAIN,
  }
  if not chain_types <= implemented_chain_types:
    raise ValueError(
        f'The only implemented chain types are {implemented_chain_types}, but'
        f' structure has chain types {chain_types}.'
    )

  protein_struct = struct.filter_to_entity_type(protein=True)
  ligand_struct = struct.filter_to_entity_type(ligand=True)

  protein_residues = protein_struct.num_residues(count_unresolved=True)

  if use_ccd_for_ligands:
    ligand_residues = 0
    for ccd_code in _extract_ligand_component_ids(ligand_struct):
      # Fall back to heavy atom names in ligand_struct when ccd_code is custom
      # ligand (e.g. 'pd00') and chemical_components_data lacks the entry.
      comp_atoms = list(
          ligand_struct.without_hydrogen().filter(res_name=ccd_code).atom_name
      )
      num_atoms = len(
          ligand_data.get_ligand_ccd_atom_names(
              ccd_code,
              chemical_components_data=ligand_struct.chemical_components_data,  # pyrefly: ignore[bad-argument-type]
              default_atom_names=comp_atoms or None,
          )
      )
      ligand_residues += num_atoms
  else:
    ligand_residues = ligand_struct.num_atoms

  return protein_residues + ligand_residues


def _leading_dim(arr: np.ndarray) -> np.ndarray:
  """Accesses leading dimension of an array."""
  return arr if arr.ndim == 1 else arr[0]


def _format_coord(coord: float) -> str:
  """Formats a coordinate float into exactly 8 characters for PDB columns.

  PDB standard specifies 8-character fields for Cartn_x, Cartn_y, Cartn_z
  (columns 31-38, 39-46, 47-54). Standard %8.3f formatting overflows to 9+
  characters when coord <= -1000.0 or >= 10000.0 (e.g. from unconstrained or
  random parameter sampling), which shifts subsequent columns and breaks PDB
  parsers like ProDy. This function adaptively reduces decimal precision to
  guarantee the string is exactly 8 characters long while preserving maximum
  precision and parser compatibility.

  Args:
    coord: Coordinate value in Angstroms.

  Returns:
    An 8-character formatted string for the coordinate column.
  """
  if not np.isfinite(coord):
    raise ValueError(
        f'Cannot format non-finite coordinate {coord} into PDB record.'
    )
  for fmt in ('8.3f', '8.2f', '8.1f', '8.0f'):
    s = f'{coord:{fmt}}'
    if len(s) == 8:
      return s
  clamped = max(-999999.0, min(9999999.0, float(coord)))
  logging.warning(
      'Clamping out-of-range coordinate %f to %f for PDB coordinate column.',
      coord,
      clamped,
  )
  s = f'{clamped:8.0f}'
  return s[:8]


def _format_float(val: float, width: int, precision: int = 2) -> str:
  """Formats a float into exactly `width` characters.

  Args:
    val: Float value to format.
    width: Target field width in characters.
    precision: Initial decimal precision to attempt.

  Returns:
    A formatted string of length `width`.
  """
  if not np.isfinite(val):
    raise ValueError(f'Cannot format non-finite float {val} into PDB record.')
  for p in range(precision, -1, -1):
    s = f'{val:{width}.{p}f}'
    if len(s) == width:
      return s
  max_val = 10 ** (width - 1) - 1
  min_val = -(10 ** (width - 2) - 1)
  clamped = max(min_val, min(max_val, float(val)))
  logging.warning(
      'Clamping out-of-range value %f to %f for PDB %d-character field.',
      val,
      clamped,
      width,
  )
  s = f'{clamped:{width}.0f}'
  return s[:width]


def add_license_and_terms_of_use_header(content: str) -> str:
  """Adds the license and terms of use header at the top of an output file.

  This prepends a `#` comment block, which is valid in mmCIF (where `#` is the
  CIF/STAR comment character) and ignored by PDB readers. It must NOT be used
  for FASTA, whose readers require the file to begin with `>`; use
  `format_fasta_with_terms_of_use` for FASTA output instead.

  Args:
    content: The mmCIF or PDB file content to stamp.

  Returns:
    `content` with the notice prepended as a comment block.
  """
  return f'{terms_of_use.OUTPUT_FILE_NOTICE}\n{content}'


def format_fasta_with_terms_of_use(description: str, sequence: str) -> str:
  """Builds a single-record FASTA file carrying the terms of use notice.

  The notice is placed in the description field of the header line rather than
  in a leading comment block, because most FASTA readers (Biopython's default
  `fasta` parser, biotite, pyfastx, and htslib's `faidx` indexer) reject a file
  that does not begin with `>`. Readers take the record ID to be the text up to
  the first whitespace, so the ID remains `description` and the notice is
  carried as free text after it.

  Args:
    description: The record identifier, written immediately after `>`. Must not
      contain whitespace, which would otherwise truncate the ID that readers
      report.
    sequence: The single-letter amino acid sequence.

  Returns:
    The contents of a one-record FASTA file.
  """
  return (
      f'>{description} {terms_of_use.OUTPUT_FASTA_HEADER_NOTICE}\n{sequence}\n'
  )


def save_structure_to_pdb(
    struct: structure.Structure,
    filepath: epath.PathLike,
) -> None:
  """Writes a Structure to PDB format for LigandMPNN.

  PDB output contains ATOM records for protein backbone atoms, TER records
  at chain boundaries, and HETATM records for ligand atoms, compatible with
  ProDy and LigandMPNN parsing.

  Args:
    struct: Structure to serialize.
    filepath: Destination file path.
  """
  lines = []

  # Ensure 1D coordinate arrays (handle leading model dimensions if present).
  atom_xs = _leading_dim(struct.atom_x)
  atom_ys = _leading_dim(struct.atom_y)
  atom_zs = _leading_dim(struct.atom_z)
  atom_occ = _leading_dim(struct.atom_occupancy)
  atom_b = _leading_dim(struct.atom_b_factor)

  atom_idx = 1
  prev_is_prot = False
  prev_res_name = ''
  prev_chain = ''
  prev_res_id = 0

  for (
      is_prot,
      atom_name,
      res_name,
      chain_id,
      res_id,
      atom_x,
      atom_y,
      atom_z,
      occ,
      b_fac,
      elem,
  ) in zip(
      struct.is_protein_mask,
      struct.atom_name,
      struct.res_name,
      struct.chain_id,
      struct.res_id,
      atom_xs,
      atom_ys,
      atom_zs,
      atom_occ,
      atom_b,
      struct.atom_element,
  ):
    # Emit TER record when transitioning from protein to ligand or chain change.
    if prev_is_prot and (not is_prot or chain_id != prev_chain):
      lines.append(
          f'TER   {atom_idx % 100000:5d}      {prev_res_name[:3]:>3s}'
          f' {prev_chain[:1]:1s}{prev_res_id % 10000:>4d}'
      )
      atom_idx += 1

    prev_is_prot = bool(is_prot)
    prev_res_name = res_name
    prev_chain = chain_id
    prev_res_id = res_id

    record = 'ATOM  ' if is_prot else 'HETATM'
    atom_serial = atom_idx % 100000
    name = atom_name if len(atom_name) == 4 else f' {atom_name}'
    name = name[:4]
    alt_loc = ''
    residue_index = res_id % 10000
    insertion_code = ''
    # Ensure occupancy is > 0 so LigandMPNN (atoms.select("occupancy > 0")) does
    # not drop atoms.
    occupancy = occ if occ > 0 else 1.00
    b_factor = b_fac
    element = elem[:2]
    charge = ''

    x_str = _format_coord(atom_x)
    y_str = _format_coord(atom_y)
    z_str = _format_coord(atom_z)
    occ_str = _format_float(occupancy, 6, 2)
    b_fac_str = _format_float(b_factor, 6, 2)

    lines.append(
        f'{record}{atom_serial:>5} {name:<4}{alt_loc:>1}'
        f'{res_name[:3]:>3} {chain_id[:1]:>1}'
        f'{residue_index:>4}{insertion_code:>1}   '
        f'{x_str}{y_str}{z_str}'
        f'{occ_str}{b_fac_str}          '
        f'{element:>2}{charge:>2}'
    )
    atom_idx += 1

  # Emit TER record if the final atom was from a protein chain.
  if prev_is_prot:
    lines.append(
        f'TER   {atom_idx % 100000:5d}      {prev_res_name[:3]:>3s}'
        f' {prev_chain[:1]:1s}{prev_res_id % 10000:>4d}'
    )

  lines.append('END')
  epath.Path(filepath).write_text(
      add_license_and_terms_of_use_header('\n'.join(lines) + '\n')
  )


def extract_fixed_residues(
    *,
    fixed_seq_mask: np.ndarray,
    sequence_mask: np.ndarray,
    residue_indices: np.ndarray | None = None,
    is_ligand_mask: np.ndarray | None = None,
    chain_id: str = 'A',
) -> tuple[list[str], str]:
  """Extracts fixed motif residue identifiers for LigandMPNN sequence redesign.

  Args:
    fixed_seq_mask: Mask where 1 indicates a fixed motif residue.
    sequence_mask: Mask where 1 indicates a valid residue in the chain.
    residue_indices: Optional array of integer residue IDs corresponding to each
      residue in sequence_mask. If provided, the actual residue IDs are used
      instead of assuming 1-based sequential indices.
    is_ligand_mask: Optional mask where 1 indicates a ligand token rather than a
      protein residue. Ligand tokens occupy trailing positions of
      `fixed_seq_mask` and are always fixed, but they are not residues of the
      redesigned protein chain. Without this mask they would be emitted as
      phantom identifiers (e.g. 'A132' on a 131-residue chain) that no
      downstream consumer can resolve.
    chain_id: Chain identifier for the fixed residues (default "A").

  Returns:
    Tuple of (list of residue identifiers like ["A12", "A45"], space-separated
    string like "A12 A45").
  """
  num_res = int(np.sum(np.asarray(sequence_mask)))
  fixed_mask = np.asarray(fixed_seq_mask[:num_res]) == 1

  if is_ligand_mask is not None:
    ligand_mask = np.asarray(is_ligand_mask).reshape(-1)[:num_res] == 1
    fixed_mask = fixed_mask & ~ligand_mask

  if residue_indices is None:
    res_ids = np.where(fixed_mask)[0] + 1
  else:
    res_ids = np.asarray(residue_indices[:num_res])[fixed_mask]
  residue_list = [f'{chain_id}{idx}' for idx in res_ids]

  return residue_list, ' '.join(residue_list)


def atom_neighbors(
    struct: structure.Structure,
    atoms: Sequence[tuple[str, int, str]],
    ca_dist_cutoffs: Sequence[int] = (6, 8, 12),
    allatom_dist_cutoffs: Sequence[int] = (4, 6, 8),
) -> tuple[Sequence[float], Sequence[float]]:
  """Count number of protein atoms that are near specific atoms.

  Args:
    struct: Structure to analyze.
    atoms: List of (chain_id, residue_id, atom_name) tuples to calculate the
      number of neighbors for.
    ca_dist_cutoffs: Distances at which to count C-alpha neighbors of `atoms`.
      The default values correspond to roughly the distance between a catalytic
      residue and the second coordination sphere.
    allatom_dist_cutoffs: Distances at which to count heavy atom neighbors of
      `atoms`. The default values correspond roughly to atoms in the van der
      Waals radius. Several radii are chosen to reduce noise.

  Returns:
    ca_neighbors: Mean over all cutoff distances of the number of C-alpha
      neighbors, list of values for each specified atom.
    allatom_neighbors: Mean over all cutoff distances of the number of heavy
      atom neighbors, list of values for each specified atom.
  """
  atom_mask = get_mask_from_atoms(struct, atoms)
  struct_from = struct.filter(atom_mask)
  struct_to = struct.filter(~atom_mask)

  d = scipy.spatial.distance.cdist(
      struct_from.coords,
      struct_to.coords[struct_to.atom_name == 'CA'],
  )
  result = d[None] < np.array(ca_dist_cutoffs)[:, None, None]
  ca_neighbors = result.sum(-1).mean(0)

  d = scipy.spatial.distance.cdist(struct_from.coords, struct_to.coords)
  result = d[None] < np.array(allatom_dist_cutoffs)[:, None, None]
  allatom_neighbors = result.sum(-1).mean(0)

  return ca_neighbors, allatom_neighbors


def get_mask_from_atoms(
    struct: structure.Structure,
    atoms: Sequence[tuple[str, int, str]],
) -> np.ndarray:
  """Returns a binary mask from `(chain_id, residue_id, atom_name)`."""
  mask = np.zeros((struct.num_atoms,), dtype=bool)
  atom_ids = {atom_id: i for i, atom_id in enumerate(struct.atom_ids)}
  for chain_id, residue_id, atom_name in atoms:
    mask[atom_ids[(chain_id, str(residue_id), None, atom_name)]] = True
  return mask


def get_atoms(struct: structure.Structure) -> Sequence[tuple[str, int, str]]:
  """Returns a list of `(chain_id, residue_id, atom_name)` from a structure.

  Converts `atom_ids` from the Structure format to a list of
  `(chain_id, residue_id, atom_name)`, leaving out the `None` `insertion_code`
  and making the `residue_id` an integer.

  Args:
    struct: Structure from which to extract atoms.

  Returns:
    A list of `(chain_id, residue_id, atom_name)`.
  """
  # atom_ids is a sequence of (chain_id, res_id, insertion_code, atom_name)
  # where insertion_code is always None. We leave that out here.
  return [
      (atom_id[0], int(atom_id[1]), atom_id[3]) for atom_id in struct.atom_ids
  ]


def filter_to_residues_with_any_atom_fixed(
    struct: structure.Structure,
) -> structure.Structure:
  """Filters to residues with any atom fixed as indicated by `atom_b_factor`."""
  fixed_res_keys = set(
      struct.atoms_table.res_key[np.isclose(struct.atom_b_factor, 1)]
  )
  return struct.filter(
      membership.isin(struct.atoms_table.res_key, fixed_res_keys)
  )


_RESIDUE_SPEC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'^([A-Za-z]+)(-?\d+)$'
)


def parse_residue_spec(spec: str) -> tuple[str, int]:
  """Parses a single residue specifier (e.g. 'A12' or 'AB12') into (chain_id, res_id).

  Args:
    spec: Residue identifier consisting of an alphabetic chain identifier
      followed by a residue number (e.g. 'A12' or 'AB12').

  Returns:
    A (chain_id, res_id) tuple.

  Raises:
    ValueError: If `spec` cannot be parsed.
  """
  match = _RESIDUE_SPEC_PATTERN.match(spec.strip())
  if match is None:
    raise ValueError(
        f'Invalid fixed residue specifier: {spec!r}. Expected string like'
        " 'A12' or 'AB12'."
    )
  return match.group(1), int(match.group(2))


def parse_fixed_residues(
    fixed_residues: Sequence[str] | str | None,
) -> set[tuple[str, int]]:
  """Parses fixed residue specifications into a set of (chain_id, res_id).

  Args:
    fixed_residues: Sequence of `<Chain><ResID>` residue identifiers (e.g.
      `['A12', 'B45']`), a single string of space-separated identifiers (e.g.
      `'A12 B45'`), or None.

  Returns:
    A set of (chain_id, res_id) tuples.

  Raises:
    ValueError: If an item cannot be parsed.
    TypeError: If an item is not a str.
  """
  if fixed_residues is None:
    return set()

  if isinstance(fixed_residues, str):
    raw_items = fixed_residues.strip().split()
  else:
    raw_items = []
    for item in fixed_residues:
      if not isinstance(item, str):
        raise TypeError(
            f'Unsupported type for fixed residue: {type(item).__name__}'
            f' ({item!r})'
        )
      if ' ' in item.strip():
        raw_items.extend(item.strip().split())
      else:
        raw_items.append(item)

  return {parse_residue_spec(item) for item in raw_items}


def fixed_atom_mask(
    struct: structure.Structure,
    fixed_set: Set[tuple[str, int]],
) -> np.ndarray:
  """Returns a per-atom boolean mask of atoms in `fixed_set` residues.

  Args:
    struct: The structure whose atoms are to be masked.
    fixed_set: Set of (chain_id, res_id) tuples, as returned by
      `parse_fixed_residues`.

  Returns:
    A boolean array of length `struct.num_atoms`.
  """
  if not fixed_set:
    return np.zeros(struct.num_atoms, dtype=bool)
  return np.array(
      [
          (chain_id, res_id) in fixed_set
          for chain_id, res_id in zip(
              struct.chain_id, struct.res_id, strict=True
          )
      ],
      dtype=bool,
  )


def populate_fixed_b_factors(
    struct: structure.Structure,
    fixed_residues: Sequence[str] | str | None,
) -> structure.Structure:
  """Populates atom_b_factor as 1.0 for fixed motif residues, 0.0 otherwise.

  Note that this overwrites any existing `atom_b_factor` values. Use
  `has_binary_atom_b_factor` first if an existing fixed-atom annotation needs
  to be preserved.

  Args:
    struct: The macromolecular structure to annotate with motif B-factors.
    fixed_residues: Fixed residue identifiers, as accepted by
      `parse_fixed_residues`.

  Returns:
    A copy of `struct` with atom_b_factor set to 1.0 for fixed motif residues
    and 0.0 for all other atoms.

  Raises:
    ValueError: If a string element cannot be parsed into chain and residue ID.
    TypeError: If an element is not a str.
  """
  fixed_set = parse_fixed_residues(fixed_residues)
  b_factors = fixed_atom_mask(struct, fixed_set).astype(np.float32)
  return struct.copy_and_update_atoms(atom_b_factor=b_factors)


def has_binary_atom_b_factor(
    struct: structure.Structure, tol: float = 1e-6
) -> bool:
  """Returns True if `atom_b_factor` encodes a binary fixed-atom indicator.

  A binary `atom_b_factor` field marks fixed motif (and ligand) atoms with 1.0
  and everything else with 0.0. Any other value distribution (e.g. pLDDT or
  crystallographic B-factors) carries no fixed-atom information.

  Args:
    struct: The structure to inspect.
    tol: Absolute tolerance when comparing against 0.0 and 1.0.

  Returns:
    True if every `atom_b_factor` value is close to either 0.0 or 1.0.
  """
  if struct.atom_b_factor is None or struct.atom_b_factor.size == 0:
    return False
  b_factors = np.asarray(struct.atom_b_factor, dtype=np.float32)
  return bool(
      np.all(
          np.isclose(b_factors, 0.0, atol=tol)
          | np.isclose(b_factors, 1.0, atol=tol)
      )
  )


def _assert_has_full_backbone_with_oxygen(
    struct: structure.Structure, chain_id: str
) -> None:
  """Raises if any residue in `chain_id` lacks a complete N/CA/C/O backbone.

  Args:
    struct: The structure to check.
    chain_id: Chain identifier of the chain to check.

  Raises:
    ValueError: If a residue on `chain_id` is missing a backbone atom.
  """
  chain_struct = struct.filter(chain_id=chain_id)
  required_atoms = set(atom_types.PROTEIN_BACKBONE_WITH_OXYGEN)
  present_atoms_by_res_id: dict[int, set[str]] = {}
  for res_id, atom_name in zip(
      chain_struct.res_id, chain_struct.atom_name, strict=True
  ):
    present_atoms_by_res_id.setdefault(int(res_id), set()).add(atom_name)

  for res_id, present_atoms in sorted(present_atoms_by_res_id.items()):
    missing_atoms = required_atoms - present_atoms
    if missing_atoms:
      raise ValueError(
          f'Residue {res_id} on chain {chain_id} is missing the following'
          f' backbone atoms: {sorted(missing_atoms)}. It only has:'
          f' {sorted(present_atoms)}.'
      )


def create_resequenced_structure(
    base_struct: structure.Structure,
    redesigned_seq: str,
    fixed_residues: Sequence[str] | str | None = None,
    chain_id: str = 'A',
) -> structure.Structure:
  """Derives a resequenced structure reflecting the redesigned sequence.

  Retains full catalytic sidechains for fixed motif residues; retains only
  backbone atoms (N, CA, C, O) for redesigned residues; retains non-target
  chains (e.g. ligands) unchanged.

  Fixed motif atoms are marked with atom_b_factor = 1.0. Any binary fixed-atom
  annotation already present on `base_struct` is preserved rather than
  overwritten, since upstream motif preparation also marks atoms that are not
  covered by `fixed_residues` (in particular ligand atoms), and downstream
  metrics infer the motif from atom_b_factor. Non-binary b-factors (e.g. pLDDT)
  carry no fixed-atom information and are reset to 0.0.

  Args:
    base_struct: Base Structure (e.g. from generation stage).
    redesigned_seq: 1-letter amino acid sequence for the redesigned chain.
    fixed_residues: Optional fixed residue specifiers.
    chain_id: Chain identifier of the redesigned protein chain (default 'A').

  Returns:
    A new Structure instance representing the resequenced complex.

  Raises:
    ValueError: If `chain_id` is not present in `base_struct`, if the sequence
      length does not match the residue count in that chain, if a residue on
      that chain is missing backbone atoms, if the sequence contains a
      non-canonical amino acid, or if a fixed residue would be renamed.
  """
  fixed_set = parse_fixed_residues(fixed_residues)

  # 1. Update residue names in residues table for target chain.
  res_table = base_struct.residues_table
  chain_ids = base_struct.chains_table.apply_array_to_column(
      'id', res_table.chain_key
  )
  res_ids = res_table.id
  res_names = np.array(res_table.name, dtype=object)

  chain_mask = chain_ids == chain_id
  chain_res_count = int(np.sum(chain_mask))
  if chain_res_count == 0:
    raise ValueError(
        f'Chain {chain_id!r} is not present in the base structure. Available'
        f' chains: {sorted(set(chain_ids))}.'
    )
  if len(redesigned_seq) != chain_res_count:
    raise ValueError(
        f'Length of redesigned sequence ({len(redesigned_seq)}) does not match'
        f' number of residues in chain {chain_id} ({chain_res_count}).'
    )

  # Every residue on the target chain is either reduced to its backbone or kept
  # as a fixed motif residue, so an incomplete backbone would silently produce
  # truncated residues.
  _assert_has_full_backbone_with_oxygen(base_struct, chain_id)

  chain_res_indices = np.where(chain_mask)[0]
  for k, res_idx in enumerate(chain_res_indices):
    one_letter = redesigned_seq[k]
    res_id = int(res_ids[res_idx])
    res_name = residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(one_letter)
    if res_name is None:
      raise ValueError(
          f'Unsupported amino acid {one_letter!r} at position {k} of the'
          f' redesigned sequence (chain {chain_id}, residue {res_id}). Only'
          ' the 20 canonical amino acids are supported.'
      )
    if (chain_id, res_id) in fixed_set and res_name != res_names[res_idx]:
      # Fixed residues keep their full sidechain, so renaming one would emit a
      # residue whose atoms do not match its name. This most likely indicates a
      # misalignment between the redesigned sequence and the base structure.
      raise ValueError(
          f'Fixed residue {chain_id}{res_id} would be renamed from'
          f' {res_names[res_idx]} to {res_name} by the redesigned sequence.'
          ' The redesigned sequence is likely misaligned with the base'
          ' structure.'
      )
    res_names[res_idx] = res_name

  struct_with_names = base_struct.copy_and_update_residues(res_name=res_names)

  # 2. Filter atoms: keep all non-target chain atoms, keep all atoms for fixed
  # residues, and keep only backbone atoms (N, CA, C, O) for redesigned
  # residues.
  is_target_chain = struct_with_names.chain_id == chain_id
  is_fixed = fixed_atom_mask(struct_with_names, fixed_set)
  is_backbone = string_array.isin(
      struct_with_names.atom_name, set(atom_types.PROTEIN_BACKBONE_WITH_OXYGEN)
  )
  atom_mask = (~is_target_chain) | is_fixed | is_backbone

  filtered_struct = struct_with_names.filter(atom_mask)

  # 3. Mark fixed motif atoms with atom_b_factor = 1.0, preserving any existing
  # binary fixed-atom annotation (e.g. ligand atoms marked upstream).
  if has_binary_atom_b_factor(filtered_struct):
    b_factors = np.asarray(
        filtered_struct.atom_b_factor, dtype=np.float32
    ).copy()
  else:
    b_factors = np.zeros(filtered_struct.num_atoms, dtype=np.float32)
  b_factors[fixed_atom_mask(filtered_struct, fixed_set)] = 1.0

  return filtered_struct.copy_and_update_atoms(
      atom_b_factor=b_factors
  ).copy_and_update_globals(name=f'{base_struct.name}_resequenced')
