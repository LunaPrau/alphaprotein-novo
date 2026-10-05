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

"""Data types and containers for AlphaProtein Novo."""

from __future__ import annotations

import binascii
from collections.abc import Callable, Mapping
import dataclasses
import functools
from typing import Any, TypeAlias

from absl import logging
from alphafold3 import structure
from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.constants import ligand_constants
from alphaprotein_novo.data import structure_utils
import chex
import jax
import jax.numpy as jnp
import numpy as np


class _GenericType:

  def __getitem__(self, item):
    return Any


ArrayT = Any  # pylint: disable=invalid-name
Bool = _GenericType()  # pylint: disable=invalid-name
Float = _GenericType()  # pylint: disable=invalid-name
Int = _GenericType()  # pylint: disable=invalid-name
L = Any
_ARRAY_TYPES = (np.ndarray, jax.Array)


def typed(fn):
  return fn


# Hard-coded array dimensions.
MAX_NUM_LIGANDS = 9
NUM_ATOMS_PER_LIGAND = 140


def _string_to_base16int(s: str) -> int:
  if len(s) > 4:
    raise ValueError(
        f'The string length must be at most 4, but got length {len(s)}.'
    )
  b_in_base16 = binascii.hexlify(s.encode('ascii'))
  return int(b_in_base16, 16)


def _base16int_to_string(base16_int: int) -> str:
  hex_repr = hex(base16_int)[2:]
  # Make sure there is an even number for decoding.
  if len(hex_repr) % 2 != 0:
    hex_repr = '0' + hex_repr
  b = binascii.unhexlify(hex_repr)
  return b.decode(encoding='ascii')


def _map_over_ndarray(
    xs: np.ndarray,
    f: Callable[..., Any],
    return_type: type[np.int32] | type[object],
) -> np.ndarray:
  shape = xs.shape
  res = []
  for x in xs.flatten():
    res.append(f(x))
  return np.asarray(res, dtype=return_type).reshape(shape)


def string_to_numerical_array(string_array: np.ndarray) -> np.ndarray:
  if string_array.dtype != object:
    raise ValueError(
        f'Expected an array of dtype `object`, but got {string_array.dtype}.'
    )
  return _map_over_ndarray(string_array, _string_to_base16int, np.int32)


def numerical_to_string_array(numerical_array: np.ndarray) -> np.ndarray:
  if numerical_array.dtype != np.int32:
    raise ValueError(
        'Expected an array of dtype `np.int32`, but got'
        f' {numerical_array.dtype}.'
    )
  # The default type for strings in Structure is `object`.
  return _map_over_ndarray(numerical_array, _base16int_to_string, object)


def get_placeholder_ligand_features(
    num_ligands: int,
    num_atoms_per_ligand: int,
    batch_sizes: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
  """Creates placeholders for ligand features."""
  ligand_features = {
      'component_atom_names_numerical': (
          ord('0')
          * np.ones(
              batch_sizes + (num_ligands, num_atoms_per_ligand), dtype=np.int32
          )
      ),
      'component_atomic_numbers': np.zeros(
          batch_sizes + (num_ligands, num_atoms_per_ligand), dtype=np.int32
      ),
      'component_entity_id': np.zeros(
          batch_sizes + (num_ligands,), dtype=np.int32
      ),
      'component_fragment_indices': np.zeros(
          batch_sizes + (num_ligands, num_atoms_per_ligand), dtype=np.int32
      ),
      'component_id_numerical': (
          ord('0') * np.ones(batch_sizes + (num_ligands,), dtype=np.int32)
      ),
      'component_padding_mask_1d': np.zeros(
          batch_sizes + (num_ligands, num_atoms_per_ligand), dtype=np.int32
      ),
  }
  return ligand_features


_Array = Any


def _arrays_allclose(
    x: _Array,
    y: _Array,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
  """Compares two numpy or JAX arrays for elementwise approximate equality.

  Note: if one of the arrays is a JAX array and the other a numpy array this
  will trigger a device-to-host transfer of the JAX array.

  Args:
    x: Numpy or JAX array.
    y: Numpy or JAX array.
    rtol: The relative tolerance parameter for `np.allclose()`.
    atol: The absolute tolerance parameter for `np.allclose()`.

  Returns:
    Boolean indicating whether `x` and `y` are elementwise equal up to an
    absolute and relative tolerance.

  Raises:
    ValueError: If one of the arrays is not of numeric dtype.
  """
  x_is_numpy = isinstance(x, np.ndarray)
  y_is_numpy = isinstance(y, np.ndarray)

  np_allclose = functools.partial(np.allclose, rtol=rtol, atol=atol)
  jnp_allclose = functools.partial(jnp.allclose, rtol=rtol, atol=atol)

  is_numeric = lambda x: np.issubdtype(x.dtype, np.number)  # pyrefly: ignore[attribute-error]
  if not is_numeric(x) or not is_numeric(y):
    raise ValueError('Both input arrays must be of numeric type.')

  if x_is_numpy and y_is_numpy:
    return np_allclose(x, y)
  elif x_is_numpy and not y_is_numpy:
    return np_allclose(x, np.asarray(y))
  elif not x_is_numpy and y_is_numpy:
    return np_allclose(np.asarray(x), y)
  else:
    return jnp_allclose(x, y).item()


@chex.dataclass(frozen=True)
class Protein:
  """Array-based container for protein and ligand chains.

  Shape annotation:
    B: batching dimensions
    L: sequence length
    A: number of atoms
    3: fixed spatial dimension
    N_lig: number of ligands
    140: fixed number of atoms per ligand

  Attributes:
    aatype: Amino acid type. Taken from [0, 19] for protein residues, 20 for
      UNK, and from [21, 21 + num_ligand_atom_types] for ligand residues.
    atom_mask: Mask indicating which atoms are present.
    atom_positions: Spatial atom positions.
    chain_index: An array that represents each chain with a separate integer.
    is_ligand_mask: Mask indicating which residues belong to ligands.
    residue_index: Integer array that enumerates residues.
    sequence_mask: Mask that indicates which residues are part of the sequence.
    component_atom_names: Ligand atom names.
    component_atom_names_numerical: A numerical representation of the ligand
      atom names.
    component_atomic_numbers: Atomic number of the ligand atoms.
    component_entity_id: Chain index for each ligand.
    component_fragment_indices: Enumerates the fragments for each ligand. If
      `use_atomistic_fragmentation` is True, than each fragment represents an
      atom.
    component_id: Ligand CCD codes.
    component_id_numerical: Numerical representation of the ligand CCD code.
    component_padding_mask_1d: Mask where True indicates a real atom and False
      indicates padding.
    entity_id: Denotes chains of the same entity (e.g., protein, ligand) by the
      same integer. In contrast `asym_id` uses separate integers for each chain,
      also within the same class.
    is_batched: Indicates whether the Protein has a batching dimension.
  """

  aatype: Int[ArrayT, '*B L']
  atom_mask: Int[ArrayT, '*B L A']
  atom_positions: Float[ArrayT, '*B L A 3']
  chain_index: Int[ArrayT, '*B L']
  residue_index: Int[ArrayT, '*B L']
  sequence_mask: Int[ArrayT, '*B L']
  is_ligand_mask: Int[ArrayT, '*B L'] | None = None
  entity_id: Int[ArrayT, '*B L'] | None = None

  # Ligand features.
  component_atom_names_numerical: Int[ArrayT, '*B N_lig 140'] | None = None
  component_atomic_numbers: Int[ArrayT, '*B N_lig 140'] | None = None
  component_entity_id: Int[ArrayT, '*B N_lig'] | None = None
  component_fragment_indices: Int[ArrayT, '*B N_lig 140'] | None = None
  component_id_numerical: Int[ArrayT, '*B N_lig'] | None = None
  component_padding_mask_1d: Int[ArrayT, '*B N_lig 140'] | None = None

  # Internal options.
  # Declare unhashable. This is a safety measure, since the Protein can be
  # encoded by numpy arrays, which are mutable.
  __hash__ = None

  # Post initialization methods.
  # Is also called with `dataclasses.replace()`.
  def __post_init__(self):
    if self.component_atomic_numbers is None:
      self.init_ligand_features()

    if self.entity_id is None:
      self.init_entity_id()

    if self.is_ligand_mask is None:
      self.init_ligand_mask()

  def init_ligand_features(self, num_ligands: int = MAX_NUM_LIGANDS):
    ligand_init_kwargs = get_placeholder_ligand_features(
        num_ligands=num_ligands,
        num_atoms_per_ligand=NUM_ATOMS_PER_LIGAND,
        batch_sizes=self.aatype.shape[:-1],
    )
    if isinstance(self.aatype, jax.Array):
      ligand_init_kwargs = jax.device_put(ligand_init_kwargs)
    for k, v in ligand_init_kwargs.items():
      object.__setattr__(self, k, v)

  def init_entity_id(self):
    # If not specified otherwise, every chain is its own entity.
    object.__setattr__(self, 'entity_id', self.chain_index)

  def init_ligand_mask(self):
    ligand_mask = np.zeros(self.aatype.shape, dtype=np.int32)
    if isinstance(self.aatype, jax.Array):
      ligand_mask = jax.device_put(ligand_mask)
    object.__setattr__(self, 'is_ligand_mask', ligand_mask)

  def _eq(self, other, log_if_different: bool):
    """Compares two Proteins ONLY based on their array data fields."""
    if not isinstance(other, Protein):
      return NotImplemented
    field_names = [field.name for field in dataclasses.fields(self)]
    for field_name in field_names:
      if (field_name == 'cached_mmcif') or ('component_' in field_name):
        continue
      attr_self = getattr(self, field_name)
      attr_other = getattr(other, field_name)
      self_is_array = isinstance(attr_self, _ARRAY_TYPES)
      other_is_array = isinstance(attr_other, _ARRAY_TYPES)
      if self_is_array and other_is_array:
        if not _arrays_allclose(attr_self, attr_other):
          if log_if_different:
            logging.info(
                '%s differs: %s versus %s', field_name, attr_self, attr_other
            )
          return False
      elif (self_is_array and not other_is_array) or (
          not self_is_array and other_is_array
      ):
        if log_if_different:
          logging.info('%s differs: different array types', field_name)
        return False
      else:
        logging.warning(
            'Field `%s` is not of array type for `self` and `other`. It is'
            ' neglected for array comparison, i.e., `self` and `other` might be'
            ' considered equal despite having different attributes for this'
            ' field.',
            field_name,
        )
    return True

  def __eq__(self, other):
    """Compares two Proteins ONLY based on their array data fields."""
    return self._eq(other, log_if_different=False)

  def replace(self, **kwargs):
    """Returns a new Protein with the given fields replaced.

    Args:
      **kwargs: The fields to replace.

    Overrides the base class method in chex._src/dataclass.py to silence a
    type-checking attribute error.
    """
    return dataclasses.replace(self, **kwargs)

  # Getter methods to convert numerical representation back to string.
  @property
  def component_id(self) -> np.ndarray:
    return numerical_to_string_array(self.component_id_numerical)  # pyrefly: ignore[bad-argument-type]

  @property
  def component_atom_names(self) -> np.ndarray:
    return numerical_to_string_array(self.component_atom_names_numerical)  # pyrefly: ignore[bad-argument-type]

  @property
  def is_batched(self) -> bool:
    return len(self.aatype.shape) > 1

  # Data structure conversion methods.
  def to_structure(
      self,
      b_factors: jax.Array | np.ndarray | None = None,
  ) -> structure.Structure:
    if self.is_batched:
      raise ValueError('Require un-batched data.')
    features = self.to_feature_dict()
    return structure_utils.structure_from_features(
        features=features,
        b_factors=b_factors,
    )

  def to_feature_dict(self) -> dict[str, Any]:
    return {
        'aatype': np.array(self.aatype),
        'all_atom_mask': np.array(self.atom_mask),
        'all_atom_positions': np.array(self.atom_positions),
        'asym_id': np.array(self.chain_index),
        'is_ligand_mask': np.array(self.is_ligand_mask),
        # TODO(b/370269566): Ligand residue indices are different per atom,
        # and thus will be overwritten by 1-indexed residue-wise integers upon
        # Structure construction.
        'residue_index': np.array(self.residue_index),
        'seq_mask': np.array(self.sequence_mask),
        'component_atom_names': self.component_atom_names,
        'component_atomic_numbers': np.array(self.component_atomic_numbers),
        'component_entity_id': np.array(self.component_entity_id),
        'component_fragment_indices': np.array(self.component_fragment_indices),
        'component_id': self.component_id,
        'component_padding_mask_1d': np.array(self.component_padding_mask_1d),
        'entity_id': np.array(self.entity_id),
    }

  @classmethod
  def from_structure(
      cls: type['Protein'],
      struct: structure.Structure,
      num_residues: int | None = None,
      num_ligands: int | None = None,
      num_prot_aatypes: int = ligand_constants.NUM_PROT_AATYPES,
  ) -> 'Protein':
    """Creates a Protein from a Structure."""
    # Check feasibility of the provided parameters.
    num_total_residues_in_struct = structure_utils.count_all_residues(
        struct,
        use_ccd_for_ligands=True,
    )
    num_ligands_in_struct = struct.filter_to_entity_type(
        ligand=True
    ).num_residues(count_unresolved=True)

    if num_residues is None:
      num_residues = num_total_residues_in_struct
    else:
      if num_total_residues_in_struct > num_residues:
        raise ValueError(
            'Number of residues encoded in struct is larger than requested'
            f' number of residues: {num_total_residues_in_struct} >'
            f' {num_residues}.'
        )
    if num_ligands is None:
      num_ligands = num_ligands_in_struct
    else:
      if num_ligands_in_struct > num_ligands:
        raise ValueError(
            'Number of ligands encoded in struct is larger than requested'
            f' number of ligands: {num_ligands_in_struct} > {num_ligands}.'
        )

    features = structure_utils.features_from_structure(
        struct,
        num_residues,
        num_ligands,
        NUM_ATOMS_PER_LIGAND,
        ligand_constants.NUM_LIGAND_ELEMS,
        num_prot_aatypes=num_prot_aatypes,  # pyrefly: ignore[bad-argument-type]
    )
    return Protein.from_feature_dict(features)

  @classmethod
  def from_feature_dict(
      cls: type['Protein'],
      features: Mapping[str, Any],
  ) -> 'Protein':
    """Creates a Protein from a feature dictionary."""
    # Make sure a ligand mask exists.
    if 'is_ligand_mask' in features:
      is_ligand_mask = features['is_ligand_mask'].astype(np.int32)
    else:
      is_ligand_mask = np.zeros(features['seq_mask'].shape, dtype=np.int32)

    # `astype()` copies the array even if its dtype was already correct.
    init_kwargs = dict(
        aatype=features['aatype'].astype('int32'),
        atom_mask=features['all_atom_mask'].astype('int32'),
        atom_positions=features['all_atom_positions'].astype('float32'),
        chain_index=features['asym_id'].astype('int32'),
        is_ligand_mask=is_ligand_mask.astype('int32'),
        residue_index=features['residue_index'].astype('int32'),
        sequence_mask=features['seq_mask'].astype('int32'),
        entity_id=features['entity_id'].astype('int32')
        if 'entity_id' in features
        else None,
    )
    # Check if ligand features are provided.
    if 'component_atomic_numbers' in features:
      if 'component_atom_names_numerical' not in features:
        component_atom_names_numerical = string_to_numerical_array(
            features['component_atom_names']
        )
      else:
        component_atom_names_numerical = features[
            'component_atom_names_numerical'
        ]

      if 'component_id_numerical' not in features:
        component_id_numerical = string_to_numerical_array(
            features['component_id']
        )
      else:
        component_id_numerical = features['component_id_numerical']
      ligand_init_kwargs = dict(
          component_atom_names_numerical=component_atom_names_numerical.astype(
              'int32'
          ),
          component_atomic_numbers=features['component_atomic_numbers'].astype(
              'int32'
          ),
          component_entity_id=features['component_entity_id'].astype('int32'),
          component_fragment_indices=features[
              'component_fragment_indices'
          ].astype('int32'),
          component_id_numerical=component_id_numerical.astype('int32'),
          component_padding_mask_1d=features[
              'component_padding_mask_1d'
          ].astype('int32'),
      )
      init_kwargs.update(ligand_init_kwargs)

    return Protein(**init_kwargs)


def _pad_to_size(
    arr: np.ndarray,
    size: int,
    dim: int = 0,
    padding_token: Any = 0,
) -> np.ndarray:
  """Pad an array to the specified size in dim."""
  if not hasattr(arr, 'shape') or not arr.shape:
    return arr
  arr_shape = arr.shape
  if arr_shape[dim] > size:
    raise RuntimeError(
        f'Array is too large in dim {dim} to pad. {arr.shape[dim]} vs {size}'
    )
  arr_shape = list(arr.shape)
  pad_size = size - arr_shape[dim]
  arr_shape[dim] = pad_size
  pad_arr = np.full(
      shape=arr_shape,
      fill_value=padding_token,
      dtype=arr.dtype,
  )
  return np.concatenate([arr, pad_arr], axis=dim)


def make_blank_protein(
    max_num_res: int,
    max_num_ligands: int = 0,
) -> Protein:
  """Creates a blank placeholder Protein."""
  protein_features = dict(
      aatype=np.zeros([max_num_res]).astype(np.int32),
      all_atom_positions=np.zeros([max_num_res, atom_types.ATOM37_NUM, 3]),
      all_atom_mask=np.zeros([max_num_res, atom_types.ATOM37_NUM]).astype(
          np.int32
      ),
      seq_mask=np.ones([max_num_res]).astype(np.int32),
      residue_index=np.arange(1, max_num_res + 1),
      asym_id=np.zeros([max_num_res]).astype(np.int32),
      is_ligand_mask=np.zeros([max_num_res]).astype(np.int32),
      entity_id=np.zeros([max_num_res]).astype(np.int32),
  )
  ligand_features = get_placeholder_ligand_features(
      num_ligands=max_num_ligands,
      num_atoms_per_ligand=NUM_ATOMS_PER_LIGAND,
  )
  features = {**ligand_features, **protein_features}
  return Protein.from_feature_dict(features)


def pad_protein(
    prot: Protein,
    num_residues: int,
    num_ligands: int | None = None,
) -> Protein:
  """Pad protein features along the sequence dimension."""
  padded_protein_features = {}
  protein_features = set()
  ligand_features = set()
  for field in dataclasses.fields(prot):
    if 'component_' in field.name:
      ligand_features.add(field.name)
    else:
      protein_features.add(field.name)

  seq_dim = len(prot.aatype.shape) - 1
  ligand_dim = len(prot.component_entity_id.shape) - 1  # pyrefly: ignore[missing-attribute]

  for k, v in dataclasses.asdict(prot).items():
    if k in protein_features:
      padded_protein_features[k] = _pad_to_size(
          v, num_residues, dim=seq_dim  # pyrefly: ignore[bad-argument-type]
      ).astype(
          v.dtype  # pyrefly: ignore[missing-attribute]
      )
    elif k in ligand_features and num_ligands is not None:
      if k in ['component_id_numerical', 'component_atom_names_numerical']:
        padding_token = ord('0')
      else:
        padding_token = 0
      padded_protein_features[k] = _pad_to_size(
          v,  # pyrefly: ignore[bad-argument-type]
          num_ligands,
          dim=ligand_dim,
          padding_token=padding_token,
      ).astype(
          v.dtype  # pyrefly: ignore[missing-attribute]
      )
    else:
      padded_protein_features[k] = v

  if isinstance(prot.aatype, jax.Array):
    padded_protein_features = jax.tree.map(jnp.asarray, padded_protein_features)
  return prot.replace(**padded_protein_features)


@chex.dataclass(frozen=True, eq=True)
class DiffusionInput:
  """Input to diffusion models."""

  protein: Protein
  fixed_atom_mask: jax.Array
  fixed_seq_mask: jax.Array
  t_struct: jax.Array | None = None
  t_seq: jax.Array | None = None
  self_cond_atom_positions: jax.Array | None = None
  self_cond_aatype: jax.Array | None = None
  crop_cond_atom_positions: jax.Array | None = None
  crop_cond_aatype: jax.Array | None = None
  unindexed_motif_atom_positions: jax.Array | None = None
  unindexed_motif_atom_mask: jax.Array | None = None
  unindexed_motif_aatype: jax.Array | None = None
  unindexed_motif_aatype_mask: jax.Array | None = None
  unindexed_motif_orig_res_id: jax.Array | None = None
  unindexed_motif_orig_chain_index: jax.Array | None = None
  # Ligand conditioning features.
  ligand_charge: jax.Array | None = None
  ligand_hybridization: jax.Array | None = None

  def __post_init__(self):
    """Populate `None` fields with defaults to have a valid JAX array."""
    if self.fixed_atom_mask is None:
      object.__setattr__(
          self,
          'fixed_atom_mask',
          jnp.zeros_like(self.protein.atom_mask),
      )
    if self.fixed_seq_mask is None:
      object.__setattr__(
          self,
          'fixed_seq_mask',
          self.fixed_atom_mask.any(-1),
      )
    if self.self_cond_atom_positions is None:
      object.__setattr__(
          self,
          'self_cond_atom_positions',
          jnp.zeros_like(self.protein.atom_positions),
      )
    if self.self_cond_aatype is None:
      num_tokens = (
          ligand_constants.NUM_PROT_AATYPES
          + ligand_constants.NUM_LIGAND_ELEMS
          + 1
      )
      object.__setattr__(
          self,
          'self_cond_aatype',
          jnp.zeros(
              self.protein.aatype.shape + (num_tokens,), dtype=jnp.float32
          ),
      )
    if self.crop_cond_atom_positions is None:
      object.__setattr__(
          self,
          'crop_cond_atom_positions',
          jnp.zeros_like(self.protein.atom_positions),
      )
    if self.crop_cond_aatype is None:
      object.__setattr__(
          self,
          'crop_cond_aatype',
          jnp.zeros_like(self.protein.aatype),
      )
    if self.ligand_charge is None:
      object.__setattr__(
          self,
          'ligand_charge',
          jnp.zeros_like(self.protein.aatype),
      )
    if self.ligand_hybridization is None:
      protein_mask = (
          1 - self.protein.is_ligand_mask  # pyrefly: ignore[unsupported-operation]
      ) * self.protein.sequence_mask
      object.__setattr__(
          self,
          'ligand_hybridization',
          jnp.where(protein_mask, -1, 0),
      )

  def replace(self, **kwargs) -> 'DiffusionInput':
    return dataclasses.replace(self, **kwargs)


@chex.dataclass(frozen=True, eq=True)
class DiffusionOutput:
  """Output from diffusion models."""

  protein: Protein
  aatype_logits: jax.Array
  predicted_motif_indices_mask_logits: jax.Array | None = None


@chex.dataclass(frozen=True, kw_only=True)
class DenoiserOutput:
  """Output of the denoiser network in a diffusion model."""

  denoised_coords: jax.Array
  denoised_aatype_logits: jax.Array | None
  coords_uncertainty: jax.Array | None
  seq_uncertainty: jax.Array | None
  predicted_motif_indices_mask_logits: jax.Array | None


PredictFn: TypeAlias = Callable[
    [jax.Array, DiffusionInput],
    tuple[jax.Array, DenoiserOutput],
]
