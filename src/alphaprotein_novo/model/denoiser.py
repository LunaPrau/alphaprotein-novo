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

"""Protein denoiser network and feature encoding utilities."""

from __future__ import annotations

import dataclasses
from typing import Any, Final

from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.constants import ligand_constants
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.model import config as config_lib
from alphaprotein_novo.model import modules as cm
from alphaprotein_novo.model import types as types_lib
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def center_positions(
    atom_positions: jax.Array,
    centering_atom_mask: jax.Array,
    nonzero_atom_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Centers atom positions based on the centering atom mask."""
  centering_atom_mask = jnp.where(
      jnp.allclose(centering_atom_mask, 0, atol=0.01),
      jnp.ones_like(centering_atom_mask),
      centering_atom_mask,
  )
  centering_atom_mask = centering_atom_mask * nonzero_atom_mask
  s = jnp.sum(
      atom_positions * centering_atom_mask[..., None],
      keepdims=True,
      axis=(-2, -3),
  )
  m = jnp.sum(
      jnp.ones_like(atom_positions) * centering_atom_mask[..., None],
      keepdims=True,
      axis=(-2, -3),
  )
  center = s / (1e-6 + m)
  centered = atom_positions - center
  centered = centered * nonzero_atom_mask[..., None]
  return centered, center


def expand_dims_to(x: jax.Array, target: jax.Array) -> jax.Array:
  """Expands dimensions of x to match the number of dimensions of target."""
  num_dims_to_add = target.ndim - x.ndim
  assert num_dims_to_add >= 0
  return jnp.reshape(x, x.shape + (1,) * num_dims_to_add)


def _get_np_module(x: Any) -> Any:
  return jnp if isinstance(x, jax.Array) else np


_LIGAND_ATOM_INDEX = 0


def random_rotation_matrix(key: jax.Array) -> jax.Array:
  """Create a random 3D rotation matrix via Gram-Schmidt orthogonalization."""
  v0, v1 = jax.random.normal(key, shape=(2, 3))
  e0 = v0 / jnp.maximum(1e-10, jnp.linalg.norm(v0))
  v1 = v1 - e0 * jnp.dot(v1, e0)
  e1 = v1 / jnp.maximum(1e-10, jnp.linalg.norm(v1))
  e2 = jnp.cross(e0, e1)
  return jnp.stack([e0, e1, e2])


def apply_so3_data_augmentation(
    key: jax.Array,
    dense_atom_positions: jax.Array,
    dense_atom_mask: jax.Array,
    fixed_atom_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Centers and randomly applies rotation to data."""
  assert dense_atom_positions.ndim == 3
  if dense_atom_positions.ndim != 3:
    raise ValueError(f'dense_atom_positions.shape={dense_atom_positions.shape}')

  rotation_key, _ = jax.random.split(key)
  rot = random_rotation_matrix(rotation_key)

  if fixed_atom_mask is not None:
    center_atom_mask = jnp.where(
        jnp.any(fixed_atom_mask), fixed_atom_mask, dense_atom_mask
    )
    center_atom_mask = center_atom_mask * dense_atom_mask
  else:
    center_atom_mask = dense_atom_mask
  dense_atom_positions, center = center_positions(
      atom_positions=dense_atom_positions,
      centering_atom_mask=center_atom_mask,
      nonzero_atom_mask=dense_atom_mask,
  )

  dense_atom_positions_target = jnp.einsum(
      '...i,ij->...j',
      dense_atom_positions,
      rot,
      precision=jax.lax.Precision.HIGHEST,
  )

  translation = jnp.zeros((3,))
  modified_positions = dense_atom_positions_target * dense_atom_mask[..., None]
  return modified_positions, center, rot, translation


def _make_restype_atom37_mask() -> np.ndarray:
  """Mask of which atoms are present for which residue type in atom37."""
  restype_atom37_mask = np.zeros(
      [len(residue_names.PROTEIN_TYPES_ONE_LETTER), 37], dtype=np.float32
  )
  for restype, restype_letter in enumerate(
      residue_names.PROTEIN_TYPES_ONE_LETTER
  ):
    restype_name = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]
    atom_names = atom_types.RESIDUE_ATOMS[restype_name]
    for atom_name in atom_names:
      atom_type = atom_types.ATOM37_ORDER[atom_name]
      restype_atom37_mask[restype, atom_type] = 1
  return restype_atom37_mask.astype(np.int32)


_RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()

# Number of standard amino acid types. Residue types outside [0,
# _NUM_STANDARD_AATYPES) are not amino acids: 20 is UNK, and the sequence
# diffusion mask token and the ligand atom types live above it.
_NUM_STANDARD_AATYPES: Final[int] = len(residue_names.PROTEIN_TYPES_ONE_LETTER)


def _make_backbone_atom37_mask() -> np.ndarray:
  """Mask of the backbone atoms, which are present in every residue type."""
  backbone_atom37_mask = np.zeros(atom_types.ATOM37_NUM, dtype=np.int32)
  for atom_name in atom_types.PROTEIN_BACKBONE_WITH_OXYGEN:
    backbone_atom37_mask[atom_types.ATOM37_ORDER[atom_name]] = 1
  return backbone_atom37_mask


_BACKBONE_ATOM37_MASK = _make_backbone_atom37_mask()


def aatype_from_atom_mask(atom_mask: Any) -> jax.Array:
  """Infers aatypes from atom masks."""
  atom_mask = jax.device_get(atom_mask.astype(int))
  atom_mask_overlap = np.sum(
      atom_mask[:, np.newaxis, :-1] == _RESTYPE_ATOM37_MASK[np.newaxis, :, :-1],
      axis=-1,
  )
  aatype = np.argmax(atom_mask_overlap, axis=-1)
  return jnp.asarray(aatype, dtype=jnp.int32)


def atom_mask_from_aatype(aatype: Any) -> Any:
  """Converts amino acid types to corresponding atom37 masks."""
  aatype_to_atom_mask_array = _RESTYPE_ATOM37_MASK
  atom_format_dim = atom_types.ATOM37_NUM

  num_vocab_tokens = ligand_constants.NUM_PROT_AATYPES
  ligand_atom_mask = np.zeros(atom_format_dim, dtype=np.int32)
  ligand_atom_mask[_LIGAND_ATOM_INDEX] = 1
  padding = num_vocab_tokens - aatype_to_atom_mask_array.shape[0] + 1
  assert padding > 0, 'Original atom mask should not include ligand index.'
  aatype_to_atom_mask_array = np.pad(
      aatype_to_atom_mask_array, ((0, padding), (0, 0)), mode='constant'
  )
  aatype_to_atom_mask_array[-1] = ligand_atom_mask

  if isinstance(aatype, jax.Array):
    aatype_to_atom_mask_array = jnp.asarray(aatype_to_atom_mask_array)
    minimum_fct = jnp.minimum
  else:
    minimum_fct = np.minimum

  array_idx = minimum_fct(aatype, num_vocab_tokens)
  return aatype_to_atom_mask_array[array_idx]


def _placeholder_dist_mask(
    atom_positions: jax.Array,
    atom_mask: jax.Array,
    threshold: float = 1.0,
    placeholder_atom: str = 'CA',
) -> jax.Array:
  """Compute mask of atoms not within threshold distance to placeholder atom."""
  placeholder_idx = atom_types.ATOM37_ORDER[placeholder_atom]
  assert atom_positions.shape[1] == 37
  placeholder_pos = atom_positions[:, placeholder_idx][:, None]
  placeholder_dist = ((atom_positions - placeholder_pos) ** 2).sum(-1) ** 0.5
  if isinstance(placeholder_pos, jax.Array):
    placeholder_dist = placeholder_dist.at[:, placeholder_idx].add(
        2 * threshold
    )
  else:
    placeholder_dist[:, placeholder_idx] += 2 * threshold
  return (placeholder_dist > threshold).astype(float) * atom_mask


def regular_to_super_all_atom(
    atom_positions: Any,
    atom_mask: Any,
    is_ligand_mask: Any | None = None,
    placeholder_atom: str = 'CA',
) -> tuple[Any, Any]:
  """Converts a regular all-atom position/mask to super-all-atom."""
  xnp = _get_np_module(atom_positions)

  placeholder_idx = atom_types.ATOM37_ORDER[placeholder_atom]
  placeholder_pos = atom_positions[..., placeholder_idx, :]
  placeholder_pos = xnp.stack(
      [placeholder_pos] * atom_positions.shape[-2], axis=-2
  )
  new_atom_positions = atom_positions * atom_mask[
      ..., None
  ] + placeholder_pos * (1 - atom_mask[..., None])
  placeholder_atom_mask = atom_mask[..., placeholder_idx][..., None]
  new_atom_mask = atom_mask + placeholder_atom_mask * (1 - atom_mask)

  if is_ligand_mask is not None:
    new_atom_positions = xnp.where(
        is_ligand_mask[..., xnp.newaxis, xnp.newaxis],
        atom_positions,
        new_atom_positions,
    )
    new_atom_mask = xnp.where(
        is_ligand_mask[..., xnp.newaxis],
        atom_mask,
        new_atom_mask,
    )

  return new_atom_positions, new_atom_mask


@dataclasses.dataclass
class SAAParseInfo:
  """Diagnostics produced while parsing a super-all-atom protein.

  Attributes:
    virtual_atom_discrepancy: (L,) Per-residue number of atom37 slots where the
      atom mask parsed from the super-all-atom positions disagrees with the
      idealized atom mask of the residue type inferred from those positions,
      i.e. how cleanly the virtual atom clouds collapsed. Does not depend on
      `use_discrete_aatype`.
    aatype_discrepancy_mask: (L,) Per-residue mask of disagreements between the
      discrete sequence track and the residue types inferred from the virtual
      atom clouds.
    virtual_atom_discrepancy_count: Sum of `virtual_atom_discrepancy`.
    aatype_discrepancy_count: Sum of `aatype_discrepancy_mask`.
    inferred_atom_mask: (L, 37) Atom mask parsed from the super-all-atom
      positions.
    idealized_atom_mask: (L, 37) Atom mask of the emitted residue types, before
      any unsupported atom pruning.
    saa_idealized_atom_mask: (L, 37) Atom mask of the residue types inferred
      from the virtual atom clouds.
    unsupported_atom_mask: (L, 37) Side chain atoms that the emitted residue
      type requires but whose virtual atom never left the placeholder position.
    unsupported_atom_count: Sum of `unsupported_atom_mask`.
    discrete_fallback_mask: (L,) Residues whose discrete sequence token was not
      a standard amino acid, so the inferred residue type was emitted instead.
    discrete_fallback_count: Sum of `discrete_fallback_mask`.
    discrete_aatype: (L,) Residue types from the discrete sequence track.
    saa_aatype: (L,) Residue types inferred from the virtual atom clouds.
  """

  virtual_atom_discrepancy: jnp.ndarray
  aatype_discrepancy_mask: jnp.ndarray
  virtual_atom_discrepancy_count: int
  aatype_discrepancy_count: int
  inferred_atom_mask: jnp.ndarray
  idealized_atom_mask: jnp.ndarray
  saa_idealized_atom_mask: jnp.ndarray
  unsupported_atom_mask: jnp.ndarray
  unsupported_atom_count: int
  discrete_fallback_mask: jnp.ndarray
  discrete_fallback_count: int
  discrete_aatype: jnp.ndarray
  saa_aatype: jnp.ndarray


def parse_super_all_atom_prot(
    prot: Any,
    threshold: float = 1.0,
    placeholder_atom: str = 'CA',
    enforce_input_aatype_for_fixed_residues: bool = True,
    fixed_aatype_mask: jnp.ndarray | None = None,
    diffusion_input_aatype: jnp.ndarray | None = None,
    use_discrete_aatype: bool = True,
    prune_unsupported_atoms: bool = False,
) -> tuple[Any, SAAParseInfo]:
  """Extract the mask from a position-dimension and return a 3D protein.

  Extracts the mask from a position-dimension only for the protein part. If
  is_ligand_mask is passed in the Protein, ligand parts are passed through
  without change.

  Args:
    prot: The "super-all-atom" input Protein object to extract the valid protein
      from.
    threshold: Float-valued threshold to apply to the distance mask computation.
    placeholder_atom: Name of atom to use for distance mask computation.
    enforce_input_aatype_for_fixed_residues: Whether to retain aatype specified
      in diffusion input for motif residues (protein and ligand). Residues with
      motif tip atoms will be considered as fixed residues too.
    fixed_aatype_mask: Fixed aatype mask from the diffusion sampling input.
    diffusion_input_aatype: Specified aatype in the diffusion sampling input.
    use_discrete_aatype: Whether to emit the residue types sampled by the
      discrete sequence track instead of the ones inferred from the virtual atom
      clouds, which degenerate to the residue with the largest heavy-atom
      footprint whenever a cloud fails to collapse. Residues whose discrete
      token is not a standard amino acid (the sequence diffusion mask token, UNK
      or a ligand type on a residue that is not flagged by is_ligand_mask) fall
      back to the inferred residue type. Set to False to derive the sequence
      purely from the virtual atom clouds.
    prune_unsupported_atoms: Whether to drop side chain atoms that the emitted
      residue type requires but whose virtual atom never left the placeholder
      position. The coordinates of such atoms carry no information, so emitting
      them stacks several atoms on top of the placeholder atom. Backbone atoms
      and ligands are never pruned.

  Returns:
    A valid protein with 3 spatial dimensions and a discrete atom mask as
    expected, and an SAAParseInfo containing discrepancies between designed and
    inferred quantities and intermediate results.

  Raises:
    ValueError: If fixed residue aatypes must be enforced but the fixed aatype
      mask or the diffusion input aatype is missing.
  """
  xnp = _get_np_module(prot.atom_positions)

  inferred_atom_mask = _placeholder_dist_mask(
      prot.atom_positions,
      prot.atom_mask,
      threshold=threshold,
      placeholder_atom=placeholder_atom,
  ).astype(xnp.int32)
  saa_aatype = aatype_from_atom_mask(inferred_atom_mask)

  # For ligands, keep the aatype predicted by the model (and not the aatype
  # inferred from the atom positions calculated above). The model's predicted
  # aatype may be different from the specified input aatype.
  is_ligand = None
  if prot.is_ligand_mask is not None:
    is_ligand = prot.is_ligand_mask.astype(bool)
    saa_aatype = xnp.where(is_ligand, prot.aatype, saa_aatype)

  # Prefer the sequence sampled by the discrete track over the one implied by
  # the virtual atom clouds, falling back to the latter for tokens that are not
  # standard amino acids.
  if use_discrete_aatype:
    is_valid_discrete = (prot.aatype >= 0) & (
        prot.aatype < _NUM_STANDARD_AATYPES
    )
    if is_ligand is not None:
      is_valid_discrete = is_valid_discrete | is_ligand
    aatype = xnp.where(is_valid_discrete, prot.aatype, saa_aatype)
    discrete_fallback_mask = (~is_valid_discrete) * prot.sequence_mask
  else:
    aatype = saa_aatype
    discrete_fallback_mask = xnp.zeros_like(prot.sequence_mask)

  # If you are enforcing that the aatype for fixed residues must match the
  # specified input aatype, then retain the specified input aatype for fixed
  # motif (protein and ligand) residues (and tip atoms).
  if enforce_input_aatype_for_fixed_residues:
    if fixed_aatype_mask is None or diffusion_input_aatype is None:
      raise ValueError(
          'Must provide fixed atom mask and diffusion input aatype if you want'
          ' to enforce that the specified input aatype is retained in the'
          ' output protein.'
      )
    aatype = xnp.where(
        fixed_aatype_mask,
        diffusion_input_aatype,
        aatype,
    )
    saa_aatype = xnp.where(
        fixed_aatype_mask,
        diffusion_input_aatype,
        saa_aatype,
    )

  # Set atom mask to the one corresponding to the emitted aatype.
  idealized_atom_mask = atom_mask_from_aatype(aatype)
  # Atom mask corresponding to the aatype inferred from the virtual atom clouds.
  # Keeping it separate is what makes the virtual atom discrepancy below a
  # measure of cloud collapse rather than of sequence-structure agreement.
  saa_idealized_atom_mask = atom_mask_from_aatype(saa_aatype)

  # Adjust aatypes and atom masks with sequence mask.
  aatype = aatype * prot.sequence_mask
  saa_aatype = saa_aatype * prot.sequence_mask
  idealized_atom_mask = idealized_atom_mask * prot.sequence_mask[..., None]
  saa_idealized_atom_mask = (
      saa_idealized_atom_mask * prot.sequence_mask[..., None]
  )

  # Side chain atoms that the emitted residue type requires but whose virtual
  # atom stayed on the placeholder atom. Backbone atoms are exempt because every
  # residue type contains them, and ligands are exempt because their single atom
  # does not follow the placeholder convention.
  unsupported_atom_mask = (
      idealized_atom_mask
      * (1 - inferred_atom_mask)
      * (1 - _BACKBONE_ATOM37_MASK)
  )
  if is_ligand is not None:
    unsupported_atom_mask = xnp.where(
        is_ligand[..., None], 0, unsupported_atom_mask
    )

  if prune_unsupported_atoms:
    emitted_atom_mask = idealized_atom_mask * (1 - unsupported_atom_mask)
  else:
    emitted_atom_mask = idealized_atom_mask

  # Clean atom positions according to the final atom mask.
  atom_positions = prot.atom_positions * emitted_atom_mask[..., None]

  # Discrepancy between the SAA-designed and ideal atom masks.
  virtual_atom_discrepancy = (
      np.sum(saa_idealized_atom_mask != inferred_atom_mask, axis=-1)
      * prot.sequence_mask
  )

  # Discrepancy between discrete and SAA-designed aatypes.
  aatype_discrepancy = (saa_aatype != prot.aatype) * prot.sequence_mask

  aux = SAAParseInfo(
      virtual_atom_discrepancy=virtual_atom_discrepancy,
      aatype_discrepancy_mask=aatype_discrepancy,
      virtual_atom_discrepancy_count=np.sum(virtual_atom_discrepancy),
      aatype_discrepancy_count=np.sum(aatype_discrepancy),
      inferred_atom_mask=inferred_atom_mask,
      idealized_atom_mask=idealized_atom_mask,
      saa_idealized_atom_mask=saa_idealized_atom_mask,
      unsupported_atom_mask=unsupported_atom_mask,
      unsupported_atom_count=np.sum(unsupported_atom_mask),
      discrete_fallback_mask=discrete_fallback_mask,
      discrete_fallback_count=np.sum(discrete_fallback_mask),
      discrete_aatype=prot.aatype,
      saa_aatype=saa_aatype,
  )
  prot = dataclasses.replace(
      prot,
      atom_mask=emitted_atom_mask.astype(prot.atom_mask.dtype),
      atom_positions=atom_positions,
      aatype=aatype.astype(int),
  )
  return prot, aux


ArrayT = types_lib.ArrayT  # pylint: disable=invalid-name
Bool = types_lib.Bool  # pylint: disable=invalid-name
Float = types_lib.Float  # pylint: disable=invalid-name
Int = types_lib.Int  # pylint: disable=invalid-name
L = types_lib.L
typed = types_lib.typed


@typed
def create_relative_encoding(
    residue_index: Int[ArrayT, 'L'],
    asym_id: Int[ArrayT, 'L'],
    max_relative_idx: int,
) -> Float[ArrayT, 'L L Features']:
  """Add relative Position encodings.

  Args:
    residue_index: Linear index of a residue, i.e., can be used to put the
      residues in their correct linear ordering.
    asym_id: Assigns each residue to a residue chain, e.g., to model protein
      complexes.
    max_relative_idx: Cutoff for relative residue distance to consider.

  Returns:
    Relative encoding features.
  """
  rel_feats = []
  asym_id_same = jnp.equal(asym_id[:, None], asym_id[None, :])
  offset = residue_index[:, None] - residue_index[None, :]

  clipped_offset = jnp.clip(
      offset + max_relative_idx, min=0, max=2 * max_relative_idx
  )

  final_offset = jnp.where(
      asym_id_same,
      clipped_offset,
      (2 * max_relative_idx + 1) * jnp.ones_like(clipped_offset),
  )

  rel_pos = jax.nn.one_hot(final_offset, 2 * max_relative_idx + 2)

  rel_feats.append(rel_pos)

  return jnp.concatenate(rel_feats, axis=-1)


@typed
def create_neighbour_encoding(
    residue_index: Int[ArrayT, 'L'],
    asym_id: Int[ArrayT, 'L'],
) -> Float[ArrayT, 'L L 3']:
  """Add neighbour encodings.

  Args:
    residue_index: Linear index of a residue, i.e., can be used to put the
      residues in their correct linear ordering.
    asym_id: Assigns each residue to a residue chain, e.g., to model protein
      complexes.

  Returns:
    One-hot neighbour encoding features:
      [1, 0, 0]: one residue previous
      [0, 1, 0]: self
      [0, 0, 1]: one residue after
      [0, 0, 0]: everything else
  """
  asym_id_same = jnp.equal(asym_id[:, None], asym_id[None, :])
  offset = residue_index[:, None] - residue_index[None, :]
  # {-1, 0, 1} get one-hot encoded, everything else is zeroed out.
  neighbour_one_hot = jax.nn.one_hot(offset + 1, 3)
  return jnp.where(
      asym_id_same[..., None],
      neighbour_one_hot,
      jnp.zeros_like(neighbour_one_hot),
  )


@typed
def random_fourier_encoding(
    x: Float[ArrayT, '*B'],
    dim: int,
) -> Float[ArrayT, '*B D_emb']:
  """Random fourier encoding."""
  w_key, b_key = jax.random.split(jax.random.PRNGKey(42))
  weight = jax.random.normal(w_key, shape=[dim])
  bias = jax.random.uniform(b_key, shape=[dim])
  return jnp.cos(2 * jnp.pi * (x[..., None] * weight + bias))


@typed
def sinusoidal_pos_emb(
    x: Float[ArrayT, '*B'],
    dim: int,
) -> Float[ArrayT, '*B D_emb']:
  """Sinusoidal position embeddings."""
  half_dim = dim // 2
  e = jnp.log(10000) / (half_dim - 1)
  embedding = jnp.exp(-e * jnp.arange(half_dim))
  inputs_rescaled = x * 1000
  embedding = inputs_rescaled * embedding
  embedding = jnp.concatenate([jnp.cos(embedding), jnp.sin(embedding)], axis=-1)
  if dim % 2 == 1:
    embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
  return embedding


@typed
def per_residue_center(
    positions: Float[ArrayT, 'L A 3'],
    atom_mask: Int[ArrayT, 'L A'],
) -> tuple[Float[ArrayT, 'L 3'], Bool[ArrayT, 'L']]:
  """Calculates the per-residue center over where atom_mask=1."""
  # [L A 3]
  positions *= atom_mask[:, :, None]
  # [L 3]
  pos_sum = jnp.sum(positions, axis=-2)
  # [L]
  mask_sum = jnp.sum(atom_mask, axis=-1)
  valid_res = jnp.greater(mask_sum, 0)
  # [L 3]
  res_center = pos_sum / (mask_sum[:, None] + 1e-6)

  return res_center, valid_res


def masked_distogram(
    positions: Float[ArrayT, 'L A 3'],
    atom_mask: Int[ArrayT, 'L A'],
) -> Float[ArrayT, 'L L C']:
  center, valid_res = per_residue_center(positions, atom_mask)
  dgram = cm.dgram_from_positions(center)
  dgram *= valid_res[:, None, None]
  dgram *= valid_res[None, :, None]
  return dgram


def transform_unindexed_motif_coords(
    dinput: types_lib.DiffusionInput,
    center: jax.Array,
    rot: jax.Array,
    translation: jax.Array,
):
  """Apply a centering and rigid transform to unindexed motif coords."""
  if dinput.unindexed_motif_atom_positions is not None:
    new_unindexed_coords = dinput.unindexed_motif_atom_positions - center
    new_unindexed_coords = jnp.einsum(
        '...i,ij->...j',
        new_unindexed_coords,
        rot,
        precision=jax.lax.Precision.HIGHEST,
    )
    new_unindexed_coords += translation
    new_unindexed_coords = (
        new_unindexed_coords * dinput.unindexed_motif_atom_mask[..., None]  # pyrefly: ignore[unsupported-operation]
    )
    dinput = dataclasses.replace(
        dinput,
        unindexed_motif_atom_positions=new_unindexed_coords,
    )
  return dinput


def _residual_mlp(
    act: jax.Array,
    global_config: config_lib.GlobalConfig,
    num_layers: int = 2,
) -> jax.Array:
  """Applies residual LayerNorm -> Linear(4x) -> GELU -> Linear(zeros) blocks."""
  for _ in range(num_layers):
    input_act = act
    act = cm.LayerNorm(global_config)(act)
    act = cm.Linear(
        4 * act.shape[-1],
        global_config=global_config,
        initializer='relu',
    )(act)
    act = jax.nn.gelu(act)
    act = cm.Linear(
        input_act.shape[-1],
        global_config=global_config,
        initializer='zeros',
    )(act)
    act += input_act
  return act


class ConditioningEmbedding(nn.Module):
  """Embed conditioning to a single and pair cond."""

  config: config_lib.ModelBaseConfig
  global_config: config_lib.GlobalConfig

  @nn.compact
  def __call__(
      self,
      inputs: types_lib.DiffusionInput,
  ):
    c = self.config.denoiser
    sigma_data = (
        self.config.sampling_config.structure_sampling_algorithm.sigma_data
    )

    num_tokens = self.global_config.num_vocab_tokens

    self_cond_aatype = inputs.self_cond_aatype
    assert self_cond_aatype is not None
    if self_cond_aatype.shape[-1] != num_tokens:
      self_cond_aatype = jnp.zeros(inputs.protein.aatype.shape + (num_tokens,))

    crop_cond_aatype = inputs.crop_cond_aatype
    fixed_atom_mask = inputs.fixed_atom_mask
    if crop_cond_aatype is None:
      crop_cond_aatype = jnp.zeros_like(inputs.protein.aatype)
    if inputs.fixed_atom_mask is None:
      fixed_atom_mask = jnp.zeros_like(inputs.protein.atom_mask)

    aatype = inputs.protein.aatype
    atom_mask = inputs.protein.atom_mask
    residue_index = inputs.protein.residue_index
    asym_id = inputs.protein.chain_index
    ligand_charge = inputs.ligand_charge
    ligand_hybridization = inputs.ligand_hybridization
    assert inputs.t_struct is not None
    sigma = inputs.t_struct
    scaled_coords_noise_level = sigma / sigma_data
    alpha = jnp.ones_like(inputs.t_struct)
    c_in = 1 / jnp.sqrt((alpha * sigma_data) ** 2 + sigma**2)
    c_in = expand_dims_to(c_in, inputs.protein.atom_positions)
    noised_coords_ = (
        inputs.protein.atom_positions * inputs.protein.atom_mask[..., None]
    )
    scaled_noised_coords = c_in * noised_coords_
    self_cond_coords = inputs.self_cond_atom_positions
    assert self_cond_coords is not None

    rel_features = create_relative_encoding(
        residue_index,
        asym_id,
        max_relative_idx=c.conditioning.max_relative_idx,
    )
    noisy_coords_for_dgram = scaled_noised_coords
    noisy_atom_positions_dgram = masked_distogram(
        noisy_coords_for_dgram, atom_mask
    )
    self_cond_coords_for_dgram = self_cond_coords / sigma_data
    self_conditioning_dgram = masked_distogram(
        self_cond_coords_for_dgram, atom_mask
    )
    rel_features = jnp.concatenate(
        [
            rel_features,
            noisy_atom_positions_dgram,
            self_conditioning_dgram,
        ],
        axis=-1,
    )

    # Mask out features for empty residues.
    seq_mask = jnp.any(atom_mask, axis=-1)
    rel_features_mask = seq_mask[..., None, :] * seq_mask[..., :, None]
    rel_features *= rel_features_mask[..., None]

    pair_cond = cm.Linear(
        c.conditioning.pair_channel, self.global_config, precision='highest'
    )(cm.LayerNorm(self.global_config)(rel_features))
    pair_cond = _residual_mlp(pair_cond, self.global_config)

    def center_and_flatten(positions, atom_mask_):
      # [L 3]
      res_center, _ = per_residue_center(positions, atom_mask_)
      # [L A 3]
      centered = positions - res_center[:, None, :]
      centered *= atom_mask_[..., None]
      # [L A*3]
      return jnp.reshape(centered, centered.shape[:-2] + (-1,))

    single_cond = jnp.zeros((aatype.shape + (c.conditioning.seq_channel,)))
    per_residue_feats = [
        center_and_flatten(scaled_noised_coords, atom_mask),
        center_and_flatten(self_cond_coords, atom_mask),
    ]
    aatype_1h = jax.nn.one_hot(aatype, num_tokens)
    crop_cond_aatype_1h = jax.nn.one_hot(crop_cond_aatype, num_tokens)
    # Set to zero for non fixed residues (otherwise a fixed ALA, which has int
    # aatype zero, is not distinguishable from unfixed residues).
    crop_cond_aatype_1h *= inputs.fixed_seq_mask[..., None]
    features_1d = [
        atom_mask,
        aatype_1h,
        self_cond_aatype,
        crop_cond_aatype_1h,
        fixed_atom_mask,
        *per_residue_feats,
        ligand_charge[..., None],  # "Continuous" embed? or 1h.  # pyrefly: ignore[unsupported-operation]
        jax.nn.one_hot(ligand_hybridization, 8),  # See `bond_constants.py`.
        # Features for research purposes, unused in this model.
        jnp.zeros_like(atom_mask),
    ]
    features_1d = jnp.concatenate(features_1d, axis=-1)
    # Mask out features for empty residues (can have non-zero one-hots).
    features_1d *= seq_mask[..., None]
    single_cond = cm.Linear(
        c.conditioning.seq_channel,
        self.global_config,
        precision='highest',
    )(cm.LayerNorm(self.global_config)(features_1d))

    # Structure noise level position embedding.
    noise_embedding = random_fourier_encoding(
        x=(1 / 4) * jnp.log(scaled_coords_noise_level),
        dim=256,
    )
    seq_noise_embedding = sinusoidal_pos_emb(x=inputs.t_seq, dim=256)
    noise_embedding = jnp.concatenate(
        [noise_embedding, seq_noise_embedding], axis=-1
    )

    single_cond += cm.Linear(
        c.conditioning.seq_channel, self.global_config, precision='highest'
    )(cm.LayerNorm(self.global_config)(noise_embedding))
    single_cond = _residual_mlp(single_cond, self.global_config)

    aux = {'features_1d': features_1d, 'noise_embedding': noise_embedding}
    return single_cond, pair_cond, aux


class UnindexedMotifConditioningEmbedding(nn.Module):
  """Single and pair cond embedding for unindexed motifs."""

  config: config_lib.ModelBaseConfig
  global_config: config_lib.GlobalConfig

  @nn.compact
  def __call__(
      self,
      inputs: types_lib.DiffusionInput,
  ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    sigma_data = (
        self.config.sampling_config.structure_sampling_algorithm.sigma_data
    )
    assert inputs.unindexed_motif_atom_positions is not None
    scaled_coords = inputs.unindexed_motif_atom_positions / sigma_data
    atom_mask = inputs.unindexed_motif_atom_mask
    assert atom_mask is not None

    def center_and_flatten(positions):
      # [L 3]
      res_center, _ = per_residue_center(positions, atom_mask)
      # [L A 3]
      centered = positions - res_center[:, None, :]
      centered *= atom_mask[..., None]
      # [L A*3]
      return jnp.reshape(centered, centered.shape[:-2] + (-1,))

    num_all_tokens = (
        self.global_config.num_protein_tokens
        + self.global_config.num_ligand_tokens
    )
    unindexed_motif_aatype_1h = jax.nn.one_hot(
        inputs.unindexed_motif_aatype, num_all_tokens
    )
    # Set to zero for non fixed residues (otherwise a fixed ALA, which has int
    # aatype zero, is not distinguishable from unfixed residues).
    unindexed_motif_aatype_1h *= inputs.unindexed_motif_aatype_mask[..., None]  # pyrefly: ignore[unsupported-operation]
    features_1d = [
        atom_mask,
        unindexed_motif_aatype_1h,
        center_and_flatten(scaled_coords),
    ]
    features_1d = jnp.concatenate(features_1d, axis=-1)
    aux = {'features_1d': features_1d}
    single_cond = cm.Linear(
        self.config.denoiser.conditioning.seq_channel,
        self.global_config,
        precision='highest',
    )(cm.LayerNorm(self.global_config)(features_1d))
    single_cond = _residual_mlp(single_cond, self.global_config)

    seq_mask = jnp.any(atom_mask, axis=-1)
    single_cond *= seq_mask[..., None]

    neighbour_features = create_neighbour_encoding(
        residue_index=inputs.unindexed_motif_orig_res_id,
        asym_id=inputs.unindexed_motif_orig_chain_index,
    )
    scaled_coords_dgram = masked_distogram(scaled_coords, atom_mask)
    rel_features = jnp.concatenate(
        [
            neighbour_features,
            scaled_coords_dgram,
        ],
        axis=-1,
    )
    # Mask out features for empty residues.
    rel_features_mask = seq_mask[..., None, :] * seq_mask[..., :, None]
    rel_features *= rel_features_mask[..., None]

    pair_cond = cm.Linear(
        self.config.denoiser.conditioning.pair_channel,
        self.global_config,
        precision='highest',
    )(cm.LayerNorm(self.global_config)(rel_features))
    pair_cond = _residual_mlp(pair_cond, self.global_config)

    # Mask out features for empty residues.
    pair_cond *= rel_features_mask[..., None]

    return single_cond, pair_cond, aux


class Denoiser(nn.Module):
  """Single-step protein denoiser network."""

  config: config_lib.ModelBaseConfig
  global_config: config_lib.GlobalConfig

  @nn.compact
  def __call__(
      self, inputs: types_lib.DiffusionInput
  ) -> types_lib.DenoiserOutput:
    num_tokens = self.global_config.num_vocab_tokens
    sigma_data = (
        self.config.sampling_config.structure_sampling_algorithm.sigma_data
    )

    input_coords = (
        inputs.protein.atom_positions * inputs.protein.atom_mask[..., None]
    )
    assert inputs.t_struct is not None
    alpha = jnp.ones_like(inputs.t_struct)
    sigma = inputs.t_struct
    c_in = 1 / jnp.sqrt((alpha * sigma_data) ** 2 + sigma**2)
    c_in = expand_dims_to(c_in, input_coords)
    input_coords = c_in * input_coords

    # Prepare self- and crop-conditioning.
    self_cond_coords = inputs.self_cond_atom_positions
    assert self_cond_coords is not None

    crop_cond_coords = inputs.crop_cond_atom_positions
    if crop_cond_coords is None:
      crop_cond_coords = jnp.zeros_like(inputs.protein.atom_positions)
    crop_cond_coords = crop_cond_coords / sigma_data

    # Get single and pair cond, and noise embedding.
    single_cond, pair_cond, cond_aux = ConditioningEmbedding(
        config=self.config,
        global_config=self.global_config,
    )(
        inputs,
    )

    # Prepare input features.
    num_res, _, _ = input_coords.shape
    act = jnp.concatenate(
        [
            jnp.reshape(input_coords, (num_res, -1)),
            jnp.reshape(self_cond_coords, (num_res, -1)),
            jnp.reshape(crop_cond_coords, (num_res, -1)),
        ],
        axis=-1,
    )
    assert cond_aux['features_1d'] is not None
    act = jnp.concatenate([act, cond_aux['features_1d']], axis=-1)

    # Initial linear projection and run the core neural net.
    act = cm.Linear(
        self.config.denoiser.num_channel,
        self.global_config,
        precision='highest',
    )(act)
    act += cm.Linear(
        act.shape[-1],
        self.global_config,
        precision='highest',
        initializer='zeros',
    )(cm.LayerNorm(self.global_config)(cond_aux['noise_embedding']))

    (
        unindexed_motif_single_cond,
        unindexed_motif_pair_cond,
        unindexed_motif_cond_aux,
    ) = UnindexedMotifConditioningEmbedding(self.config, self.global_config)(
        inputs
    )
    assert inputs.unindexed_motif_atom_positions is not None
    keys_act = jnp.reshape(
        inputs.unindexed_motif_atom_positions / sigma_data,
        (inputs.unindexed_motif_atom_positions.shape[0], -1),
    )
    assert inputs.unindexed_motif_atom_mask is not None
    keys_mask = inputs.unindexed_motif_atom_mask.any(axis=-1).astype(jnp.int32)
    keys_act = jnp.concatenate(
        [keys_act, unindexed_motif_cond_aux['features_1d']], axis=-1
    )
    keys_act = cm.Linear(
        self.config.denoiser.num_channel,
        self.global_config,
        precision='highest',
    )(keys_act)
    keys_act *= keys_mask[..., None]

    net = cm.TransformerWithCrossAttention(
        self.config.denoiser.transformer, self.global_config
    )
    act = net(
        queries=act,
        keys=keys_act,
        queries_mask=inputs.protein.sequence_mask,
        keys_mask=keys_mask,
        single_cond_queries=single_cond,
        single_cond_keys=unindexed_motif_single_cond,
        pair_cond=pair_cond,
        keys_pair_cond=unindexed_motif_pair_cond,
    )

    act = cm.LayerNorm(self.global_config)(act)

    # Predict denoised sequence.
    denoised_aatype_logits = cm.Linear(
        num_tokens,
        self.global_config,
        initializer='linear',
    )(act)

    predicted_motif_indices_mask_logits = cm.Linear(
        2,
        self.global_config,
        initializer='linear',
    )(act)

    f_theta_out = cm.Linear(
        input_coords.shape[-2:],
        self.global_config,
        initializer='zeros',
        precision='highest',
    )(act)
    # [L, A, 1]
    skip_scaling_logits = cm.Linear(
        (input_coords.shape[-2], 1),
        self.global_config,
        initializer='linear',
        precision='highest',
    )(act)
    c_skip = jax.nn.sigmoid(skip_scaling_logits)
    c_out = 1 - c_skip
    denoised_coords = c_skip * input_coords + c_out * f_theta_out
    denoised_coords *= sigma_data

    denoised_coords *= inputs.protein.atom_mask[..., None]

    # Uncertainties for dynamic loss weighting from arxiv:2312.02696.
    coords_uncertainty = cm.Linear(
        1,
        self.global_config,
        precision='highest',
        initializer='relu',
    )(cond_aux['noise_embedding'])
    seq_uncertainty = cm.Linear(
        1,
        self.global_config,
        precision='highest',
        initializer='relu',
    )(cond_aux['noise_embedding'])

    return types_lib.DenoiserOutput(
        denoised_coords=denoised_coords,
        denoised_aatype_logits=denoised_aatype_logits,
        coords_uncertainty=coords_uncertainty,
        seq_uncertainty=seq_uncertainty,
        predicted_motif_indices_mask_logits=predicted_motif_indices_mask_logits,
    )
