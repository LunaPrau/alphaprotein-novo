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

"""Tools for creating ligand features in the data and model pipeline."""

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
import dataclasses
from typing import Any, TypeVar

from alphafold3.constants import chemical_components as chem_comp_constants
from alphafold3.structure import chemical_components
from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.data import data_constants
import jax
import numpy as np
import tree


@dataclasses.dataclass(frozen=True)
class ShapeDtypeSpec:
  """Holds shape and dtype to describe a numpy array."""

  shape: tuple[int, ...]
  dtype: np.dtype


# Defines an interface that exposes `shape` and `dtype` properties.
ArraySpec = TypeVar('ArraySpec', np.ndarray, jax.Array, ShapeDtypeSpec)


def _make_ligand_feature_dict(
    ligand_dict: Mapping[str, np.ndarray],
    num_atoms_per_ligand: int,
    features: MutableMapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
  """Filters the ligand features and pads to a fixed number of atoms.

  Args:
    ligand_dict: A dictionary of features for a single ligand.
    num_atoms_per_ligand: The number of atoms in each ligand. Features are
      cropped / padded to this size.
    features: A dictionary of feature sizes.

  Returns:
    The same dictionary but with atom-like and bond-like features padded to
    fixed numbers of atoms.
  """
  num_atoms = len(ligand_dict['component_atom_names'])

  if num_atoms > num_atoms_per_ligand:
    component_id = ligand_dict['component_id'].item()
    raise ValueError(
        f'Encountered ligand {component_id} with more atoms '
        f'{num_atoms} than the maximum {num_atoms_per_ligand}'
    )

  num_padding_atoms = num_atoms_per_ligand - num_atoms

  padded_ligand = {}
  for feature_name, feature in ligand_dict.items():
    if feature_name not in features:
      continue
    # Drop the leading num ligand dimension which is produced in stacking.
    feature_shapes = features[feature_name][1:]

    # Only pad per-atom features.
    pad_width = [
        (0, num_padding_atoms)
        if size == data_constants.NUM_LIGAND_ATOMS
        else (0, 0)
        for size in feature_shapes
    ]

    if pad_width:
      feature = np.pad(
          feature,
          pad_width=pad_width,
          mode='constant',
          constant_values='0' if feature.dtype == object else 0,
      )

    padded_ligand[feature_name] = feature

  # Padding masks.
  mask_1d = np.arange(num_atoms_per_ligand) < num_atoms

  padded_ligand['component_padding_mask_1d'] = mask_1d.astype(np.int32)
  padded_ligand['component_padding_mask_2d'] = np.logical_and(
      mask_1d[:, None], mask_1d[None, :]
  ).astype(np.int32)

  # Manually set type.
  padded_ligand['component_entity_id'] = padded_ligand[
      'component_entity_id'
  ].astype(np.int32)

  return padded_ligand


def _make_padding_ligand(
    num_atoms_per_ligand: int,
    include_strings: bool,
) -> dict[str, np.ndarray]:
  """Creates a ligand for which the features are all padding.

  Can be used to standardise the number of ligands in a biological assembly.

  Args:
    num_atoms_per_ligand: The number of atoms in each ligand. Features are
      cropped / padded to this size.
    include_strings: If True, add string values to the batch (usually we only
      want this during evaluation).

  Returns:
    A dictionary with all the keys in `config.LIGAND_FEATURES`, pointing to
    arrays with the expected shape and padding values.
  """
  num_atoms = num_atoms_per_ligand
  ligand = {
      'component_atomic_numbers': np.zeros((num_atoms,), dtype=np.int32),
      'component_hybridizations': np.zeros((num_atoms,), dtype=np.int32),
      'component_charges': np.zeros((num_atoms,), dtype=np.float32),
      'component_bond_types': np.zeros((num_atoms, num_atoms), dtype=np.int32),
      'component_bonds_is_aromatic': np.zeros(
          (num_atoms, num_atoms), dtype=np.int32
      ),
      'component_bonds_is_stereo': np.zeros(
          (num_atoms, num_atoms), dtype=np.int32
      ),
      'component_padding_mask_1d': np.zeros((num_atoms,), dtype=np.int32),
      'component_padding_mask_2d': np.zeros(
          (num_atoms, num_atoms), dtype=np.int32
      ),
      'component_fragment_indices': np.zeros((num_atoms,), dtype=np.int32),
      'component_entity_id': np.array(0, dtype=np.int32),
      'component_frame_labels': np.zeros(num_atoms, dtype=np.int32),
  }

  if include_strings:
    ligand.update({
        'component_id': np.array('0', dtype=object),
        'component_smiles_descriptor': np.array('', dtype=object),
        'component_atom_names': np.full(
            shape=(num_atoms,), fill_value='0', dtype=object
        ),
    })
  return ligand


def _create_dense_features_for_ligand(
    atomic_numbers: np.ndarray,
    charges: np.ndarray,
    hybridizations: np.ndarray,
    atom_positions: np.ndarray,
    experimental_mask: np.ndarray,
    b_factors: np.ndarray,
    atom_names: np.ndarray,
) -> MutableMapping[str, np.ndarray]:
  """Create dense_atom features for a ligand fragment.

  Args:
    atomic_numbers: for each of the atoms in the molecule / fragment from CCD.
    charges: formal electronic charge for each atom.
    hybridizations: for each atom, eg `sp3` or `sp3d2`.
    atom_positions: x,y,z coordinates for each atom in the molecule / fragment.
      shape=(num_atoms, 3)
    experimental_mask: whether atom coordinates are experimentally determined.
    b_factors: b factors for each fragment.
    atom_names: names of the atoms, eg 'NO1', 'O11', 'O21'.

  Returns:
    dict containing ligand features.
  """
  aa_type = np.array(
      residue_names.PROTEIN_TYPES_ONE_LETTER_TO_INT['G'], dtype=np.int32
  )

  if len(atomic_numbers) > atom_types.ATOM37_NUM:
    raise ValueError(
        f'Ligand residue has too many atoms ({len(atomic_numbers)}) '
        f'for atom_types.ATOM37_NUM ({atom_types.ATOM37_NUM}).'
    )

  pad_width = atom_types.ATOM37_NUM - len(atomic_numbers)

  def pad_1d(feat):
    return np.pad(
        feat, pad_width=(0, pad_width), mode='constant', constant_values=0
    )

  def pad_1d_string(feat):
    return np.pad(
        feat,
        pad_width=(0, pad_width),
        mode='constant',
        constant_values='',
    )

  padded_atomic_numbers = pad_1d(atomic_numbers)
  padded_charges = pad_1d(charges)
  padded_hybridizations = pad_1d(hybridizations)
  padded_atom_names = pad_1d_string(atom_names)

  # Atom exists from CCD.
  pred_dense_atom_mask = np.arange(atom_types.ATOM37_NUM) < len(atomic_numbers)

  # Atom exists in this crystal.
  gt_dense_atom_mask = np.pad(
      experimental_mask,
      pad_width=(0, pad_width),
      mode='constant',
      constant_values=0,
  )
  dense_atom_pos = np.pad(
      atom_positions,
      pad_width=((0, pad_width), (0, 0)),
      mode='constant',
      constant_values=0,
  )
  dense_b_factors = np.pad(
      b_factors,
      pad_width=(0, pad_width),
      mode='constant',
      constant_values=0,
  )

  # Pad dense representation to atom37.
  extra_pad_width = 37 - atom_types.ATOM37_NUM
  atom37_pos = np.pad(
      dense_atom_pos,
      pad_width=((0, extra_pad_width), (0, 0)),
      mode='constant',
      constant_values=0,
  )
  gt_atom37_mask = np.pad(
      gt_dense_atom_mask,
      pad_width=(0, extra_pad_width),
      mode='constant',
      constant_values=0,
  )
  atom37_b_factors = np.pad(
      dense_b_factors,
      pad_width=(0, extra_pad_width),
      mode='constant',
      constant_values=0,
  )
  pred_atom37_mask = np.pad(
      pred_dense_atom_mask,
      pad_width=(0, extra_pad_width),
      mode='constant',
      constant_values=0,
  )
  atom37_atom_names = np.pad(
      padded_atom_names,
      pad_width=(0, extra_pad_width),
      mode='constant',
      constant_values='',
  )

  return {
      'aatype': aa_type,
      'all_atom_positions': atom37_pos.astype(np.float32),
      'all_atom_mask': gt_atom37_mask.astype(np.int32),  # Present in PDB model.
      'b_factors': atom37_b_factors.astype(np.float32),
      'dense_atom_gt_positions': dense_atom_pos.astype(np.float64),
      'dense_atom_gt_mask': gt_dense_atom_mask.astype(np.float32),
      'ligand_atomic_number': padded_atomic_numbers.astype(np.int32),
      'ligand_charge': padded_charges.astype(np.int32),
      'ligand_hybridization': padded_hybridizations.astype(np.int32),
      'pred_dense_atom_mask': pred_dense_atom_mask.astype(np.float32),
      'pred_atom37_mask': pred_atom37_mask.astype(np.int32),
      'is_protein': np.zeros_like(aa_type, dtype=np.int32),
      'ligand_atom_names': atom37_atom_names.astype(object),
  }


def pad_ligand_features(
    ligands: Iterable[MutableMapping[str, np.ndarray | jax.Array]],
    feature_shapes: Mapping[str, Sequence[str | None]],
    num_ligands: int,
    num_atoms_per_ligand: int,
) -> list[Mapping[str, np.ndarray | jax.Array]]:
  """Pads ligand features to specified size.

  Args:
    ligands: Feature dict for each ligand.
    feature_shapes: A dictionary containing at least all ligand feature shapes.
    num_ligands: Number of ligands to pad to.
    num_atoms_per_ligand: Number of ligand atoms to pad to.

  Returns:
    An iterable of padded ligands.
  """
  # Identify the per ligand features using the leading NUM_LIGANDS dimension.
  ligand_feature_shapes = {
      name: shape
      for name, shape in feature_shapes.items()
      if shape[:1] == [data_constants.NUM_LIGANDS]
  }

  padded_ligands = []
  for ligand in ligands:
    # Pad to fixed atom number and produce AA like keys.
    padded_ligands.append(
        _make_ligand_feature_dict(
            ligand, num_atoms_per_ligand, ligand_feature_shapes  # pyrefly: ignore[bad-argument-type]
        )
    )

  # Too few ligands -> pad with 0 or more complete padding ligands up to a fixed
  # size. The padding masks are themselves padded consistently.
  if len(padded_ligands) < num_ligands:
    num_padding_ligands = num_ligands - len(padded_ligands)
    padding_ligand = _make_padding_ligand(
        num_atoms_per_ligand,
        include_strings='component_id' in ligand_feature_shapes,
    )

    padded_ligands.extend([padding_ligand] * num_padding_ligands)

  return padded_ligands


def add_chain_representation_to_ligands(
    ligands: list[MutableMapping[str, np.ndarray]],
    protein_array_spec_dict: MutableMapping[str, ArraySpec],
) -> list[MutableMapping[str, Any]]:
  """Adds per-ligand-residue features to ligand dictionaries under 'chain'.

  This allows per-residue features to be merged in with the AA chains.

  Args:
    ligands: data3 feature dicts for each ligand to include.
    protein_array_spec_dict: Maps feature names of a protein chain to an array
      specification that defines shape and dtype.

  Returns:
    The input `ligands` dictionaries, each with a new key 'chain' containing
    features equivalent to those for an AA chain.
  """
  # Only interested in keys which are present in the example dictionary,
  # because only those will be merged.
  chain_keys = set(protein_array_spec_dict)

  for ligand in ligands:
    residues_for_ligand = {}
    # Number of residues in ligand chain is equal to number of fragments.
    ligand_fragment_indices = ligand['component_fragment_indices']
    num_fragments = len(np.unique(ligand_fragment_indices))

    # Sequence features.
    for key in set(data_constants.SEQ_FEATURES) & chain_keys:
      # Take and tile the ligand value if it's in the dictionary.
      ligand_key = 'component_' + key
      if ligand_key in ligand:
        residues_for_ligand[key] = np.repeat(ligand[ligand_key], num_fragments)
      # Or fill with zeros.
      else:
        shape = list(protein_array_spec_dict[key].shape)
        dtype = protein_array_spec_dict[key].dtype
        shape[0] = num_fragments
        residues_for_ligand[key] = np.zeros(shape, dtype)

    # Manually set some features.
    residues_for_ligand['residue_index'] = np.arange(
        num_fragments, dtype=np.int32
    )
    residues_for_ligand['seq_length'] = np.array(num_fragments)
    residues_for_ligand['num_alignments'] = np.array(1)
    residues_for_ligand['sequence'] = np.array('X', dtype=object)

    # AA type and dense atom positions.
    aa_features = []
    for fragment_index in np.unique(ligand_fragment_indices):
      ligand_fragment_mask = ligand_fragment_indices == fragment_index

      aa_features.append(
          _create_dense_features_for_ligand(
              ligand['component_atomic_numbers'][ligand_fragment_mask],
              ligand['component_charges'][ligand_fragment_mask],
              ligand['component_hybridizations'][ligand_fragment_mask],
              ligand['component_atom_positions'][ligand_fragment_mask],
              ligand['component_experimental_mask'][ligand_fragment_mask],
              ligand['component_b_factors'][ligand_fragment_mask],
              ligand['component_atom_names'][ligand_fragment_mask],
          )
      )

    # Stack into (num_ligand_residues,)
    stacked_aa_features = tree.map_structure(
        lambda *ts: np.stack(ts), *aa_features
    )

    # Take 0th atom37 index for ligand features.
    for feat in [
        'ligand_charge',
        'ligand_hybridization',
        'ligand_atomic_number',
        'ligand_atom_names',
    ]:
      stacked_aa_features[feat] = stacked_aa_features[feat][..., 0]
    residues_for_ligand.update(stacked_aa_features)

    # Insert this chain into the ligand dictionary.
    ligand['chain'] = residues_for_ligand  # pyrefly: ignore[unsupported-operation]

  return ligands


def add_is_ligand_mask(
    np_chains_list: Iterable[MutableMapping[str, np.ndarray]],
    np_ligands: Iterable[MutableMapping[str, Any]],
) -> tuple[
    Iterable[MutableMapping[str, np.ndarray]],
    Iterable[MutableMapping[str, Any]],
]:
  """Adds a per-residue mask with zeros for each chain and ones for each ligand.

  Args:
    np_chains_list: An iterable of feature dictionaries, 1 for each AA chain.
    np_ligands: An iterable of nested feature dictionaries, 1 for each ligand.

  Returns:
    The same data structures as the inputs but with a new key 'is_ligand_mask'
    added to each entry in np_chains_list and each np_ligand['chain'].
  """
  # Chains.
  for chain in np_chains_list:
    chain['is_ligand_mask'] = np.zeros(len(chain['asym_id']), dtype=np.int32)

  # Ligands.
  for ligand in np_ligands:
    ligand['chain']['is_ligand_mask'] = np.ones(
        len(ligand['chain']['asym_id']), dtype=np.int32
    )

  return np_chains_list, np_ligands


def get_ligand_ccd_atom_names(
    ccd_code: str,
    chemical_components_data: (
        chemical_components.ChemicalComponentsData | None
    ) = None,
    default_atom_names: Sequence[str] | None = None,
) -> list[str]:
  """Retrieves list of a ligand's atom names based on its CCD code."""
  if (
      chemical_components_data is not None
      and ccd_code in chemical_components_data.chem_comp
  ):
    entry = chemical_components_data.chem_comp[ccd_code]
    if entry.chem_comp_atoms is not None:
      return sorted([
          name
          for name, atom in entry.chem_comp_atoms.items()
          if atom.type_symbol not in {'H', 'D'}
      ])
  ccd = chem_comp_constants.Ccd()
  if ccd_code in ccd:
    res_atoms = chemical_components.get_all_atoms_in_entry(ccd, ccd_code)
    atom_ids = res_atoms.get('_chem_comp_atom.atom_id', [])
    type_symbols = res_atoms.get('_chem_comp_atom.type_symbol', [])
    return sorted([
        atom_id
        for atom_id, type_symbol in zip(atom_ids, type_symbols)
        if type_symbol not in {'H', 'D'}
    ])
  if default_atom_names is not None:
    # Upper-case before sorting so that the resulting order matches the order
    # derived from the CCD / SMILES graph, which upper-cases element symbols
    # (e.g. a structure's 'Fe1' / 'Cl1' become 'FE1' / 'CL1'). Sorting the raw,
    # mixed-case names can otherwise yield a different order, which would
    # misalign ligand tokens against their coordinates.
    return sorted(name.upper() for name in default_atom_names)
  raise ValueError(f'Unknown CCD code or missing component data: {ccd_code}')
