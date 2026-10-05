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

"""Functions for getting features from a Structure / the CCD.

CCD stands for the Chemical Components Dictionary
https://www.ebi.ac.uk/pdbe-srv/pdbechem/doc/chem_comp/help.htm
"""

import collections
from collections.abc import Mapping, MutableMapping, Sequence
import enum
import functools
import threading
from typing import Any

from absl import logging
from alphafold3 import structure
from alphafold3.constants import chemical_components as chem_comp_constants
from alphafold3.data.tools import rdkit_utils
from alphafold3.structure import chemical_components
from alphaprotein_novo.constants import periodic_table
import numpy as np
import rdkit.Chem as rd_chem

FeatureDict = MutableMapping[str, Any]
Structure = structure.Structure


class BondType(enum.IntEnum):
  NO_BOND = 0
  SINGLE = 1
  DOUBLE = 2
  TRIPLE = 3


class BondStereo(enum.IntEnum):
  NO_BOND = 0
  OPPOSITE = 1
  NONE = 2
  TOGETHER = 3


class BondAromatic(enum.IntEnum):
  NO_BOND = 0
  AROMATIC = 1
  NOT_AROMATIC = 2


class HybridizationType(enum.IntEnum):
  UNSPECIFIED = 0
  S = 1
  SP = 2
  SP2 = 3
  SP3 = 4
  SP3D = 5
  SP3D2 = 6
  OTHER = 7


_INT_FROM_BOND_TYPE = {'SING': 1, 'DOUB': 2, 'TRIP': 3}
_INT_FROM_BOND_IS_STEREO = {'N': 2, 'E': 1, 'Z': 3}
_INT_FROM_BOND_IS_AROMATIC = {'N': 2, 'Y': 1, '?': 0}

_CCD = chem_comp_constants.Ccd()
_RDKIT_LOCK = threading.Lock()


def _locked_rdkit(func, *args, **kwargs):
  """Calls RDKit function under lock as RDKit is not fully thread-safe."""
  with _RDKIT_LOCK:
    return func(*args, **kwargs)


@functools.lru_cache(maxsize=5000)
def _get_ccd_mol_data_cached(
    comp_id: str,
) -> tuple[bytes | None, Mapping[str, Sequence[str]] | None]:
  """Cached retrieval of CCD molecule data."""
  if comp_id not in _CCD:
    return None, None
  mol_mmcif = _CCD[comp_id]
  try:
    mol = _locked_rdkit(
        rdkit_utils.mol_from_ccd_cif,
        mol_mmcif,
        force_parse=True,
        sort_alphabetically=True,
    )
    mol_bin = mol.ToBinary() if mol is not None else None
  except rdkit_utils.MolFromMmcifError as e:
    logging.warning(
        'Failed to parse CCD component: %s. Error: %s. '
        'Certain ligand features will be populated with sensible defaults.',
        comp_id,
        repr(e),
    )
    mol_bin = None
  return mol_bin, mol_mmcif


def get_ligand_features(
    chain: Structure,
    force_use_pdbx_smiles: bool = False,
) -> FeatureDict:
  """Returns a standard set of features for a non-polymer ligand chain.

  Args:
    chain: Input chain. Must be non-polymer, implying a single component.
    force_use_pdbx_smiles: Whether to force ligand features to be calculated
      based on the molecule object constructed from the SMILES string, even if
      the ligand is coming from CCD and certain features could be calculated
      directly from mmCIF content.
  """
  comp_id = chain['res_name'][0]
  mol, mol_mmcif = _get_chemical_component_mol_data(
      comp_id, chain.chemical_components_data, force_use_pdbx_smiles  # pyrefly: ignore[bad-argument-type]
  )

  # Only this atom order is supported - it's hardcoded inside ligand_utils.
  atom_order = _get_sorted_ccd_heavy_atom_order(mol_mmcif)
  atom_names = sorted([(idx, atom) for atom, idx in atom_order.items()])
  atom_names = [atom for _, atom in atom_names]

  chain_atoms = list(chain.iter_atoms())
  observed_atoms = _get_experimental_atom_data(
      chain_atoms, atom_order, logging_name=chain.name
  )
  ccd_atoms = _get_ccd_atom_data(mol_mmcif, atom_order, logging_name=chain.name)
  ccd_bonds = _get_ccd_bond_data(mol_mmcif, atom_order, logging_name=chain.name)

  num_non_h_atoms = len(atom_order)
  hybridizations = _get_ligand_hybridization_features(
      mol, default_num_atoms=num_non_h_atoms
  )

  fragments = np.arange(num_non_h_atoms)
  # Choose which ligands define the frames for their fragments.
  frame_labels = _assign_default_frame_labels(fragments)

  smiles_descriptor = _get_smiles_descriptor(mol_mmcif)

  component_atom_positions = observed_atoms['positions']
  component_experimental_mask = observed_atoms['mask']

  return {
      'component_id': np.array(comp_id, dtype=object),
      'component_smiles_descriptor': np.array(smiles_descriptor, dtype=object),
      'component_atom_names': np.array(atom_names, dtype=object),
      'component_atom_positions': component_atom_positions,
      'component_experimental_mask': component_experimental_mask,
      'component_b_factors': observed_atoms['b_factors'],
      'component_charges': ccd_atoms['charges'],
      'component_atomic_numbers': ccd_atoms['atomic_numbers'],
      'component_bond_types': ccd_bonds['types'],
      'component_bonds_is_stereo': ccd_bonds['stereo'],
      'component_bonds_is_aromatic': ccd_bonds['aromatic'],
      'component_fragment_indices': fragments,
      'component_hybridizations': hybridizations,
      'component_frame_labels': frame_labels,
      'component_chain_id': np.array([chain.chain_id[0]], dtype=object),
  }


def _upper_case_atom_order(atom_order: Mapping[str, int]) -> Mapping[str, int]:
  """Returns `atom_order` re-keyed by upper-cased atom name.

  Atom names whose upper-cased form is ambiguous (i.e. shared with another,
  differently-cased atom name in `atom_order`) are omitted, so that the result
  can only ever be used to resolve an unambiguous match.

  Args:
    atom_order: Maps atom name to an integer index.
  """
  upper_counts = collections.Counter(name.upper() for name in atom_order)
  return {
      name.upper(): index
      for name, index in atom_order.items()
      if upper_counts[name.upper()] == 1
  }


def _get_experimental_atom_data(
    component: Sequence[Mapping[str, Any]],
    atom_order: Mapping[str, int],
    atom_dim_size: int | None = None,
    max_b_factor: float | None = None,
    remapped_component_id: str | None = None,
    logging_name: str | None = None,
) -> FeatureDict:
  """Returns experimentally resolved atom data for a single component.

  Args:
    component: A list of Structure atom dictionaries for the component.
    atom_order: Defines the fixed atom order to use for this component. Each
      atom name is mapped to an integer index. Atom names that do not match
      exactly fall back to an unambiguous case-insensitive match. Atoms not
      included in this mapping will be dropped.
    atom_dim_size: If specified, the atoms dimension will be padded to this
      size. Otherwise, the size of the atoms dimension will be inferred based on
      the largest index appearing in atom_order.
    max_b_factor: If specified, atoms with b_factor > max_b_factor are dropped.
    remapped_component_id: Indicates that this component is being interpreted as
      something else (a typical example is MSE being remapped to MET).
    logging_name: A meaningful name for this call used in error messages.

  Returns:
    A FeatureDict providing:
    - positions. Shape [atom_dim_size, 3].
    - mask. Shape [atom_dim_size].
    - b_factors. Shape [atom_dim_size].

  Raises:
    ValueError: If an index larger than atom_dim_size appears in atom_order, or
      if atom_dim_size is not specified and atom_order is empty.
  """
  if atom_dim_size and atom_order:
    if max(atom_order.values()) >= atom_dim_size:
      raise ValueError(f'{logging_name}: in {component[0]["res_name"]}')

  if not atom_dim_size:
    if atom_order:
      atom_dim_size = max(atom_order.values()) + 1
    else:
      raise ValueError((
          f'{logging_name}: in {component[0]["res_name"]}'
          'Cannot infer atom dimension size: empty atom order.'
      ))

  positions = np.zeros((atom_dim_size, 3), dtype=np.float32)
  mask = np.zeros(atom_dim_size, dtype=np.int32)
  b_factors = np.zeros(atom_dim_size, dtype=np.float32)

  # Built lazily, only if some atom name fails to match exactly.
  upper_case_atom_order: Mapping[str, int] | None = None

  for atom in component:
    if max_b_factor and atom['atom_b_factor'] > max_b_factor:
      continue

    atom_name = atom['atom_name']
    # Special case only for an MSE that's been remapped to MET: treat SE as SD.
    if (
        remapped_component_id
        and remapped_component_id == 'MET'
        and atom_name == 'SE'
    ):
      atom_name = 'SD'
    if atom_name in atom_order:
      a_idx = atom_order[atom_name]
    else:
      # Structures may name atoms with the element symbol cased as in the
      # periodic table (e.g. 'Fe1', 'Cl1'), whereas `atom_order` is derived
      # from the CCD / SMILES graph, which upper-cases the element symbol
      # ('FE1', 'CL1'). Fall back to an unambiguous case-insensitive match so
      # such atoms are not silently dropped.
      if upper_case_atom_order is None:
        upper_case_atom_order = _upper_case_atom_order(atom_order)
      a_idx = upper_case_atom_order.get(atom_name.upper())
      if a_idx is None:
        continue
      logging.warning(
          '%s: in %s, atom %r was matched case-insensitively to the canonical'
          ' atom name %r.',
          logging_name,
          component[0]['res_name'],
          atom_name,
          atom_name.upper(),
      )

    positions[a_idx, :] = np.array(
        [atom['atom_x'], atom['atom_y'], atom['atom_z']], dtype=np.float32
    )
    mask[a_idx] = 1
    b_factors[a_idx] = atom['atom_b_factor']

  return {'positions': positions, 'mask': mask, 'b_factors': b_factors}


def _get_ccd_atom_data(
    ccd: Mapping[str, Sequence[str]],
    atom_order: Mapping[str, int],
    atom_dim_size: int | None = None,
    logging_name: str | None = None,
) -> FeatureDict:
  """Returns atom data extracted from the CCD entry for a component.

  Args:
    ccd: A parsed Chemical Components Dictionary entry.
    atom_order: Defines the fixed atom order to use for this component. Each
      atom name is mapped to an integer index. Atoms not included in this
      mapping will be dropped.
    atom_dim_size: If specified, the atoms dimension will be padded to this
      size. Otherwise, the size of the atoms dimension will be inferred based on
      the largest index appearing in atom_order.
    logging_name: A meaningful name for this call used in error messages.

  Returns:
    A FeatureDict providing:
    - positions. Shape [atom_dim_size, 3]. Default to 0 if ideal conformer
      positions are not provided by the mmCIF.
    - atomic_numbers. Shape [atom_dim_size]. Default -1 if element not in our
      map.
    - charges. Shape [atom_dim_size]. Default 0.

  Raises:
    ValueError: If an index larger than atom_dim_size appears in atom_order, or
      if atom_dim_size is not specified and atom_order is empty.
  """
  if atom_dim_size and atom_order:
    if max(atom_order.values()) >= atom_dim_size:
      raise ValueError(f'{logging_name}: in CCD component')

  if not atom_dim_size:
    if atom_order:
      atom_dim_size = max(atom_order.values()) + 1
    else:
      raise ValueError((
          f'{logging_name}: in CCD component.'
          'Cannot infer atom dimension size: empty atom order.'
      ))

  atom_ids = ccd.get('_chem_comp_atom.atom_id', [])
  type_symbols = ccd.get('_chem_comp_atom.type_symbol', [])
  charges_raw = ccd.get('_chem_comp_atom.charge', [])

  atomic_numbers = np.zeros(atom_dim_size, dtype=np.int32)
  charges = np.zeros(atom_dim_size, dtype=np.float32)

  for name, sym, chg in zip(atom_ids, type_symbols, charges_raw):
    if name not in atom_order:
      continue
    atom_idx = atom_order[name]
    element = sym.capitalize()
    atomic_numbers[atom_idx] = periodic_table.ATOMIC_NUMBER.get(element, -1)
    charges[atom_idx] = float(chg) if chg != '?' else 0.0

  positions = np.zeros((atom_dim_size, 3), dtype=np.float32)
  ideal_x = ccd.get('_chem_comp_atom.pdbx_model_Cartn_x_ideal', [])
  ideal_y = ccd.get('_chem_comp_atom.pdbx_model_Cartn_y_ideal', [])
  ideal_z = ccd.get('_chem_comp_atom.pdbx_model_Cartn_z_ideal', [])

  for name, x, y, z in zip(atom_ids, ideal_x, ideal_y, ideal_z):
    if name not in atom_order:
      continue
    coords = (x, y, z)
    if '?' in coords:
      positions = np.zeros((atom_dim_size, 3), dtype=np.float32)
      break
    positions[atom_order[name]] = (float(x), float(y), float(z))

  return {
      'positions': positions,
      'atomic_numbers': atomic_numbers,
      'charges': charges,
  }


def _get_ccd_bond_data(
    ccd: Mapping[str, Sequence[str]],
    atom_order: Mapping[str, int],
    atom_dim_size: int | None = None,
    logging_name: str | None = None,
) -> FeatureDict:
  """Returns bond data extracted from the CCD entry for a component.

  Args:
    ccd: A parsed Chemical Components Dictionary entry.
    atom_order: Defines the fixed atom order to use for this component. Each
      atom name is mapped to an integer index. Atoms not included in this
      mapping will be dropped.
    atom_dim_size: If specified, the atoms dimension will be padded to this
      size. Otherwise, the size of the atoms dimension will be inferred based on
      the largest index appearing in atom_order.
    logging_name: A meaningful name for this call used in error messages.

  Returns:
    A FeatureDict providing bond data as adjacency matrices of shape
    [atom_dim_size, atom_dim_size]:
    - types. Integer code indicates unbonded / single / double / triple.
    - stereo. Integer code indicates unbonded / opposite / none / together.
    - aromatic. Integer code indicates unbonded / aromatic / not aromatic.

  Raises:
    ValueError: If an index larger than atom_dim_size appears in atom_order, or
      if atom_dim_size is not specified and atom_order is empty.
  """
  if atom_dim_size and atom_order:
    if max(atom_order.values()) >= atom_dim_size:
      raise ValueError(f'{logging_name}: in CCD component')

  if not atom_dim_size:
    if atom_order:
      atom_dim_size = max(atom_order.values()) + 1
    else:
      raise ValueError((
          f'{logging_name}: in CCD component.'
          'Cannot infer atom dimension size: empty atom order.'
      ))

  types = np.full(
      (atom_dim_size, atom_dim_size),
      BondType.NO_BOND.value,
      dtype=np.int32,
  )
  stereo = np.full(
      (atom_dim_size, atom_dim_size),
      BondStereo.NO_BOND.value,
      dtype=np.int32,
  )
  aromatic = np.full(
      (atom_dim_size, atom_dim_size),
      BondAromatic.NO_BOND.value,
      dtype=np.int32,
  )

  atom_1_list = ccd.get('_chem_comp_bond.atom_id_1', [])
  atom_2_list = ccd.get('_chem_comp_bond.atom_id_2', [])
  order_list = ccd.get('_chem_comp_bond.value_order', [])
  stereo_list = ccd.get('_chem_comp_bond.pdbx_stereo_config', [])
  aromatic_list = ccd.get('_chem_comp_bond.pdbx_aromatic_flag', [])

  for a1, a2, vo, st, ar in zip(
      atom_1_list, atom_2_list, order_list, stereo_list, aromatic_list
  ):
    a_idx_1 = atom_order.get(a1)
    a_idx_2 = atom_order.get(a2)
    if a_idx_1 is None or a_idx_2 is None:
      continue
    b_type = _INT_FROM_BOND_TYPE[vo]
    b_stereo = _INT_FROM_BOND_IS_STEREO[st]
    b_aromatic = _INT_FROM_BOND_IS_AROMATIC[ar]
    for i, j in ((a_idx_1, a_idx_2), (a_idx_2, a_idx_1)):
      types[i][j] = b_type
      stereo[i][j] = b_stereo
      aromatic[i][j] = b_aromatic
  return {'types': types, 'stereo': stereo, 'aromatic': aromatic}


def _get_chemical_component_mol_data(
    comp_id: str,
    chemical_components_data: (
        chemical_components.ChemicalComponentsData | None
    ) = None,
    force_use_pdbx_smiles: bool = False,
) -> tuple[rd_chem.Mol | None, Mapping[str, Sequence[str]]]:
  """Returns RDKit Mol and mmCIF representation of a chemical component."""
  # Using mmCIF as a primary representation for chemical components from CCD
  # unless otherwise requested via `force_use_pdbx_smiles`.
  if not force_use_pdbx_smiles:
    mol_bin, mol_mmcif = _get_ccd_mol_data_cached(comp_id)
    if mol_mmcif is not None:
      if mol_bin is not None:
        mol = rd_chem.Mol(mol_bin)
      else:
        mol = None
      return mol, mol_mmcif

  # Otherwise using RDKit Mol constructed from a SMILES string as a primary
  # representation and constructing mmCIF out of it.
  if (
      not chemical_components_data
      or comp_id not in chemical_components_data.chem_comp
  ):
    raise ValueError(f'ChemicalComponentsData for {comp_id} is not found.')
  chem_comp_entry = chemical_components_data.chem_comp[comp_id]

  if not chem_comp_entry.pdbx_smiles:
    raise ValueError(f'SMILES string for {comp_id} is not found.')

  mol = _locked_rdkit(rd_chem.MolFromSmiles, chem_comp_entry.pdbx_smiles)
  if mol is None:
    raise ValueError(
        'Fail to construct RDKit Mol from the SMILES string: '
        f'{chem_comp_entry.pdbx_smiles}.'
    )

  # Sort atom names to match behavior of _get_ccd_mol_data_cached().
  mol = _locked_rdkit(rdkit_utils.assign_atom_names_from_graph, mol)
  mol = _locked_rdkit(rdkit_utils.sort_atoms_by_name, mol)

  try:
    mol_mmcif = _locked_rdkit(
        rdkit_utils.mol_to_ccd_cif,
        mol,
        comp_id,
        pdbx_smiles=chem_comp_entry.pdbx_smiles,
        include_hydrogens=True,
    )
  except rdkit_utils.UnsupportedMolBondError as e:
    raise ValueError(
        'Fail to construct mmCIF from the SMILES string: '
        f'{chem_comp_entry.pdbx_smiles}.'
    ) from e

  return mol, mol_mmcif


def _get_sorted_ccd_heavy_atom_order(
    ccd: Mapping[str, Sequence[str]],
) -> Mapping[str, int]:
  """Returns an atom order based on sorted CCD atom names, without hydrogens."""
  atom_ids = ccd.get('_chem_comp_atom.atom_id', [])
  type_symbols = ccd.get('_chem_comp_atom.type_symbol', [])
  atoms = list(zip(atom_ids, type_symbols))
  heavy_atoms = [name for name, elt in atoms if elt not in {'H', 'D'}]
  return {atom: idx for idx, atom in enumerate(sorted(heavy_atoms))}


def _get_ligand_hybridization_features(
    mol: rd_chem.Mol | None,
    default_num_atoms: int,
) -> np.ndarray:
  """Returns hybridization features for the ligand molecule."""

  if mol is not None:
    mol = _locked_rdkit(rd_chem.RemoveAllHs, mol)

    hybridizations = []
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtomWithIdx(atom_idx)
      rd_hybridization = atom.GetHybridization()
      if rd_hybridization == rd_chem.HybridizationType.S:
        hybridizations.append(HybridizationType.S)
      elif rd_hybridization == rd_chem.HybridizationType.SP:
        hybridizations.append(HybridizationType.SP)
      elif rd_hybridization == rd_chem.HybridizationType.SP2:
        hybridizations.append(HybridizationType.SP2)
      elif rd_hybridization == rd_chem.HybridizationType.SP3:
        hybridizations.append(HybridizationType.SP3)
      elif rd_hybridization == rd_chem.HybridizationType.SP3D:
        hybridizations.append(HybridizationType.SP3D)
      elif rd_hybridization == rd_chem.HybridizationType.SP3D2:
        hybridizations.append(HybridizationType.SP3D2)
      elif rd_hybridization == rd_chem.HybridizationType.OTHER:
        hybridizations.append(HybridizationType.OTHER)
      else:
        hybridizations.append(HybridizationType.UNSPECIFIED)
    return np.array(hybridizations).astype(np.int32)

  return np.zeros(default_num_atoms, dtype=np.int32)


def _get_smiles_descriptor(mol_mmcif: Mapping[str, Sequence[str]]) -> str:
  pdbx_smiles_column = mol_mmcif.get('_chem_comp.pdbx_smiles', [])
  if not pdbx_smiles_column:
    return ''
  return pdbx_smiles_column[0]


def _assign_default_frame_labels(fragment_indices: np.ndarray) -> np.ndarray:
  """Labels every atom to indicate its role in frame construction.

  Atoms labelled with 0, 1, 2 are used to construct the frame for the fragement
    they belong to. Atoms labelled with 3 are not.

  Args:
    fragment_indices: A numpy array of integers labelling each atom with its
      fragment.

  Returns:
    An integer numpy array with shape=(num_atoms,) with values in {0, 1, 2, 3}.
  """
  # For now, choose the frames based on the dense index 0, 1, 2.
  frame_labels = np.full(len(fragment_indices), fill_value=3)

  for fragment in np.unique(fragment_indices):
    in_fragment_indices = np.where(fragment_indices == fragment)[0]
    frame_indices_for_frag = np.sort(in_fragment_indices)[:3]

    # Handle the case where we have fewer than 3 atoms.
    labels = np.array([0, 1, 2])[: len(frame_indices_for_frag)]
    frame_labels[frame_indices_for_frag] = labels

  return frame_labels.astype(np.int32)
