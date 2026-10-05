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

"""Functions for analyzing hydrogen bonds.

Uses distance, angle, and dihedral thresholds to determine if the geometry
between two heavy atoms is roughly consistent with having a hydrogen bond or
being able to participate in a proton-transfer reaction. This method and
thresholds come from Lauko et al. 2025:
  https://www.science.org/doi/10.1126/science.adu2454
"""

from collections.abc import Mapping, Sequence
import enum
from typing import Any, Final

from alphafold3 import structure
from alphafold3.constants import chemical_components
from alphafold3.data.tools import rdkit_utils
from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.constants import residue_names
import numpy as np
import rdkit.Chem as rd_chem

AtomTuple = tuple[str, int, str]
_RDKIT_ATOM_NAME = 'atom_name'


@enum.unique
class Hybridization(enum.StrEnum):
  """Molecular orbital hybridization types at an atom."""

  SP2_180 = enum.auto()
  SP2_180_OR_0 = enum.auto()
  SP3 = enum.auto()
  SINGLE_ATOM = enum.auto()


# Angle and dihedral reference atoms and hybridizations for evaluating polar
# interactions involving standard amino acids.
_REF_ATOMS_HYBRIDIZATION_BY_RESIDUE: Final[
    Mapping[str, Mapping[str, tuple[str, str, Hybridization]]]
] = {
    residue_names.ARG: {
        atom_types.NH1: (
            atom_types.CZ,
            atom_types.NE,
            Hybridization.SP2_180_OR_0,
        ),
        atom_types.NH2: (
            atom_types.CZ,
            atom_types.NE,
            Hybridization.SP2_180_OR_0,
        ),
        atom_types.NE: (atom_types.CZ, atom_types.CD, Hybridization.SP2_180),
    },
    residue_names.ASN: {
        atom_types.ND2: (
            atom_types.CG,
            atom_types.CB,
            Hybridization.SP2_180_OR_0,
        ),
        atom_types.OD1: (
            atom_types.CG,
            atom_types.CB,
            Hybridization.SP2_180_OR_0,
        ),
    },
    residue_names.ASP: {
        atom_types.OD1: (
            atom_types.CG,
            atom_types.CB,
            Hybridization.SP2_180_OR_0,
        ),
        atom_types.OD2: (
            atom_types.CG,
            atom_types.CB,
            Hybridization.SP2_180_OR_0,
        ),
    },
    residue_names.GLN: {
        atom_types.OE1: (
            atom_types.CD,
            atom_types.CG,
            Hybridization.SP2_180_OR_0,
        ),
        atom_types.NE2: (
            atom_types.CD,
            atom_types.CG,
            Hybridization.SP2_180_OR_0,
        ),
    },
    residue_names.GLU: {
        atom_types.OE1: (
            atom_types.CD,
            atom_types.CG,
            Hybridization.SP2_180_OR_0,
        ),
        atom_types.OE2: (
            atom_types.CD,
            atom_types.CG,
            Hybridization.SP2_180_OR_0,
        ),
    },
    residue_names.HIS: {
        atom_types.ND1: (atom_types.CE1, atom_types.NE2, Hybridization.SP2_180),
        atom_types.NE2: (atom_types.CE1, atom_types.ND1, Hybridization.SP2_180),
    },
    residue_names.LYS: {
        atom_types.NZ: (atom_types.CE, atom_types.CD, Hybridization.SP3),
    },
    residue_names.SER: {
        atom_types.OG: (atom_types.CB, atom_types.CA, Hybridization.SP3),
    },
    residue_names.THR: {
        atom_types.OG1: (atom_types.CB, atom_types.CA, Hybridization.SP3),
    },
    residue_names.TRP: {
        atom_types.NE1: (atom_types.CD1, atom_types.CG, Hybridization.SP2_180),
    },
    residue_names.TYR: {
        atom_types.OH: (
            atom_types.CZ,
            atom_types.CE1,
            Hybridization.SP2_180_OR_0,
        ),
    },
}

# Geometry thresholds based on Lauko et al. 2025.
LAUKO_HBOND_THRESHOLDS: Final[Mapping[str, tuple[float, float]]] = {
    'dist': (2.0, 3.6),  # In angstroms.
    'angle_sp2': (80, 160),  # In degrees.
    'dihedral_sp2_180': (130, 230),  # ~180 degrees.
    'dihedral_sp2_0': (310, 50),  # ~0 degrees.
    'angle_sp3': (69.5, 149.5),
    'dihedral_sp3': (0, 360),  # No constraint.
}


def _get_atom_coords(
    struct: structure.Structure, atom: AtomTuple
) -> np.ndarray:
  """Get coordinates for a specific atom (chain_id, res_id, atom_name)."""
  chain_id, res_id, atom_name = atom
  mask = (
      (struct.chain_id == chain_id)
      & (struct.res_id == res_id)
      & (struct.atom_name == atom_name)
  )
  indices = np.where(mask)[0]
  if len(indices) == 0:
    raise ValueError(f'Atom {atom} not found in structure.')
  return struct.coords[indices[0]]


def _angle_from_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
  """Calculates the angle (in degrees) at b formed by points a, b, and c."""
  v1 = a - b
  v2 = c - b
  norm1 = np.linalg.norm(v1)
  norm2 = np.linalg.norm(v2)
  if norm1 == 0 or norm2 == 0:
    return 0.0
  cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
  return float(np.degrees(np.arccos(cos_angle)))


def _dihedral_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
  """Calculates the dihedral angle (in degrees) for points a-b-c-d."""
  v1 = a - b
  v2 = b - c
  v3 = d - c

  c1 = np.cross(v1, v2)
  c2 = np.cross(v3, v2)
  c3 = np.cross(c2, c1)

  v2_mag = np.linalg.norm(v2)
  if v2_mag == 0 or np.linalg.norm(c1) == 0 or np.linalg.norm(c2) == 0:
    return 0.0
  return float(np.degrees(np.arctan2(np.dot(c3, v2), v2_mag * np.dot(c1, c2))))


def _get_rdkit_atom_by_name(
    mol: rd_chem.Mol,
    atom_name: str,
) -> rd_chem.Atom:
  """Returns the RDKit atom with a certain name from the given molecule."""
  for a in mol.GetAtoms():
    if a.HasProp(_RDKIT_ATOM_NAME) and a.GetProp(_RDKIT_ATOM_NAME) == atom_name:
      return a
  raise ValueError(f'Atom with name {atom_name} not found in mol {mol}.')


def get_struct_and_mol_for_residue(
    struct: structure.Structure,
    atom: AtomTuple,
) -> tuple[structure.Structure, rd_chem.Mol]:
  """Get Structure object and RDKit molecule for a given residue or ligand."""
  res_struct = struct.filter(
      (struct.chain_id == atom[0]) & (struct.res_id == atom[1])
  )
  if res_struct.num_atoms == 0:
    raise ValueError(f'Residue {atom[0]}:{atom[1]} not found in structure.')
  res_name = res_struct.res_name[0]
  ccd = chemical_components.Ccd()
  x = ccd.get(res_name)
  if x is not None:
    mol = rdkit_utils.mol_from_ccd_cif(x)
  else:
    if (
        res_struct.chemical_components_data is None
        or res_name not in res_struct.chemical_components_data.chem_comp
    ):
      raise ValueError(
          f'No chemical components data for residue {res_name} in structure.'
      )
    smiles = res_struct.chemical_components_data.chem_comp[res_name].pdbx_smiles
    mol = rd_chem.MolFromSmiles(smiles)
    mol = rdkit_utils.assign_atom_names_from_graph(mol)

  return res_struct, mol


def get_neighbor_atoms(
    mol: rd_chem.Mol,
    atom_name: str,
) -> Sequence[rd_chem.Atom]:
  """Get the neighbor atoms of a given atom in an RDKit molecule."""
  bonds = [
      b
      for b in mol.GetBonds()
      if b.GetBeginAtom().GetProp(_RDKIT_ATOM_NAME) == atom_name
      or b.GetEndAtom().GetProp(_RDKIT_ATOM_NAME) == atom_name
  ]
  return [
      b.GetBeginAtom()
      if b.GetEndAtom().GetProp(_RDKIT_ATOM_NAME) == atom_name
      else b.GetEndAtom()
      for b in bonds
  ]


def get_reference_atoms_and_hybridization(
    res_struct: structure.Structure,
    mol: rd_chem.Mol,
    atom: AtomTuple,
) -> tuple[AtomTuple | None, AtomTuple | None, Hybridization]:
  """Get reference atoms and hybridization for h-bond thresholding.

  Input atom is the "distance" atom (for calculating distance to query h-bond
  partner). Angle atom is an atom bonded to the distance atom. Dihedral atom is
  either a second atom bonded to the distance atom, or if this doesn't exist, an
  atom bonded to the angle atom. When multiple bonded partners exist, the choice
  is arbitrary (and determined by the iteration order of bonds from RDKit).

  The h-bond angle is angle_atom-distance_atom-partner_atom and the h-bond
  dihedral is dihedral_atom-angle_atom-distance_atom-partner_atom.

  Args:
    res_struct: Structure object.
    mol: RDKit molecule.
    atom: Tuple of (chain_id, res_id, atom_name).

  Returns:
    angle_atom: tuple of (chain_id, res_id, atom_name) for the atom
      used to calculate the angle and dihedral, or None if not found.
    dihedral_atom: tuple of (chain_id, res_id, atom_name) for the atom
      used to calculate the dihedral, or None if not found.
    hybridization: the hybridization of `atom`.
  """
  res_name = res_struct.res_name[0]
  if res_name in residue_names.PROTEIN_TYPES:
    # Backbone amide nitrogen at res_id i is in-plane with C, O at res_id i-1.
    if (atom[2] == atom_types.N) and (res_struct.res_id[0] != 1):
      angle_atom = (atom[0], atom[1] - 1, atom_types.C)
      dihedral_atom = (atom[0], atom[1] - 1, atom_types.O)
      return angle_atom, dihedral_atom, Hybridization.SP2_180
    elif atom[2] == atom_types.O:
      angle_atom = (atom[0], atom[1], atom_types.C)
      dihedral_atom = (atom[0], atom[1] + 1, atom_types.N)
      return angle_atom, dihedral_atom, Hybridization.SP2_180_OR_0
    elif (
        res_name in _REF_ATOMS_HYBRIDIZATION_BY_RESIDUE
        and atom[2] in _REF_ATOMS_HYBRIDIZATION_BY_RESIDUE[res_name]
    ):
      angle_atom_name, dihedral_atom_name, hybridization = (
          _REF_ATOMS_HYBRIDIZATION_BY_RESIDUE[res_name][atom[2]]
      )
      angle_atom = (atom[0], atom[1], angle_atom_name)
      dihedral_atom = (atom[0], atom[1], dihedral_atom_name)
      return angle_atom, dihedral_atom, hybridization

  # Non-standard amino acid or ligand.
  angle_atom_name, dihedral_atom_name = None, None
  neighbor_atoms = get_neighbor_atoms(mol, atom[2])
  if len(neighbor_atoms) >= 1:
    angle_atom_name = neighbor_atoms[0].GetProp(_RDKIT_ATOM_NAME)

    if len(neighbor_atoms) >= 2:
      dihedral_atom_name = neighbor_atoms[1].GetProp(_RDKIT_ATOM_NAME)
    else:
      neighbor_atoms = [
          n
          for n in get_neighbor_atoms(mol, angle_atom_name)
          if n.GetProp(_RDKIT_ATOM_NAME) != atom[2]
      ]
      if neighbor_atoms:
        dihedral_atom_name = neighbor_atoms[0].GetProp(_RDKIT_ATOM_NAME)

  angle_atom, dihedral_atom = None, None
  if angle_atom_name:
    angle_atom = (atom[0], atom[1], angle_atom_name)
  if dihedral_atom_name:
    dihedral_atom = (atom[0], atom[1], dihedral_atom_name)

  atom_rdkit = _get_rdkit_atom_by_name(mol, atom[2])
  if atom_rdkit.GetHybridization() == rd_chem.HybridizationType.SP2:
    hybridization = Hybridization.SP2_180_OR_0
  elif atom_rdkit.GetHybridization() == rd_chem.HybridizationType.SP3:
    hybridization = Hybridization.SP3
  elif mol.GetNumAtoms() == 1:  # Single atom, e.g. metal ion.
    hybridization = Hybridization.SINGLE_ATOM
  else:
    raise ValueError(
        f'Hybridization {atom_rdkit.GetHybridization()} on atom'
        f' {atom} ({atom_rdkit.GetProp(_RDKIT_ATOM_NAME)},'
        f' {atom_rdkit.GetSymbol()})) not supported.'
    )

  return angle_atom, dihedral_atom, hybridization


def threshold_geometry_metrics(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, tuple[float, float]],
) -> tuple[bool, bool]:
  """Check geometry metrics against thresholds.

  Forward h-bond is based on distance and the angle and dihedral on the first
  interacting partner atom. Reverse h-bond is based on the angle and dihedral
  on the second interacting partner atom. A "2-sided" h-bond is defined as
  both forward and reverse h-bonds existing.

  Args:
    metrics: Dictionary of geometry metrics.
    thresholds: Dictionary of thresholds for distance, angle, and dihedral.

  Returns:
    Tuple of booleans (is_hbond_fwd, is_hbond_rev) indicating whether the
    forward and reverse h-bonds exist.
  """
  if (
      metrics['dihedral1'] < 0
      or metrics['dihedral2'] < 0
      or metrics['dihedral1'] > 360
      or metrics['dihedral2'] > 360
  ):
    raise ValueError('Dihedrals must be in [0, 360].')

  def _dihedral_ok(dihedral, hybridization):
    """Check sp2 dihedrals against ~180-degree and ~0-degree thresholds."""
    if hybridization == Hybridization.SP2_180:
      return (dihedral > thresholds['dihedral_sp2_180'][0]) and (
          dihedral < thresholds['dihedral_sp2_180'][1]
      )
    elif hybridization == Hybridization.SP2_180_OR_0:
      return (
          (dihedral > thresholds['dihedral_sp2_180'][0])
          and (dihedral < thresholds['dihedral_sp2_180'][1])
      ) or (
          (dihedral > thresholds['dihedral_sp2_0'][0])
          or (dihedral < thresholds['dihedral_sp2_0'][1])
      )
    elif hybridization == Hybridization.SP3:
      return (dihedral > thresholds['dihedral_sp3'][0]) and (
          dihedral < thresholds['dihedral_sp3'][1]
      )
    raise ValueError(f'Unsupported hybridization {hybridization}.')

  if metrics['hyb_atom1'] in [
      Hybridization.SP2_180,
      Hybridization.SP2_180_OR_0,
      Hybridization.SP3,
  ]:
    hyb_atom1_simple = str(metrics['hyb_atom1']).split('_')[0]
    is_hbond_fwd = (
        (metrics['distance'] > thresholds['dist'][0])
        and (metrics['distance'] < thresholds['dist'][1])
        and (metrics['angle1'] > thresholds[f'angle_{hyb_atom1_simple}'][0])
        and (metrics['angle1'] < thresholds[f'angle_{hyb_atom1_simple}'][1])
        and _dihedral_ok(metrics['dihedral1'], metrics['hyb_atom1'])
    )
  elif metrics['hyb_atom1'] == Hybridization.SINGLE_ATOM:
    is_hbond_fwd = False
  else:
    raise ValueError(f'Unsupported hybridization {metrics["hyb_atom1"]}.')

  if metrics['hyb_atom2'] in [
      Hybridization.SP2_180,
      Hybridization.SP2_180_OR_0,
      Hybridization.SP3,
  ]:
    hyb_atom2_simple = str(metrics['hyb_atom2']).split('_')[0]
    is_hbond_rev = (
        (metrics['angle2'] > thresholds[f'angle_{hyb_atom2_simple}'][0])
        and (metrics['angle2'] < thresholds[f'angle_{hyb_atom2_simple}'][1])
        and _dihedral_ok(metrics['dihedral2'], metrics['hyb_atom2'])
    )
  elif metrics['hyb_atom2'] == Hybridization.SINGLE_ATOM:
    is_hbond_rev = False
  else:
    raise ValueError(f'Unsupported hybridization {metrics["hyb_atom2"]}.')

  return is_hbond_fwd, is_hbond_rev


def determine_hbond_by_thresholding(
    struct: structure.Structure,
    *,
    atom1: AtomTuple,
    atom2: AtomTuple,
    thresholds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
  """Determine if a hydrogen bond exists between two atoms based on thresholds.

  Args:
    struct: Structure object.
    atom1: Tuple of (chain_id, res_id, atom_name) for the first atom.
    atom2: Tuple of (chain_id, res_id, atom_name) for the second atom.
    thresholds: Dictionary of thresholds for distance, angle, and dihedral. If
      None, defaults to LAUKO_HBOND_THRESHOLDS.

  Returns:
    Dictionary with forward/reverse h-bond booleans, geometry metrics, and
    atoms.
  """
  if thresholds is None:
    thresholds = LAUKO_HBOND_THRESHOLDS

  res_struct1, mol1 = get_struct_and_mol_for_residue(struct, atom1)
  angle_atom1, dihedral_atom1, hyb_atom1 = (
      get_reference_atoms_and_hybridization(res_struct1, mol1, atom1)
  )

  res_struct2, mol2 = get_struct_and_mol_for_residue(struct, atom2)
  angle_atom2, dihedral_atom2, hyb_atom2 = (
      get_reference_atoms_and_hybridization(res_struct2, mol2, atom2)
  )

  p1 = _get_atom_coords(struct, atom1)
  p2 = _get_atom_coords(struct, atom2)
  distance = float(np.linalg.norm(p1 - p2))

  # Calculate angles and dihedrals if reference atoms exist.
  if angle_atom1 is not None:
    p_ang1 = _get_atom_coords(struct, angle_atom1)
    angle1 = _angle_from_3_points(p_ang1, p1, p2)
  else:
    angle1 = np.nan

  if dihedral_atom1 is not None and angle_atom1 is not None:
    p_dih1 = _get_atom_coords(struct, dihedral_atom1)
    p_ang1 = _get_atom_coords(struct, angle_atom1)
    dihedral1 = _dihedral_angle(p_dih1, p_ang1, p1, p2)
  else:
    dihedral1 = np.nan

  if angle_atom2 is not None:
    p_ang2 = _get_atom_coords(struct, angle_atom2)
    angle2 = _angle_from_3_points(p_ang2, p2, p1)
  else:
    angle2 = np.nan

  if dihedral_atom2 is not None and angle_atom2 is not None:
    p_dih2 = _get_atom_coords(struct, dihedral_atom2)
    p_ang2 = _get_atom_coords(struct, angle_atom2)
    dihedral2 = _dihedral_angle(p_dih2, p_ang2, p2, p1)
  else:
    dihedral2 = np.nan

  # Transform dihedrals from [-180, 180] to [0, 360].
  if dihedral1 < 0:
    dihedral1 += 360.0
  if dihedral2 < 0:
    dihedral2 += 360.0

  def _convert_to_str(x):
    return ''.join([str(y) for y in x]) if x is not None else ''

  hbond_info = {
      'distance': distance,
      'angle1': angle1,
      'dihedral1': dihedral1,
      'hyb_atom1': str(hyb_atom1),
      'angle2': angle2,
      'dihedral2': dihedral2,
      'hyb_atom2': str(hyb_atom2),
      'atom1': _convert_to_str(atom1),
      'angle_atom1': _convert_to_str(angle_atom1),
      'dihedral_atom1': _convert_to_str(dihedral_atom1),
      'atom2': _convert_to_str(atom2),
      'angle_atom2': _convert_to_str(angle_atom2),
      'dihedral_atom2': _convert_to_str(dihedral_atom2),
  }
  is_hbond_fwd, is_hbond_rev = threshold_geometry_metrics(
      hbond_info, thresholds
  )
  hbond_info['hbond_fwd'] = is_hbond_fwd
  hbond_info['hbond_rev'] = is_hbond_rev
  hbond_info['hbond'] = is_hbond_fwd and is_hbond_rev
  return hbond_info
