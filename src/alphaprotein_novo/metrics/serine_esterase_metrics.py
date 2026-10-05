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

"""Serine esterase / hydrolase catalytic triad and oxyanion hole metrics."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Final

from alphafold3 import structure
from alphafold3.cpp import string_array
from alphaprotein_novo.constants import residue_names
from alphaprotein_novo.metrics import catalytic_motifs
from alphaprotein_novo.metrics import hbonds
from alphaprotein_novo.metrics import ligand_metrics
import numpy as np

_AtomTuple = tuple[str, int, str]

_TI1_BY_ES_ATOM_NAMES_4MU_AC: Final[dict[str, str]] = {
    'O1': 'O3',
    'C4': 'C10',
    'C3': 'C9',
    'C2': 'C7',
    'C1': 'C8',
    'C12': 'C6',
    'C11': 'C5',
    'C10': 'C4',
    'C7': 'C3',
    'C6': 'C12',
    'C5': 'C11',
    'O2': 'O4',
    'O3': 'O2',
    'C8': 'C2',
}

_TI1_BY_ES_ATOM_NAMES_DEHP: Final[dict[str, str]] = {
    a: a
    for a in [f'C{i}' for i in range(1, 25)] + [f'O{i}' for i in range(1, 5)]
}

# C2 symmetry atom renaming between the two identical 2-ethylhexyl arms of DEHP.
_DEHP_C2_ATOM_PERMUTATION: Final[dict[str, str]] = {
    'C10': 'C15',
    'C15': 'C10',
    'C11': 'C14',
    'C14': 'C11',
    'C12': 'C13',
    'C13': 'C12',
    'C9': 'C16',
    'C16': 'C9',
    'O1': 'O4',
    'O4': 'O1',
    'O2': 'O3',
    'O3': 'O2',
    'C8': 'C17',
    'C17': 'C8',
    'C5': 'C18',
    'C18': 'C5',
    'C6': 'C19',
    'C19': 'C6',
    'C7': 'C20',
    'C20': 'C7',
    'C4': 'C21',
    'C21': 'C4',
    'C3': 'C22',
    'C22': 'C3',
    'C2': 'C23',
    'C23': 'C2',
    'C1': 'C24',
    'C24': 'C1',
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class SerineEsteraseLigandMetadata:
  """Metadata for a serine esterase ligand.

  Attributes:
    oxyanion_atom: Name of the carbonyl/oxyanion oxygen atom in the ligand.
    ester_atom: Name of the leaving ester oxygen atom in the ligand (if any).
  """

  oxyanion_atom: str
  ester_atom: str | None = None


# Metadata for commonly used serine esterase substrates.
SERINE_ESTERASE_LIGAND_METADATA: Final[
    Mapping[str, SerineEsteraseLigandMetadata]
] = {
    'pd00': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
    '4mu_ac': SerineEsteraseLigandMetadata(
        oxyanion_atom='O1',
        ester_atom='O2',
    ),
    '4mu_ac_ti1R': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
    '4mu_ac_ti1S': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
    '4mu_ac_aei': SerineEsteraseLigandMetadata(
        oxyanion_atom='O1',
        ester_atom=None,
    ),
    '4mu_ac_ti2R': SerineEsteraseLigandMetadata(
        oxyanion_atom='O2',
        ester_atom='O1',
    ),
    '4mu_ac_ti2S': SerineEsteraseLigandMetadata(
        oxyanion_atom='O2',
        ester_atom='O1',
    ),
    'dehp': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_SS': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_RR': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_SS_ti1S': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_SS_ti1R': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_RR_ti1S': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_RR_ti1R': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom='O4',
    ),
    'dehp_S_aei': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom=None,
    ),
    'dehp_R_aei': SerineEsteraseLigandMetadata(
        oxyanion_atom='O3',
        ester_atom=None,
    ),
    'dehp_S_ti2S': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
    'dehp_S_ti2R': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
    'dehp_R_ti2S': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
    'dehp_R_ti2R': SerineEsteraseLigandMetadata(
        oxyanion_atom='O4',
        ester_atom='O3',
    ),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class SerineCatalyticTriadAtoms:
  """Atom tuples defining the catalytic residues for a serine esterase.

  Attributes:
    nucleophile: Atom tuple for Ser OG nucleophile (chain_id, res_id, 'OG').
    his_ne2: Atom tuple for His NE2 (chain_id, res_id, 'NE2').
    his_nd1: Atom tuple for His ND1 (chain_id, res_id, 'ND1').
    acid_o1: Atom tuple for Asp OD1 or Glu OE1 (chain_id, res_id, 'OD1'/'OE1').
    acid_o2: Atom tuple for Asp OD2 or Glu OE2 (chain_id, res_id, 'OD2'/'OE2').
    oah1: Optional atom tuple for oxyanion hole 1 donor (typically backbone N).
    oah2: Optional atom tuple for oxyanion hole 2 donor (backbone N or
      sidechain).
  """

  nucleophile: _AtomTuple
  his_ne2: _AtomTuple
  his_nd1: _AtomTuple
  acid_o1: _AtomTuple
  acid_o2: _AtomTuple | None = None
  oah1: _AtomTuple | None = None
  oah2: _AtomTuple | None = None


def get_triad_atoms_from_residues(
    struct: structure.Structure,
    *,
    ser_res_id: int,
    his_res_id: int,
    acid_res_id: int,
    chain_id: str | None = None,
    oah1_res_id: int | None = None,
    oah2_res_id: int | None = None,
    oah2_atom_name: str = 'N',
) -> SerineCatalyticTriadAtoms:
  """Builds SerineCatalyticTriadAtoms from residue sequence IDs.

  Args:
    struct: Protein or complex structure.
    ser_res_id: 1-based residue sequence ID of the catalytic Ser.
    his_res_id: 1-based residue sequence ID of the catalytic His.
    acid_res_id: 1-based residue sequence ID of the catalytic Asp or Glu.
    chain_id: Optional chain ID for protein residues (defaults to first protein
      chain).
    oah1_res_id: Optional residue ID for oxyanion hole 1 donor (defaults to
      `ser_res_id`).
    oah2_res_id: Optional residue ID for oxyanion hole 2 donor.
    oah2_atom_name: Atom name for OAH2 donor (default: 'N').

  Returns:
    Configured `SerineCatalyticTriadAtoms` instance.

  Raises:
    ValueError: If catalytic residues are not found or acid is not Asp/Glu.
  """
  prot = struct.filter_to_entity_type(protein=True)
  default_chain = prot.chains[0] if prot.chains else 'A'
  chain = chain_id if chain_id is not None else default_chain

  acid_res = prot.filter(chain_id=chain, res_id=acid_res_id)
  if acid_res.num_atoms == 0:
    raise ValueError(
        f'Catalytic acid residue {chain}{acid_res_id} not found in protein.'
    )
  acid_res_name = acid_res.res_name[0]

  if acid_res_name == residue_names.ASP:
    acid_o1 = (chain, acid_res_id, 'OD1')
    acid_o2 = (chain, acid_res_id, 'OD2')
  elif acid_res_name == residue_names.GLU:
    acid_o1 = (chain, acid_res_id, 'OE1')
    acid_o2 = (chain, acid_res_id, 'OE2')
  else:
    raise ValueError(
        f'Catalytic acid residue must be ASP or GLU, found {acid_res_name}.'
    )

  # Oxyanion hole 1 defaults to the catalytic serine backbone N.
  actual_oah1_res_id = oah1_res_id if oah1_res_id is not None else ser_res_id
  oah1 = (chain, actual_oah1_res_id, 'N')

  oah2 = None
  if oah2_res_id is not None:
    oah2 = (chain, oah2_res_id, oah2_atom_name)

  return SerineCatalyticTriadAtoms(
      nucleophile=(chain, ser_res_id, 'OG'),
      his_ne2=(chain, his_res_id, 'NE2'),
      his_nd1=(chain, his_res_id, 'ND1'),
      acid_o1=acid_o1,
      acid_o2=acid_o2,
      oah1=oah1,
      oah2=oah2,
  )


evaluate_hbond_pair = catalytic_motifs.evaluate_hbond_pair


def compute_serine_hbond_metrics(
    struct: structure.Structure,
    *,
    triad_atoms: SerineCatalyticTriadAtoms,
    oxyanion_atom: _AtomTuple | None = None,
    ester_atom: _AtomTuple | None = None,
    thresholds: Mapping[str, tuple[float, float]] = (
        hbonds.LAUKO_HBOND_THRESHOLDS
    ),
) -> dict[str, float]:
  """Computes catalytic triad and oxyanion hole H-bond metrics.

  Args:
    struct: Structure containing protein and optional ligand atoms.
    triad_atoms: Configured `SerineCatalyticTriadAtoms` catalytic definitions.
    oxyanion_atom: Optional atom tuple for ligand carbonyl/oxyanion oxygen.
    ester_atom: Optional atom tuple for ligand leaving ester oxygen.
    thresholds: Geometric thresholds for H-bond determination.

  Returns:
    Dictionary of calculated metrics:
      - 'hbond_fwd_his_nuc': His NE2 -> Ser OG forward H-bond indicator (0 or 1)
      - 'hbond_his_acid': His ND1 -> Asp/Glu acid H-bond indicator (0 or 1)
      - 'hbond_fwd_oah1_oa': OAH1 -> oxyanion forward H-bond indicator
      - 'hbond_fwd_oah1_ester': OAH1 -> ester forward H-bond indicator
      - 'hbond_fwd_oah2_oa': OAH2 -> oxyanion forward H-bond indicator
      - 'hbond_fwd_oah2_ester': OAH2 -> ester forward H-bond indicator
      Along with detailed distance, angle, and dihedral measurements.
  """
  metrics = {}

  def _record_interaction(
      label: str,
      info: dict[str, Any] | None,
  ) -> None:
    if info is None:
      metrics[f'hbond_{label}'] = np.nan
      metrics[f'hbond_fwd_{label}'] = np.nan
      metrics[f'hbond_rev_{label}'] = np.nan
      metrics[f'distance_{label}'] = np.nan
      metrics[f'angle1_{label}'] = np.nan
      metrics[f'angle2_{label}'] = np.nan
      metrics[f'dihedral1_{label}'] = np.nan
      metrics[f'dihedral2_{label}'] = np.nan
    else:
      metrics[f'hbond_{label}'] = float(info['hbond'])
      metrics[f'hbond_fwd_{label}'] = float(info['hbond_fwd'])
      metrics[f'hbond_rev_{label}'] = float(info['hbond_rev'])
      metrics[f'distance_{label}'] = float(info['distance'])
      metrics[f'angle1_{label}'] = float(info['angle1'])
      metrics[f'angle2_{label}'] = float(info['angle2'])
      metrics[f'dihedral1_{label}'] = float(info['dihedral1'])
      metrics[f'dihedral2_{label}'] = float(info['dihedral2'])

  # 1. His NE2 -> Ser OG (his_nuc)
  info_his_nuc = evaluate_hbond_pair(
      struct,
      triad_atoms.his_ne2,
      [triad_atoms.nucleophile],
      thresholds=thresholds,
  )
  _record_interaction('his_nuc', info_his_nuc)

  # 2. His ND1 -> Acid OD1/OD2 or OE1/OE2 (his_acid)
  acid_candidates = [triad_atoms.acid_o1]
  if triad_atoms.acid_o2 is not None:
    acid_candidates.append(triad_atoms.acid_o2)
  info_his_acid = evaluate_hbond_pair(
      struct,
      triad_atoms.his_nd1,
      acid_candidates,
      thresholds=thresholds,
  )
  _record_interaction('his_acid', info_his_acid)

  # 3. OAH1 -> Oxyanion (oah1_oa)
  if triad_atoms.oah1 is not None and oxyanion_atom is not None:
    info_oah1_oa = evaluate_hbond_pair(
        struct,
        triad_atoms.oah1,
        [oxyanion_atom],
        thresholds=thresholds,
    )
    _record_interaction('oah1_oa', info_oah1_oa)
  else:
    _record_interaction('oah1_oa', None)

  # 4. OAH1 -> Ester (oah1_ester)
  if triad_atoms.oah1 is not None and ester_atom is not None:
    info_oah1_ester = evaluate_hbond_pair(
        struct,
        triad_atoms.oah1,
        [ester_atom],
        thresholds=thresholds,
    )
    _record_interaction('oah1_ester', info_oah1_ester)
  else:
    _record_interaction('oah1_ester', None)

  # 5. OAH2 -> Oxyanion (oah2_oa)
  if triad_atoms.oah2 is not None and oxyanion_atom is not None:
    info_oah2_oa = evaluate_hbond_pair(
        struct,
        triad_atoms.oah2,
        [oxyanion_atom],
        thresholds=thresholds,
    )
    _record_interaction('oah2_oa', info_oah2_oa)
  else:
    _record_interaction('oah2_oa', None)

  # 6. OAH2 -> Ester (oah2_ester)
  if triad_atoms.oah2 is not None and ester_atom is not None:
    info_oah2_ester = evaluate_hbond_pair(
        struct,
        triad_atoms.oah2,
        [ester_atom],
        thresholds=thresholds,
    )
    _record_interaction('oah2_ester', info_oah2_ester)
  else:
    _record_interaction('oah2_ester', None)

  return metrics


_TRIAD_ROLE_DEFINITIONS: Final[Mapping[str, tuple[str, ...]]] = {
    'ser': (residue_names.SER,),
    'his': (residue_names.HIS,),
    'acid': (residue_names.ASP, residue_names.GLU),
}


def get_valid_triad_combinations(
    struct: structure.Structure,
    motif_res_indices: Sequence[int],
) -> list[dict[str, int]]:
  """Identifies valid triad combinations matching residue types.

  Args:
    struct: Protein structure.
    motif_res_indices: List of residue indices involved in the motif (3 or 4).

  Returns:
    List of dictionaries mapping 'ser', 'his', 'acid', and optionally 'oah2'
    to residue indices.

  Raises:
    ValueError: If number of residue indices is less than 3.
  """
  if len(motif_res_indices) < 3:
    raise ValueError(
        f'Unsupported number of motif residues: {len(motif_res_indices)}. '
        'Expected at least 3 residues.'
    )

  combos = catalytic_motifs.generate_valid_role_combinations(
      struct, motif_res_indices, _TRIAD_ROLE_DEFINITIONS
  )
  if len(motif_res_indices) >= 4:
    valid_combinations = []
    for c in combos:
      assigned = {c['ser'], c['his'], c['acid']}
      for r in motif_res_indices:
        if r not in assigned:
          valid_combinations.append({**c, 'oah2': r})
    return valid_combinations

  return combos


def find_best_triad_mapping(
    struct: structure.Structure,
    triad_combinations: Sequence[Mapping[str, int]],
) -> Mapping[str, int]:
  """Finds the triad role combination minimizing catalytic H-bond distances.

  Args:
    struct: The protein structure.
    triad_combinations: Sequence of candidate role mappings.

  Returns:
    The mapping minimizing the sum of His-Ser and His-Acid distances.

  Raises:
    ValueError: If triad_combinations is empty.
  """
  if not triad_combinations:
    raise ValueError('No valid catalytic triad combinations provided.')

  best_mapping = triad_combinations[0]
  min_dist = float('inf')

  for combo in triad_combinations:
    triad = get_triad_atoms_from_residues(
        struct,
        ser_res_id=combo['ser'],
        his_res_id=combo['his'],
        acid_res_id=combo['acid'],
        oah2_res_id=combo.get('oah2'),
    )
    info_nuc = evaluate_hbond_pair(struct, triad.his_ne2, [triad.nucleophile])
    acid_cands = [triad.acid_o1]
    if triad.acid_o2 is not None:
      acid_cands.append(triad.acid_o2)
    info_acid = evaluate_hbond_pair(struct, triad.his_nd1, acid_cands)

    d1 = info_nuc.get('distance', float('inf'))
    d2 = info_acid.get('distance', float('inf'))
    total_dist = (d1 if not np.isnan(d1) else float('inf')) + (
        d2 if not np.isnan(d2) else float('inf')
    )
    if total_dist < min_dist:
      min_dist = total_dist
      best_mapping = combo

  return best_mapping


def _resolve_fallback_ligand_metadata(
    ligand_struct: structure.Structure,
    state_name: str | None = None,
) -> SerineEsteraseLigandMetadata | None:
  """Resolves ligand metadata from state_name and ligand atoms when CCD is generic."""
  atom_names = set(ligand_struct.atom_name.tolist())
  is_dehp = 'C16' in atom_names or 'C24' in atom_names
  norm_state = (state_name or '').lower()
  if is_dehp:
    if norm_state == 'aei':
      return SERINE_ESTERASE_LIGAND_METADATA['dehp_S_aei']
    if norm_state == 'ti2':
      return SERINE_ESTERASE_LIGAND_METADATA['dehp_S_ti2S']
    if 'O4' in atom_names and 'O3' in atom_names:
      return SERINE_ESTERASE_LIGAND_METADATA['dehp_SS']
    if 'O1' in atom_names:
      return SerineEsteraseLigandMetadata(
          oxyanion_atom='O1',
          ester_atom='O2' if 'O2' in atom_names else None,
      )
  else:
    if norm_state == 'es' and 'O1' in atom_names:
      return SERINE_ESTERASE_LIGAND_METADATA['4mu_ac']
    if norm_state in ('ti1', 'complex') and 'O4' in atom_names:
      return SERINE_ESTERASE_LIGAND_METADATA['4mu_ac_ti1S']
    if norm_state == 'aei':
      return SERINE_ESTERASE_LIGAND_METADATA['4mu_ac_aei']
    if norm_state == 'ti2' and 'O2' in atom_names:
      return SERINE_ESTERASE_LIGAND_METADATA['4mu_ac_ti2R']
    if 'O4' in atom_names and 'O3' in atom_names:
      return SERINE_ESTERASE_LIGAND_METADATA['4mu_ac_ti1S']
    if 'O1' in atom_names:
      return SerineEsteraseLigandMetadata(
          oxyanion_atom='O1',
          ester_atom='O2' if 'O2' in atom_names else None,
      )
  return None


def serine_esterase_metrics(
    struct: structure.Structure,
    motif_residues: Sequence[int] | str | Mapping[str, int] | None = None,
    *,
    ser_res_id: int | None = None,
    his_res_id: int | None = None,
    acid_res_id: int | None = None,
    oah1_res_id: int | None = None,
    oah2_res_id: int | None = None,
    oah2_atom_name: str = 'N',
    ligand_res_name: str | None = None,
    oxyanion_atom_name: str | None = None,
    ester_atom_name: str | None = None,
    state_name: str | None = None,
    thresholds: Mapping[str, tuple[float, float]] = (
        hbonds.LAUKO_HBOND_THRESHOLDS
    ),
) -> dict[str, float]:
  """Computes catalytic triad and oxyanion hole metrics for a serine esterase.

  Args:
    struct: Complex or protein structure.
    motif_residues: Optional motif specification. Can be: - A sequence of
      residue sequence IDs (e.g. `[188, 45, 93]`). - A motif string (e.g.
      `'24,A1,71,A2,31,A3'`). - A mapping of role names to residue IDs (e.g.
      `{'ser': 188, 'his': 45, 'acid': 93}`). - `None` to auto-detect fixed
      residues with `atom_b_factor > 0.0`.
    ser_res_id: Optional explicit catalytic Ser residue sequence ID.
    his_res_id: Optional explicit catalytic His residue sequence ID.
    acid_res_id: Optional explicit catalytic Asp/Glu residue sequence ID.
    oah1_res_id: Optional residue ID for oxyanion hole 1 donor (defaults to
      `ser_res_id`).
    oah2_res_id: Optional residue ID for oxyanion hole 2 donor.
    oah2_atom_name: Atom name for oxyanion hole 2 donor (default: 'N').
    ligand_res_name: Optional ligand CCD code (e.g. 'pd00', '4mu_ac'). If None,
      auto-detected from known ligands in the structure.
    oxyanion_atom_name: Optional explicit oxyanion atom name in ligand.
    ester_atom_name: Optional explicit ester oxygen atom name in ligand.
    state_name: Optional folded state name ('es', 'ti1', 'complex', 'aei',
      'ti2') used as fallback when `ligand_res_name` is a generic SMILES ligand.
    thresholds: Geometric thresholds for H-bonding.

  Returns:
    Dictionary of calculated H-bond metrics.

  Raises:
    ValueError: If catalytic residues cannot be resolved or fewer than 3
      catalytic residues are found, or an unsupported ligand is specified.
  """
  if isinstance(motif_residues, Mapping):
    ser_res_id = (
        ser_res_id
        or motif_residues.get('ser')
        or motif_residues.get('nucleophile')
    )
    his_res_id = (
        his_res_id
        or motif_residues.get('his')
        or motif_residues.get('histidine')
    )
    acid_res_id = (
        acid_res_id
        or motif_residues.get('acid')
        or motif_residues.get('carboxylic_acid')
    )
    oah1_res_id = oah1_res_id or motif_residues.get('oah1')
    oah2_res_id = oah2_res_id or motif_residues.get('oah2')

  if (
      ser_res_id is not None
      and his_res_id is not None
      and acid_res_id is not None
  ):
    effective_ser = ser_res_id
    effective_his = his_res_id
    effective_acid = acid_res_id
    effective_oah2 = oah2_res_id
  else:
    if motif_residues is None:
      prot = struct.filter_to_entity_type(protein=True)
      fixed_prot = prot.filter(atom_b_factor=prot.atom_b_factor > 0.0)
      cat_res_indices = sorted(list(set(fixed_prot.res_id)))
    elif isinstance(motif_residues, str):
      cat_res_indices = catalytic_motifs.get_motif_residue_indices(
          motif_residues
      )
    elif isinstance(motif_residues, Sequence):
      cat_res_indices = list(motif_residues)
    else:
      raise ValueError(
          f'Unsupported type for motif_residues: {type(motif_residues)}'
      )

    if len(cat_res_indices) < 3:
      raise ValueError(
          'At least 3 catalytic residues (Ser, His, Asp/Glu) required, '
          f'found {len(cat_res_indices)}: {cat_res_indices}.'
      )

    combos = get_valid_triad_combinations(struct, cat_res_indices)
    if not combos:
      raise ValueError(
          'No valid catalytic triad combinations found for residues '
          f'{cat_res_indices}.'
      )
    best_mapping = find_best_triad_mapping(struct, combos)
    effective_ser = best_mapping['ser']
    effective_his = best_mapping['his']
    effective_acid = best_mapping['acid']
    effective_oah2 = oah2_res_id or best_mapping.get('oah2')

  triad_atoms = get_triad_atoms_from_residues(
      struct,
      ser_res_id=effective_ser,
      his_res_id=effective_his,
      acid_res_id=effective_acid,
      oah1_res_id=oah1_res_id,
      oah2_res_id=effective_oah2,
      oah2_atom_name=oah2_atom_name,
  )

  # Resolve ligand atoms if present.
  oxyanion_atom = None
  ester_atom = None

  ligand_struct = struct.filter_to_entity_type(ligand=True)
  has_ligand = ligand_struct.num_residues(count_unresolved=False) > 0

  if ligand_res_name is not None and not has_ligand:
    raise ValueError(
        f'No ligand found in structure, but ligand_res_name={ligand_res_name!r}'
        ' was specified.'
    )

  if has_ligand:
    if ligand_res_name is None:
      ligand_names = list(set(ligand_struct.res_name))
      for candidate in SERINE_ESTERASE_LIGAND_METADATA:
        if candidate in ligand_names:
          ligand_res_name = candidate
          break
      if ligand_res_name is None:
        ligand_res_name = ligand_struct.res_name[0]

    lig_filtered = ligand_struct.filter(res_name=[ligand_res_name])
    fallback_meta = None
    if (
        lig_filtered.num_atoms > 0
        and (ligand_res_name == 'pd00' or state_name is not None)
        and (
            ligand_res_name == 'pd00'
            or ligand_res_name not in SERINE_ESTERASE_LIGAND_METADATA
        )
    ):
      fallback_meta = _resolve_fallback_ligand_metadata(
          lig_filtered, state_name=state_name
      )
    if fallback_meta is not None:
      eff_oxyanion_name = oxyanion_atom_name or fallback_meta.oxyanion_atom
      eff_ester_name = ester_atom_name or fallback_meta.ester_atom
    elif ligand_res_name in SERINE_ESTERASE_LIGAND_METADATA:
      meta = SERINE_ESTERASE_LIGAND_METADATA[ligand_res_name]
      eff_oxyanion_name = oxyanion_atom_name or meta.oxyanion_atom
      eff_ester_name = ester_atom_name or meta.ester_atom
    elif oxyanion_atom_name is not None:
      eff_oxyanion_name = oxyanion_atom_name
      eff_ester_name = ester_atom_name
    else:
      raise ValueError(
          f'Unsupported serine esterase ligand: {ligand_res_name}. Expected '
          f'one of {list(SERINE_ESTERASE_LIGAND_METADATA.keys())} or specify '
          'oxyanion_atom_name explicitly.'
      )

    if lig_filtered.num_atoms == 0:
      raise ValueError(f'Ligand {ligand_res_name} not found in structure.')

    lig_chain = lig_filtered.chain_id[0]
    lig_res_id = int(lig_filtered.res_id[0])

    if eff_oxyanion_name is not None:
      oxyanion_atom = (lig_chain, lig_res_id, eff_oxyanion_name)
    if eff_ester_name is not None:
      ester_atom = (lig_chain, lig_res_id, eff_ester_name)

  return compute_serine_hbond_metrics(
      struct,
      triad_atoms=triad_atoms,
      oxyanion_atom=oxyanion_atom,
      ester_atom=ester_atom,
      thresholds=thresholds,
  )


def _align_ligand_by_pocket(
    decoy_struc_one_lig: structure.Structure,
    gt_struc_one_lig: structure.Structure,
    pocket_radius: float = 10.0,
    atom_names: tuple[str, ...] = ('N', 'CA', 'C'),
) -> structure.Structure:
  """Aligns the decoy ligand to the reference structure's 10A pocket."""
  transform = ligand_metrics.get_pocket_aligned_transform(
      decoy_struc_one_lig,
      gt_struc_one_lig,
      pocket_radius=pocket_radius,
      atom_names=atom_names,
  )
  decoy_ligand = decoy_struc_one_lig.filter_to_entity_type(ligand=True)
  if decoy_ligand.num_atoms == 0:
    return decoy_ligand
  new_coords = transform(decoy_ligand.coords)
  return decoy_ligand.copy_and_update_coords(new_coords)


def _resolve_dehp_ligand_permutation(
    decoy_ligand: structure.Structure,
    reference_ligand: structure.Structure,
) -> structure.Structure:
  """Resolves DEHP C2 symmetry atom naming permutation against reference_ligand."""
  ref_names = reference_ligand.atom_name.tolist()
  decoy_names = decoy_ligand.atom_name.tolist()
  common = [
      a
      for a in _TI1_BY_ES_ATOM_NAMES_DEHP
      if a in ref_names and a in decoy_names
  ]
  if len(common) < 28:
    return decoy_ligand
  ref_coords = np.array(
      [reference_ligand.coords[ref_names.index(a)] for a in common]
  )
  orig_coords = np.array(
      [decoy_ligand.coords[decoy_names.index(a)] for a in common]
  )
  perm_coords = np.array([
      decoy_ligand.coords[
          decoy_names.index(_DEHP_C2_ATOM_PERMUTATION.get(a, a))
      ]
      for a in common
  ])
  rmsd_orig = float(
      np.sqrt(np.mean(np.sum(np.square(orig_coords - ref_coords), axis=-1)))
  )
  rmsd_perm = float(
      np.sqrt(np.mean(np.sum(np.square(perm_coords - ref_coords), axis=-1)))
  )
  if rmsd_perm < rmsd_orig:
    return decoy_ligand.copy_and_update_atoms(
        atom_name=string_array.remap(
            decoy_ligand.atom_name, _DEHP_C2_ATOM_PERMUTATION
        )
    )
  return decoy_ligand


def _get_cross_seed_ligand_coord_std(
    structures: Sequence[structure.Structure],
    reference: structure.Structure,
    *,
    is_dehp: bool = False,
) -> float:
  """Computes std of ligand coordinates across seeds after 10A pocket alignment."""
  if len(structures) < 2:
    return 0.0
  ref_struct = structures[0] if reference is None else reference
  ref_ligand = _align_ligand_by_pocket(ref_struct, ref_struct)
  if ref_ligand.num_atoms == 0:
    return float(np.nan)
  ref_names = ref_ligand.atom_name.tolist()
  ligand_coords = []
  for decoy in structures:
    aligned_lig = _align_ligand_by_pocket(decoy, ref_struct)
    if aligned_lig.num_atoms != ref_ligand.num_atoms:
      continue
    if is_dehp:
      aligned_lig = _resolve_dehp_ligand_permutation(aligned_lig, ref_ligand)
    decoy_names = aligned_lig.atom_name.tolist()
    if set(decoy_names) == set(ref_names) and len(set(ref_names)) == len(
        ref_names
    ):
      idx = [decoy_names.index(a) for a in ref_names]
      ligand_coords.append(aligned_lig.coords[idx])
    else:
      ligand_coords.append(aligned_lig.coords)
  if len(ligand_coords) < 2:
    return 0.0
  ligand_coords_arr = np.array(ligand_coords)
  return float(np.sqrt(np.var(ligand_coords_arr, axis=0).sum(axis=-1).mean()))


def compute_cross_state_ligand_metrics(
    folded_structs: Mapping[str, Mapping[int, structure.Structure]],
    designed_struct: structure.Structure,
) -> dict[str, float]:
  """Computes ES vs TI1/complex cross-state RMSD and cross-seed coord std."""
  del designed_struct
  ti1_state = 'complex' if 'complex' in folded_structs else 'ti1'
  if 'es' not in folded_structs or ti1_state not in folded_structs:
    return {}

  es_structures = list(folded_structs['es'].values())
  ti1_structures = list(folded_structs[ti1_state].values())
  if not es_structures or not ti1_structures:
    return {}

  reference = es_structures[0]
  reference_ligand = reference.filter_to_entity_type(ligand=True)
  if reference_ligand.num_atoms == 0:
    return {}

  ref_atom_names = reference_ligand.atom_name.tolist()
  ref_res_names = set(reference_ligand.res_name.tolist())
  is_dehp = any(r.lower().startswith('dehp') for r in ref_res_names) or {
      'C16',
      'C24',
  }.issubset(ref_atom_names)
  ti1_by_es_atom_names = (
      _TI1_BY_ES_ATOM_NAMES_DEHP if is_dehp else _TI1_BY_ES_ATOM_NAMES_4MU_AC
  )

  metrics: dict[str, float] = {
      'es/cross_seed_ligand_coord_std': _get_cross_seed_ligand_coord_std(
          es_structures, reference, is_dehp=is_dehp
      ),
      f'{ti1_state}/cross_seed_ligand_coord_std': (
          _get_cross_seed_ligand_coord_std(
              ti1_structures, ti1_structures[0], is_dehp=is_dehp
          )
      ),
  }

  # Use explicit TI1<->ES atom mapping when atom names match; fall back to
  # common atom names if ligands were folded from SMILES with generic names.
  mapped_pairs = [
      (ti1_atom, es_atom)
      for ti1_atom, es_atom in ti1_by_es_atom_names.items()
      if es_atom in ref_atom_names
  ]
  first_ti1_lig = ti1_structures[0].filter_to_entity_type(ligand=True)
  first_ti1_names = first_ti1_lig.atom_name.tolist()
  mapped_pairs = [
      (ti1_atom, es_atom)
      for ti1_atom, es_atom in mapped_pairs
      if ti1_atom in first_ti1_names
  ]

  es_aligned_coords: list[np.ndarray] = []
  ti1_aligned_coords: list[np.ndarray] = []

  if mapped_pairs:
    es_target_names = [es_atom for _, es_atom in mapped_pairs]
    ti1_target_names = [ti1_atom for ti1_atom, _ in mapped_pairs]
    ref_idx = [ref_atom_names.index(a) for a in es_target_names]
    es_aligned_coords.append(reference_ligand.coords[ref_idx])

    for decoy in es_structures[1:]:
      aligned_lig = _align_ligand_by_pocket(decoy, reference)
      if is_dehp:
        aligned_lig = _resolve_dehp_ligand_permutation(
            aligned_lig, reference_ligand
        )
      decoy_names = aligned_lig.atom_name.tolist()
      if all(a in decoy_names for a in es_target_names):
        idx = [decoy_names.index(a) for a in es_target_names]
        es_aligned_coords.append(aligned_lig.coords[idx])

    for decoy in ti1_structures:
      aligned_lig = _align_ligand_by_pocket(decoy, reference)
      if is_dehp:
        aligned_lig = _resolve_dehp_ligand_permutation(
            aligned_lig, reference_ligand
        )
      decoy_names = aligned_lig.atom_name.tolist()
      if all(a in decoy_names for a in ti1_target_names):
        idx = [decoy_names.index(a) for a in ti1_target_names]
        ti1_aligned_coords.append(aligned_lig.coords[idx])
  else:
    n_common = min(
        min(
            s.filter_to_entity_type(ligand=True).num_atoms
            for s in es_structures
        ),
        min(
            s.filter_to_entity_type(ligand=True).num_atoms
            for s in ti1_structures
        ),
    )
    if n_common == 0:
      return metrics
    es_aligned_coords.append(reference_ligand.coords[:n_common])
    for decoy in es_structures[1:]:
      aligned_lig = _align_ligand_by_pocket(decoy, reference)
      es_aligned_coords.append(aligned_lig.coords[:n_common])
    for decoy in ti1_structures:
      aligned_lig = _align_ligand_by_pocket(decoy, reference)
      ti1_aligned_coords.append(aligned_lig.coords[:n_common])

  if not es_aligned_coords or not ti1_aligned_coords:
    return metrics

  all_ligand_coords = np.array(es_aligned_coords + ti1_aligned_coords)
  coord_std = float(
      np.sqrt(np.var(all_ligand_coords, axis=0).sum(axis=-1).mean())
  )
  es_arr = np.array(es_aligned_coords)
  ti1_arr = np.array(ti1_aligned_coords)
  rmsds = np.sqrt(
      np.mean(
          np.sum(np.square(es_arr[:, None] - ti1_arr[None]), axis=-1),
          axis=-1,
      )
  )
  metrics['es_ti1_cross_seed_ligand_coord_std'] = coord_std
  metrics['es_ti1_min_cross_state_rmsd'] = float(rmsds.min())
  return metrics
