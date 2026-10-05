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

"""Nitrene transfer metrics for protein design evaluation."""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
from typing import Any, Final

from alphafold3 import structure
from alphafold3.constants import chemical_components
from alphafold3.cpp import string_array
from alphaprotein_novo.constants import atom_types
from alphaprotein_novo.data import structure_utils
from alphaprotein_novo.metrics import geometry
from alphaprotein_novo.metrics import multi_state_metrics
import numpy as np
from scipy import spatial

# Heme iron atom names.
_FE_ATOM_NAMES: Final[Sequence[str]] = ('FE', 'Fe1', 'FE1')


# Polar sidechain residues that can form hydrogen bonds with heme propionates.
_HBOND_DONOR_RES_NAMES: Final[Sequence[str]] = (
    'ARG',
    'HIS',
    'LYS',
    'SER',
    'THR',
    'ASN',
    'GLN',
    'TYR',
)


# Candidate coordinating atoms on axial histidine (ND1 or NE2).
_HIS_COORDINATING_ATOMS: Final[Sequence[str]] = ('ND1', 'NE2')


# Histidine imidazole ring atoms for RMSD evaluation against ideal CCD geometry.
_HISTIDINE_IMIDAZOLE_ATOMS: Final[Sequence[str]] = (
    'CG',
    'ND1',
    'CE1',
    'NE2',
    'CD2',
)


@functools.cache
def _get_ideal_histidine_imidazole_coords() -> np.ndarray:
  """Returns ideal CCD Cartesian coordinates for the HIS imidazole ring.

  The atoms are ordered as in `_HISTIDINE_IMIDAZOLE_ATOMS`:
  ('CG', 'ND1', 'CE1', 'NE2', 'CD2').

  Returns:
    (5, 3) numpy array of ideal coordinates in Angstroms.
  """
  ccd = chemical_components.Ccd()
  his_data = ccd['HIS']
  atom_ids = his_data['_chem_comp_atom.atom_id']
  x = [float(v) for v in his_data['_chem_comp_atom.pdbx_model_Cartn_x_ideal']]
  y = [float(v) for v in his_data['_chem_comp_atom.pdbx_model_Cartn_y_ideal']]
  z = [float(v) for v in his_data['_chem_comp_atom.pdbx_model_Cartn_z_ideal']]
  coords_by_atom = {
      aid: np.array([xi, yi, zi], dtype=float)
      for aid, xi, yi, zi in zip(atom_ids, x, y, z)
  }
  return np.stack([coords_by_atom[a] for a in _HISTIDINE_IMIDAZOLE_ATOMS])


def _get_heme_hbond_partners(
    struct: structure.Structure,
    target_coords: np.ndarray,
    hbond_dist: float = 3.6,
) -> Sequence[tuple[str, str, None, str]]:
  """Get atoms within heme-propionate hbond distance of target coordinates.

  Args:
    struct: Protein structure containing sidechain atoms.
    target_coords: Coordinates of target atoms (e.g. heme propionate oxygens).
    hbond_dist: Distance threshold in Angstroms (default 3.6 Å).

  Returns:
    List of atom IDs within distance threshold.
  """
  if target_coords.size == 0:
    return []

  protein_only = struct.filter_to_entity_type(protein=True)
  if protein_only.num_atoms == 0:
    return []

  is_backbone = string_array.isin(
      protein_only.atom_name, set(atom_types.PROTEIN_BACKBONE_WITH_OXYGEN)
  )
  rel_sidechains = protein_only.filter(
      ~is_backbone,
      res_name=list(_HBOND_DONOR_RES_NAMES),
      atom_element=['N', 'O'],
  )

  if rel_sidechains.num_atoms == 0:
    return []

  distances = spatial.distance.cdist(rel_sidechains.coords, target_coords)
  if distances.size == 0:
    return []

  min_dists = np.min(distances, axis=-1)
  keep_mask = min_dists <= hbond_dist
  binding_res = rel_sidechains.filter(keep_mask)

  return binding_res.atom_ids


def _propionate_hbond_summary(
    struct: structure.Structure,
    hbond_dist: float = 3.6,
) -> dict[str, float]:
  """Summarize propionate hydrogen bonding partners.

  Handles both separate HEM residue (with O1A, O2A and O1D, O2D) and
  covalently-bound heme-substrate complex pd00 (with O1, O2 and O3, O4).

  Args:
    struct: Folded structure containing protein and heme.
    hbond_dist: Maximum distance in Angstroms to count as an H-bond.

  Returns:
    Dictionary containing counts of H-bonding atoms and unique residues.
  """
  # Check if separate HEM residue exists.
  prop_a = struct.filter(res_name='HEM', atom_name=['O1A', 'O2A'])
  prop_d = struct.filter(res_name='HEM', atom_name=['O1D', 'O2D'])

  if prop_a.num_atoms == 0 and prop_d.num_atoms == 0:
    # Fall back to merged ligand (e.g. pd00).
    prop_a = struct.filter(res_name='pd00', atom_name=['O1', 'O2'])
    prop_d = struct.filter(res_name='pd00', atom_name=['O3', 'O4'])

  prop_a_partners = _get_heme_hbond_partners(
      struct, prop_a.coords, hbond_dist=hbond_dist
  )
  prop_d_partners = _get_heme_hbond_partners(
      struct, prop_d.coords, hbond_dist=hbond_dist
  )

  prop_a_num_atoms = len(prop_a_partners)
  prop_a_num_res = len({atom[1] for atom in prop_a_partners})
  prop_d_num_atoms = len(prop_d_partners)
  prop_d_num_res = len({atom[1] for atom in prop_d_partners})

  prop_partners = set(prop_a_partners) | set(prop_d_partners)
  num_total_atoms = len(prop_partners)
  num_total_res = len({atom[1] for atom in prop_partners})

  return {
      'propionate_hbond_num_atoms': float(num_total_atoms),
      'propionate_hbond_num_res': float(num_total_res),
      'propionate_a_hbond_num_atoms': float(prop_a_num_atoms),
      'propionate_a_hbond_num_res': float(prop_a_num_res),
      'propionate_d_hbond_num_atoms': float(prop_d_num_atoms),
      'propionate_d_hbond_num_res': float(prop_d_num_res),
  }


def _substrate_pocket_metrics(
    struct: structure.Structure,
    substrate_res_name: str = 'pd00',
) -> dict[str, float]:
  """Calculate substrate pocket metrics around tail atom C9.

  Args:
    struct: Structure containing protein and substrate.
    substrate_res_name: Residue name of the substrate ligand.

  Returns:
    Dict with min atom distance and mean neighbor counts.
  """
  prot = struct.filter_to_entity_type(protein=True)
  ligand_atoms = struct.filter(
      res_name=substrate_res_name,
      atom_name='C9',
  )

  if ligand_atoms.num_atoms == 0 or prot.num_atoms == 0:
    return {
        'pocket_C9_min_atom_dist': np.nan,
        'pocket_C9_ca_neighbors_mean': np.nan,
        'pocket_C9_allatom_neighbors_mean': np.nan,
    }

  distances = spatial.distance.cdist(prot.coords, ligand_atoms.coords)
  min_dist = float(np.min(distances))

  struct_wo_heme = struct.filter_out(res_name='HEM')
  ligand_subset = struct_wo_heme.filter(res_name=substrate_res_name)
  if ligand_subset.num_atoms == 0:
    return {
        'pocket_C9_min_atom_dist': min_dist,
        'pocket_C9_ca_neighbors_mean': np.nan,
        'pocket_C9_allatom_neighbors_mean': np.nan,
    }

  ligand_chain = str(ligand_subset.chains[0])
  ca_neighbors, allatom_neighbors = structure_utils.atom_neighbors(
      struct=struct_wo_heme,
      atoms=[(ligand_chain, 1, 'C9')],
  )

  return {
      'pocket_C9_min_atom_dist': min_dist,
      'pocket_C9_ca_neighbors_mean': float(ca_neighbors[0]),
      'pocket_C9_allatom_neighbors_mean': float(allatom_neighbors[0]),
  }


def _intra_carbon_distances(
    struct: structure.Structure,
    ligating_n: str = 'N1',
    near_carbon_name: str = 'C4',
    far_carbon_name: str = 'C5',
    substrate_res_name: str = 'pd00',
) -> dict[str, float]:
  """Compare ring-forming carbon distances to the substrate ligating nitrogen.

  Measures distances between the heme-ligating nitrogen and the two candidate
  carbons for smaller (C4, pyrrolidine) vs larger (C5, piperidine) ring
  formation.

  Args:
    struct: Structure containing the substrate ligand.
    ligating_n: Atom name of the heme-ligating nitrogen.
    near_carbon_name: Atom name of the smaller-ring carbon (C4).
    far_carbon_name: Atom name of the larger-ring carbon (C5).
    substrate_res_name: Residue name of the substrate.

  Returns:
    Dict of metric name -> float value.
  """
  substrate = struct.filter(res_name=substrate_res_name)
  substrate_coords = dict(zip(substrate.atom_name, substrate.coords))
  if (
      ligating_n not in substrate_coords
      or near_carbon_name not in substrate_coords
      or far_carbon_name not in substrate_coords
  ):
    return {
        'substrate_n_c5_pip_dist': np.nan,
        'substrate_n_c4_pyr_dist': np.nan,
        'piperidine_propensity': np.nan,
    }

  n_c_near_dist = float(
      np.linalg.norm(
          substrate_coords[ligating_n] - substrate_coords[near_carbon_name]
      )
  )
  n_c_far_dist = float(
      np.linalg.norm(
          substrate_coords[ligating_n] - substrate_coords[far_carbon_name]
      )
  )
  diff = n_c_far_dist - n_c_near_dist

  return {
      'substrate_n_c5_pip_dist': n_c_far_dist,
      'substrate_n_c4_pyr_dist': n_c_near_dist,
      'piperidine_propensity': diff,
  }


def _find_axial_his(
    struct: structure.Structure,
    fe_atom_names: Sequence[str] = _FE_ATOM_NAMES,
    distance_threshold: float = 3.5,
) -> int | None:
  """Find the residue ID of the axial histidine closest to heme Fe.

  Scans protein sidechain atoms on chain 'A' (excluding backbone N, CA, C, O,
  OXT) and returns the residue ID if the closest sidechain atom is a HIS within
  `distance_threshold`.

  Args:
    struct: Folded structure containing protein and heme.
    fe_atom_names: Atom names to search for iron (tries each in order).
    distance_threshold: Maximum distance (Å) to consider a coordination.

  Returns:
    Residue ID of the axial histidine, or None if not found within threshold.
  """
  fe_xyz = np.empty((0, 3))
  for fe_name in fe_atom_names:
    fe_coords = struct.filter(atom_name=fe_name).coords
    if fe_coords.size > 0:
      fe_xyz = fe_coords
      break
  if fe_xyz.size == 0:
    return None

  protein_only = struct.filter_to_entity_type(protein=True)
  if protein_only.num_atoms == 0:
    return None

  is_backbone = string_array.isin(
      protein_only.atom_name, set(atom_types.PROTEIN_BACKBONE_WITH_OXYGEN)
  )
  sidechain = protein_only.filter(mask=~is_backbone, chain_id='A')
  if sidechain.num_atoms == 0:
    return None

  dists = spatial.distance.cdist(sidechain.coords, fe_xyz)
  min_dist_per_atom = dists.min(axis=1)
  best_idx = int(min_dist_per_atom.argmin())
  best_dist = float(min_dist_per_atom[best_idx])

  if best_dist > distance_threshold:
    return None

  if str(sidechain.res_name[best_idx]) != 'HIS':
    return None

  return int(sidechain.res_id[best_idx])


def _his_heme_dist(
    struct: structure.Structure,
    his_res_id: int,
    fe_atom_names: Sequence[str] = _FE_ATOM_NAMES,
) -> float:
  """Distance between heme Fe and candidate coordinating atoms on axial HIS.

  Calculates the minimum distance from heme iron (FE/Fe1) to coordinating atoms
  (ND1 or NE2) on the specified axial histidine residue on chain A.

  Args:
    struct: Folded structure containing protein and heme.
    his_res_id: Residue index of the axial histidine residue.
    fe_atom_names: Atom names for heme iron.

  Returns:
    Minimum distance in Angstroms between Fe and coordinating atoms on HIS,
    or np.nan if missing or non-histidine residue.
  """
  fe = struct.filter(atom_name=list(fe_atom_names))
  if fe.num_atoms == 0:
    return np.nan

  his_candidates = struct.filter(
      chain_id='A',
      res_id=[his_res_id],
      res_name='HIS',
      atom_name=list(_HIS_COORDINATING_ATOMS),
  )
  if his_candidates.num_atoms == 0:
    return np.nan

  distances = spatial.distance.cdist(fe.coords, his_candidates.coords)
  return float(np.min(distances))


def _his_imidazole_rmsd(
    struct: structure.Structure,
    his_res_id: int,
) -> float:
  """Planar all-atom RMSD of coordinating histidine imidazole ring vs ideal CCD.

  Aligns the 5 ring atoms ('CG', 'ND1', 'CE1', 'NE2', 'CD2') to the standard
  Chemical Component Dictionary (CCD) ideal Cartesian coordinates using Kabsch
  rigid-body alignment, and returns the resulting RMSD in Angstroms.

  Args:
    struct: Structure containing protein.
    his_res_id: Residue index of the histidine residue.

  Returns:
    RMSD value in Angstroms, or np.nan if histidine or required atoms missing.
  """
  his_res = struct.filter(
      chain_id='A',
      res_id=[his_res_id],
      res_name='HIS',
  )
  if his_res.num_atoms == 0:
    return np.nan

  coords_by_atom = dict(zip(his_res.atom_name, his_res.coords))
  if not all(atom in coords_by_atom for atom in _HISTIDINE_IMIDAZOLE_ATOMS):
    return np.nan

  sample_coords = np.stack(
      [coords_by_atom[atom] for atom in _HISTIDINE_IMIDAZOLE_ATOMS]
  )
  ref_coords = _get_ideal_histidine_imidazole_coords()

  indices = np.arange(len(_HISTIDINE_IMIDAZOLE_ATOMS))
  aligned_coords = geometry.align(
      x=sample_coords,
      y=ref_coords,
      x_indices=indices,
      y_indices=indices,
  )

  sq_deviations = np.sum((aligned_coords - ref_coords) ** 2, axis=-1)
  return float(np.sqrt(np.mean(sq_deviations)))


def nitrene_structure_metrics(
    struct: structure.Structure,
    *,
    his_res_id: int | None = None,
    substrate_res_name: str = 'pd00',
    ligating_n: str = 'N1',
    near_carbon_name: str = 'C4',
    far_carbon_name: str = 'C5',
    hbond_dist: float = 3.6,
) -> dict[str, Any]:
  """Unified entry point computing required Phase 1 nitrene transfer metrics.

  Args:
    struct: Folded structure containing protein, heme, and substrate ligand.
    his_res_id: Residue index for axial histidine residue. If None,
      automatically discovered via `_find_axial_his`.
    substrate_res_name: Residue name of the substrate ligand.
    ligating_n: Ligating nitrogen atom name on substrate.
    near_carbon_name: Smaller-ring carbon name (C4).
    far_carbon_name: Larger-ring carbon name (C5).
    hbond_dist: H-bond distance threshold.

  Returns:
    Dictionary containing the required Phase 1 metrics.
  """
  ret: dict[str, Any] = {}

  # 1. Propionate hydrogen bonds (total atoms and total residues).
  hbond_summary = _propionate_hbond_summary(
      struct,
      hbond_dist=hbond_dist,
  )
  ret['propionate_hbond_num_atoms'] = hbond_summary[
      'propionate_hbond_num_atoms'
  ]
  ret['propionate_hbond_num_res'] = hbond_summary['propionate_hbond_num_res']

  # 2. Substrate pocket metrics (min distance and all-atom neighbors).
  pocket_metrics = _substrate_pocket_metrics(
      struct,
      substrate_res_name=substrate_res_name,
  )
  ret['pocket_C9_min_atom_dist'] = pocket_metrics['pocket_C9_min_atom_dist']
  ret['pocket_C9_allatom_neighbors_mean'] = pocket_metrics[
      'pocket_C9_allatom_neighbors_mean'
  ]

  # 3. Intra-carbon distances & regioselectivity.
  intra_dists = _intra_carbon_distances(
      struct,
      ligating_n=ligating_n,
      near_carbon_name=near_carbon_name,
      far_carbon_name=far_carbon_name,
      substrate_res_name=substrate_res_name,
  )
  ret.update(intra_dists)

  # 4. Axial heme distance and histidine geometry.
  # Fall back to _find_axial_his if his_res_id is not specified.
  resolved_his_res_id = his_res_id
  if resolved_his_res_id is None:
    resolved_his_res_id = _find_axial_his(struct)

  if resolved_his_res_id is not None:
    ret['his_heme_dist'] = _his_heme_dist(
        struct,
        his_res_id=resolved_his_res_id,
    )
    ret['sampled_his_rmsd'] = _his_imidazole_rmsd(
        struct,
        his_res_id=resolved_his_res_id,
    )
  else:
    ret['his_heme_dist'] = np.nan
    ret['sampled_his_rmsd'] = np.nan

  return ret


NITRENE_STATE_NAMES: Final[frozenset[str]] = frozenset({
    'heme_substrate',
    'heme_piperidine_r',
    'heme_pyrrolidine_s',
    'fiveazidopentylbenzene_heme_1',
    'fiveazidopentylbenzene_heme_2',
})


@dataclasses.dataclass(frozen=True)
class NitreneFoldedStates:
  """Container for folded structures across reaction states for nitrene transfer.

  Attributes:
    resequenced: The designed / resequenced protein structure used as alignment
      reference.
    heme_substrate: Fold of heme + substrate ligand.
    heme_piperidine_r: Fold of heme + piperidine product (R stereocenter).
    heme_pyrrolidine_s: Fold of heme + pyrrolidine byproduct (S stereocenter).
    fiveazidopentylbenzene_heme_1: Fold with covalent intermediate 1.
    fiveazidopentylbenzene_heme_2: Fold with covalent intermediate 2.
  """

  resequenced: structure.Structure | None = None
  heme_substrate: structure.Structure | None = None
  heme_piperidine_r: structure.Structure | None = None
  heme_pyrrolidine_s: structure.Structure | None = None
  fiveazidopentylbenzene_heme_1: structure.Structure | None = None
  fiveazidopentylbenzene_heme_2: structure.Structure | None = None

  def to_state_specs(self) -> list[multi_state_metrics.StateSpec]:
    """Converts available folded structures to StateSpec objects."""
    specs = []
    state_mapping = [
        ('substrate', self.heme_substrate, 'N1'),
        ('PIP_R', self.heme_piperidine_r, 'N1'),
        ('PYR_S', self.heme_pyrrolidine_s, 'N1'),
        ('SUBSTRATE_HEME_1', self.fiveazidopentylbenzene_heme_1, 'N4'),
        ('SUBSTRATE_HEME_2', self.fiveazidopentylbenzene_heme_2, 'N1'),
    ]

    for role_name, struct, n_name in state_mapping:
      if struct is None or struct.filter(atom_name=n_name).num_atoms != 1:
        continue
      fe_atoms = struct.filter(atom_name=_FE_ATOM_NAMES)
      if fe_atoms.num_atoms == 1:
        specs.append(
            multi_state_metrics.StateSpec(
                name=role_name,
                structure=struct,
                atom_selections={
                    'fe': multi_state_metrics.AtomSelection(
                        atom_name=str(fe_atoms.atom_name[0])
                    ),
                    'n': multi_state_metrics.AtomSelection(atom_name=n_name),
                },
            )
        )
    return specs


def _coerce_to_nitrene_folded_states(
    states: NitreneFoldedStates | Mapping[str, structure.Structure],
) -> NitreneFoldedStates:
  """Coerces a mapping of structures to a NitreneFoldedStates instance."""
  if isinstance(states, NitreneFoldedStates):
    return states
  lowered = {k.lower(): v for k, v in states.items()}
  return NitreneFoldedStates(
      resequenced=lowered.get('resequenced'),
      heme_substrate=lowered.get('heme_substrate'),
      heme_piperidine_r=lowered.get('heme_piperidine_r'),
      heme_pyrrolidine_s=lowered.get('heme_pyrrolidine_s'),
      fiveazidopentylbenzene_heme_1=lowered.get(
          'fiveazidopentylbenzene_heme_1'
      ),
      fiveazidopentylbenzene_heme_2=lowered.get(
          'fiveazidopentylbenzene_heme_2'
      ),
  )


def compute_nitrene_multi_state_metrics(
    states: NitreneFoldedStates | Mapping[str, structure.Structure],
    *,
    reference_structure: structure.Structure | None = None,
) -> dict[str, Any]:
  """Computes multi-state consistency and distance metrics for nitrene transfer.

  Calculates:
    - 'fe_self_rmsd': Centroid RMSD of heme iron across aligned states.
    - 'n_self_rmsd': Centroid RMSD of ligating nitrogen across aligned states.
    - 'Fe_N_dist/{role}': Minimum Fe-N distance in each reaction fold.

  Args:
    states: A NitreneFoldedStates instance or a dictionary of structures.
    reference_structure: Structure to align all states to. If None and states
      has a resequenced structure, uses states.resequenced.

  Returns:
    Dictionary of multi-state metrics.
  """
  folded_states = _coerce_to_nitrene_folded_states(states)
  ref = (
      reference_structure
      if reference_structure is not None
      else folded_states.resequenced
  )
  specs = folded_states.to_state_specs()

  results: dict[str, Any] = {}

  # Fe-N distance in each individual state
  for spec in specs:
    fe_sel = spec.atom_selections['fe']
    n_sel = spec.atom_selections['n']
    fe_xyz = spec.structure.filter(atom_name=fe_sel.atom_name).coords
    n_xyz = spec.structure.filter(atom_name=n_sel.atom_name).coords
    dists = spatial.distance.cdist(fe_xyz, n_xyz)
    results[f'Fe_N_dist/{spec.name}'] = float(np.min(dists))

  # Cross-state RMSDs
  if ref is not None and len(specs) >= 2:
    consistency = multi_state_metrics.compute_multi_state_consistency(
        states=specs,
        reference_structure=ref,
    )
    results.update(consistency)
  else:
    results['fe_self_rmsd'] = np.nan
    results['n_self_rmsd'] = np.nan

  return results


def compute_nitrene_multi_seed_metrics(
    states_by_seed: Mapping[
        str, NitreneFoldedStates | Mapping[str, structure.Structure]
    ],
    *,
    reference_structure: structure.Structure | None = None,
) -> dict[str, Any]:
  """Computes nitrene multi-state metrics across multiple seeds with summary statistics."""
  all_metrics: dict[str, Any] = {}
  seeds = list(states_by_seed.keys())

  aligned_n_by_role_seed: dict[str, dict[str, np.ndarray]] = {}

  for seed, states in states_by_seed.items():
    seed_metrics = compute_nitrene_multi_state_metrics(
        states, reference_structure=reference_structure
    )
    for k, v in seed_metrics.items():
      all_metrics[f'{k}/{seed}'] = v

    fs = _coerce_to_nitrene_folded_states(states)
    ref = (
        reference_structure
        if reference_structure is not None
        else fs.resequenced
    )
    if ref is not None:
      for spec in fs.to_state_specs():
        aligned = multi_state_metrics.align_structure_to_reference(
            spec.structure, ref
        )
        n_coords = spec.atom_selections['n'].select_coords(aligned)
        aligned_n_by_role_seed.setdefault(spec.name, {})[seed] = n_coords

  reduced = multi_state_metrics.reduce_seed_metrics(all_metrics, seeds)

  # Cross-seed per-role RMSD (already reduced across seeds).
  for role_name, seed_map in aligned_n_by_role_seed.items():
    if len(seed_map) >= 2:
      role_coords = [seed_map[s] for s in seeds if s in seed_map]
      role_rmsd = float(geometry.centroid_rmsd(role_coords))
      reduced[f'n_self_rmsd/{role_name}'] = role_rmsd

  return reduced
