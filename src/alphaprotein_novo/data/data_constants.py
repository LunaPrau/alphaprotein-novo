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

"""Constants shared across data modules."""

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

# Differs from standard definition of backbone atoms. Compatible with
# ProteinMPNN and backbone-only diffusion.
BACKBONE_ATOMS_WITH_O = ('N', 'CA', 'C', 'O')

# Shape placeholder dimension names.
NUM_RES = 'num residues placeholder'
NUM_RES_UNINDEXED = 'num unindexed residues placeholder'
NUM_ATOMS = 'num atoms placeholder'
SPATIAL_DIM = 'spatial dimension placeholder'
NUM_LIGANDS = 'num ligands placeholder'
NUM_LIGAND_ATOMS = 'num ligand atoms placeholder'

# Feature groups.
SEQ_FEATURES = (
    'chain_id',
    'residue_index',
    'aatype',
    'all_atom_positions',
    'all_atom_mask',
    'b_factors',
    'seq_mask',
    'asym_id',
    'entity_id',
    'sym_id',
    'is_protein',
    'is_ligand_mask',
    'fixed_atom_mask',
    'fixed_seq_mask',
    'ligand_atomic_number',
    'ligand_charge',
    'ligand_hybridization',
    'ligand_atom_names',
    'unindexed_motif_atom_positions',
    'unindexed_motif_atom_mask',
    'unindexed_motif_aatype',
    'unindexed_motif_aatype_mask',
    'unindexed_motif_orig_res_id',
    'unindexed_motif_orig_chain_index',
)
CHAIN_FEATURES = ('seq_length',)
HOST_ONLY_FEATURES = (
    'domain_name',
    'resolution',
    'source_file_name',
)
LIGAND_FEATURES = (
    'component_atom_names',
    'component_atom_names_numerical',
    'component_atomic_numbers',
    'component_bond_types',
    'component_bonds_is_aromatic',
    'component_bonds_is_stereo',
    'component_charges',
    'component_entity_id',
    'component_fragment_indices',
    'component_frame_labels',
    'component_hybridizations',
    'component_id',
    'component_id_numerical',
    'component_padding_mask_1d',
    'component_padding_mask_2d',
    'component_smiles_descriptor',
)

FEATURE_SIZES_TPU_SUPPORTED = {
    'resolution': [],
    'residue_index': [
        NUM_RES,
    ],
    'aatype': [
        NUM_RES,
    ],
    'all_atom_positions': [NUM_RES, NUM_ATOMS, SPATIAL_DIM],
    'all_atom_mask': [NUM_RES, NUM_ATOMS],
    'b_factors': [NUM_RES, NUM_ATOMS],
    'fixed_atom_mask': [NUM_RES, NUM_ATOMS],
    'fixed_seq_mask': [NUM_RES],
    'seq_length': [],
    'seq_mask': [NUM_RES],
    'entity_id': [NUM_RES],
    'asym_id': [NUM_RES],
    'sym_id': [NUM_RES],
    'is_protein': [NUM_RES],
    'ligand_atomic_number': [NUM_RES],
    'ligand_charge': [NUM_RES],
    'ligand_hybridization': [NUM_RES],
    'ligand_atom_names': [NUM_RES],
    'unindexed_motif_atom_positions': [
        NUM_RES_UNINDEXED,
        NUM_ATOMS,
        SPATIAL_DIM,
    ],
    'unindexed_motif_atom_mask': [NUM_RES_UNINDEXED, NUM_ATOMS],
    'unindexed_motif_aatype': [NUM_RES_UNINDEXED],
    'unindexed_motif_aatype_mask': [NUM_RES_UNINDEXED],
    'unindexed_motif_orig_res_id': [NUM_RES_UNINDEXED],
    'unindexed_motif_orig_chain_index': [NUM_RES_UNINDEXED],
    'chain_id': [NUM_RES],
    # Per-ligand features.
    'component_entity_id': [NUM_LIGANDS],
    'component_atomic_numbers': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    'component_hybridizations': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    'component_charges': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    'component_bond_types': [
        NUM_LIGANDS,
        NUM_LIGAND_ATOMS,
        NUM_LIGAND_ATOMS,
    ],
    'component_bonds_is_aromatic': [
        NUM_LIGANDS,
        NUM_LIGAND_ATOMS,
        NUM_LIGAND_ATOMS,
    ],
    'component_bonds_is_stereo': [
        NUM_LIGANDS,
        NUM_LIGAND_ATOMS,
        NUM_LIGAND_ATOMS,
    ],
    'component_padding_mask_1d': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    'component_padding_mask_2d': [
        NUM_LIGANDS,
        NUM_LIGAND_ATOMS,
        NUM_LIGAND_ATOMS,
    ],
    'component_fragment_indices': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    'component_frame_labels': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    # Per residue features used to combine proteins and ligands.
    'is_ligand_mask': [NUM_RES],
    # Other features used for debugging or eval:
    'component_id': [
        NUM_LIGANDS,
    ],
    'component_atom_names': [NUM_LIGANDS, NUM_LIGAND_ATOMS],
    'component_smiles_descriptor': [NUM_LIGANDS],
}

FEATURE_DTYPES: Final[Mapping[str, np.typing.DTypeLike]] = {
    'resolution': np.float32,
    'residue_index': np.int32,
    'aatype': np.int32,
    'all_atom_positions': np.float32,
    'all_atom_mask': np.int32,
    'fixed_atom_mask': np.int32,
    'seq_length': np.int32,
    'seq_mask': np.int32,
    'entity_id': np.int32,
    'asym_id': np.int32,
    'sym_id': np.int32,
    'is_protein': np.int32,
    'ligand_atomic_number': np.int32,
    'ligand_hybridization': np.int32,
    'ligand_charge': np.int32,
    # Per-ligand features.
    'component_entity_id': np.int32,
    'component_atomic_numbers': np.int32,
    'component_hybridizations': np.int32,
    'component_charges': np.float32,
    'component_bond_types': np.int32,
    'component_bonds_is_aromatic': np.int32,
    'component_bonds_is_stereo': np.int32,
    'component_padding_mask_1d': np.int32,
    'component_padding_mask_2d': np.int32,
    'component_fragment_indices': np.int32,
    'component_frame_labels': np.int32,
    # Per residue features used to combine proteins and ligands.
    'is_ligand_mask': np.int32,
    # Other features used for debugging or eval:
    'component_id': object,
    'component_atom_names': object,
    'component_smiles_descriptor': object,
    'chain_id': object,
    'ligand_atom_names': object,
    'unindexed_motif_atom_positions': np.float32,
    'unindexed_motif_atom_mask': np.int32,
    'unindexed_motif_aatype': np.int32,
    'unindexed_motif_aatype_mask': np.int32,
    'unindexed_motif_orig_res_id': np.int32,
    'unindexed_motif_orig_chain_index': np.int32,
    'fixed_seq_mask': np.int32,
    'b_factors': np.float32,
}


def filter_features(features: Mapping[str, Any]) -> dict[str, Any]:
  """Filters a dictionary of feature names by the feature names to keep."""
  feature_names_to_keep = set(
      SEQ_FEATURES + CHAIN_FEATURES + HOST_ONLY_FEATURES + LIGAND_FEATURES
  )
  return {
      name: value
      for name, value in features.items()
      if name in feature_names_to_keep
  }
