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

"""Ligand constants."""

from __future__ import annotations

import collections

from alphaprotein_novo.constants import periodic_table
from alphaprotein_novo.constants import residue_names

# When concatenating ligand atom types as aatypes to protein aatypes,
# NUM_PROT_AATYPES specifies how many aatypes are reserved for proteins.
NUM_PROT_AATYPES = len(
    residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN  # 21.
)


# How many of the included ligand elements to consider.
NUM_LIGAND_ELEMS = 43

# Ligand elements ordered by descending frequency in PDB complexes
# (pdb_2021_08_02), excluding H and D.
# fmt: off
INCLUDED_ELEMS = [
    'C', 'O', 'N', 'P', 'S', 'Mg', 'Zn', 'Ca', 'Fe', 'F', 'Cl', 'Mn', 'K',
    'Ni', 'Cu', 'Br', 'Co', 'Cd', 'Hg', 'As', 'B', 'I', 'Pt', 'Mo', 'Be',
    'Al', 'Ba', 'Ru', 'V', 'Cs', 'Xe', 'Se', 'W', 'Au', 'Gd', 'Yb', 'Li',
    'Ir', 'Rb', 'Y', 'Pb', 'Os', 'Ag', 'Tl', 'U', 'Sm', 'Pr', 'Tb', 'Rh',
    'Si', 'Kr', 'Pd', 'Re', 'Eu', 'Ta', 'Te', 'Ar', 'La', 'Lu', 'Cr', 'Ga',
    'Ho', 'Sb', 'Sn', 'Zr', 'Ce', 'Er', 'Ti', 'Na', 'Th', 'Bi', 'Hf', 'In',
    'Am', 'Cf', 'Cm', 'Dy', 'Pu', 'Sc', 'Sr',
]
# fmt: on

LIGAND_ELEM_IDX = collections.OrderedDict(
    {k: i for i, k in enumerate(INCLUDED_ELEMS)}
)

ATOMIC_NUMBER_TO_IDX = collections.OrderedDict(
    {periodic_table.ATOMIC_NUMBER[k]: i for i, k in enumerate(LIGAND_ELEM_IDX)}
)
