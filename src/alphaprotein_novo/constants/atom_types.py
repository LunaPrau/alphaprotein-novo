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

"""List of atom types with reverse look-up."""

from collections.abc import Mapping
import sys
from typing import Final

from alphaprotein_novo.constants import residue_names

# Note:
# `sys.intern` places the values in the Python internal db for fast lookup.

# 37 common residue atoms.
N = sys.intern('N')
CA = sys.intern('CA')
C = sys.intern('C')
CB = sys.intern('CB')
O = sys.intern('O')
CG = sys.intern('CG')
CG1 = sys.intern('CG1')
CG2 = sys.intern('CG2')
OG = sys.intern('OG')
OG1 = sys.intern('OG1')
SG = sys.intern('SG')
CD = sys.intern('CD')
CD1 = sys.intern('CD1')
CD2 = sys.intern('CD2')
ND1 = sys.intern('ND1')
ND2 = sys.intern('ND2')
OD1 = sys.intern('OD1')
OD2 = sys.intern('OD2')
SD = sys.intern('SD')
CE = sys.intern('CE')
CE1 = sys.intern('CE1')
CE2 = sys.intern('CE2')
CE3 = sys.intern('CE3')
NE = sys.intern('NE')
NE1 = sys.intern('NE1')
NE2 = sys.intern('NE2')
OE1 = sys.intern('OE1')
OE2 = sys.intern('OE2')
CH2 = sys.intern('CH2')
NH1 = sys.intern('NH1')
NH2 = sys.intern('NH2')
OH = sys.intern('OH')
CZ = sys.intern('CZ')
CZ2 = sys.intern('CZ2')
CZ3 = sys.intern('CZ3')
NZ = sys.intern('NZ')
OXT = sys.intern('OXT')

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
# pyformat: disable
RESIDUE_ATOMS: Mapping[str, tuple[str, ...]] = {
    residue_names.ALA: (C, CA, CB, N, O),
    residue_names.ARG: (C, CA, CB, CG, CD, CZ, N, NE, O, NH1, NH2),
    residue_names.ASN: (C, CA, CB, CG, N, ND2, O, OD1),
    residue_names.ASP: (C, CA, CB, CG, N, O, OD1, OD2),
    residue_names.CYS: (C, CA, CB, N, O, SG),
    residue_names.GLN: (C, CA, CB, CG, CD, N, NE2, O, OE1),
    residue_names.GLU: (C, CA, CB, CG, CD, N, O, OE1, OE2),
    residue_names.GLY: (C, CA, N, O),
    residue_names.HIS: (C, CA, CB, CG, CD2, CE1, N, ND1, NE2, O),
    residue_names.ILE: (C, CA, CB, CG1, CG2, CD1, N, O),
    residue_names.LEU: (C, CA, CB, CG, CD1, CD2, N, O),
    residue_names.LYS: (C, CA, CB, CG, CD, CE, N, NZ, O),
    residue_names.MET: (C, CA, CB, CG, CE, N, O, SD),
    residue_names.PHE: (C, CA, CB, CG, CD1, CD2, CE1, CE2, CZ, N, O),
    residue_names.PRO: (C, CA, CB, CG, CD, N, O),
    residue_names.SER: (C, CA, CB, N, O, OG),
    residue_names.THR: (C, CA, CB, CG2, N, O, OG1),
    residue_names.TRP:
        (C, CA, CB, CG, CD1, CD2, CE2, CE3, CZ2, CZ3, CH2, N, NE1, O),
    residue_names.TYR: (C, CA, CB, CG, CD1, CD2, CE1, CE2, CZ, N, O, OH),
    residue_names.VAL: (C, CA, CB, CG1, CG2, N, O),
}
# pyformat: enable

# Used to identify backbone for alignment and distance calculation for sterics.
PROTEIN_BACKBONE_ATOMS: tuple[str, ...] = (N, CA, C)
PROTEIN_BACKBONE_WITH_OXYGEN: tuple[str, ...] = PROTEIN_BACKBONE_ATOMS + (O,)

# Used when we need to store atom data in a format that requires fixed atom data
# size for every protein residue (e.g. a numpy array).
# fmt: off
ATOM37: tuple[str, ...] = (
    N, CA, C, CB, O, CG, CG1, CG2, OG, OG1, SG, CD, CD1, CD2, ND1, ND2, OD1,
    OD2, SD, CE, CE1, CE2, CE3, NE, NE1, NE2, OE1, OE2, CH2, NH1, NH2, OH, CZ,
    CZ2, CZ3, NZ, OXT,
)
# fmt: on
ATOM37_ORDER: Mapping[str, int] = {name: i for i, name in enumerate(ATOM37)}
ATOM37_NUM: Final[int] = len(ATOM37)  # := 37.
