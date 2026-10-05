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

"""Structure constants for protein metrics and evaluation."""

from collections.abc import Mapping
from typing import Final

# Backbone atom names for protein residues.
PROTEIN_BACKBONE_ATOMS: Final[tuple[str, ...]] = ('N', 'CA', 'C')


def _build_equivalent_sidechain_atoms() -> dict[str, dict[str, str]]:
  """Builds mapping of symmetric equivalent sidechain atoms with reverse swaps."""
  atoms = {
      'ASP': {'OD1': 'OD2'},
      'GLU': {'OE1': 'OE2'},
      'VAL': {'CG1': 'CG2'},
      'LEU': {'CD1': 'CD2'},
      'PHE': {'CD1': 'CD2', 'CE1': 'CE2'},
      'TYR': {'CD1': 'CD2', 'CE1': 'CE2'},
      'ARG': {'NH1': 'NH2'},
  }
  for res_atoms in atoms.values():
    for k in list(res_atoms):
      res_atoms[res_atoms[k]] = k
  return atoms


EQUIVALENT_SIDECHAIN_ATOMS: Final[Mapping[str, Mapping[str, str]]] = (
    _build_equivalent_sidechain_atoms()
)
