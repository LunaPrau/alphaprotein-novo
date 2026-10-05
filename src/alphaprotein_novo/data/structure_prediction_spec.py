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

"""Pydantic models and serialization for structure prediction specifications."""

import re
from typing import Any, Self

import pydantic

_ATOM_SPEC_RE = re.compile(r'^([A-Za-z]+)(-?\d+):(\S+)$')


class _BaseModel(pydantic.BaseModel):
  """Frozen Pydantic model that rejects unknown fields."""

  model_config = pydantic.ConfigDict(frozen=True, extra='forbid')

  def to_dict(self) -> dict[str, Any]:
    return self.model_dump(exclude_none=True)

  @classmethod
  def from_dict(cls, d: Any) -> Self:
    return cls.model_validate(d)


class BondSpec(_BaseModel):
  """Specifies a covalent bond between two atoms (`atom_1` and `atom_2`)."""

  atom_1: tuple[str, int, str]
  atom_2: tuple[str, int, str]

  @pydantic.field_validator('atom_1', 'atom_2', mode='before')
  @classmethod
  def _parse_atom(cls, value: Any) -> Any:
    if not isinstance(value, str):
      return value
    if m := _ATOM_SPEC_RE.fullmatch(value.strip()):
      return (m.group(1), int(m.group(2)), m.group(3))
    raise ValueError(
        f'Invalid atom specification {value!r}; expected'
        ' "<chain><res_id>:<atom_name>" (e.g. "A1:SG").'
    )


class LigandSpec(_BaseModel):
  """Specifies a ligand component for structure prediction.

  Attributes:
    id: Unique ligand chain identifier in the assembly (e.g. 'B', 'C', 'L').
      Referenced as `chain_id` in `BondSpec` covalent bond definitions.
    ccd_code: Chemical Component Dictionary code (e.g. 'ATP', 'HEM') when
      `smiles` is not set, or custom residue name (e.g. 'pd00') when `smiles` is
      set.
    smiles: Optional SMILES string defining the chemical structure for custom
      non-CCD ligands.
    name: Optional descriptive name for the ligand, used to populate the
      AlphaFold 3 ligand description field.
  """

  id: str
  ccd_code: str = pydantic.Field(min_length=1)
  smiles: str | None = pydantic.Field(default=None, min_length=1)
  name: str | None = pydantic.Field(default=None, min_length=1)


class StructurePredictionSpec(_BaseModel):
  """Definition of a specific physical state for structure prediction and scoring.

  Attributes:
    name: Unique identifier for this specification (e.g. 'monomer', 'complex').
    ligands: Sequence of non-polymer ligands (defined by CCD code or SMILES) to
      model in complex with the protein chain. If empty, only the designed
      protein chain is modeled (unbound monomer).
    covalent_bonds: Sequence of covalent bonds between specific atoms
      (protein-ligand, ligand-ligand, or protein-protein).
  """

  name: str
  ligands: tuple[LigandSpec, ...] = ()
  covalent_bonds: tuple[BondSpec, ...] = ()
