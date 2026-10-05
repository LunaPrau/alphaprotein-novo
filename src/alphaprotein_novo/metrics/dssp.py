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

"""DSSP and secondary structure scoring metrics."""

from collections.abc import Sequence
import itertools
from typing import Any

from absl import logging
from alphafold3 import structure
from alphafold3.cpp import mkdssp


def secondary_structure_metrics(
    struct: structure.Structure,
    *,
    min_element_length: int = 3,
) -> dict[str, Any]:
  """Computes 3-state secondary structure, topology string, and topology length.

  Args:
    struct: Structure containing protein atoms.
    min_element_length: Minimum length of secondary structure elements to
      include in topology string.

  Returns:
    Dictionary containing:
      - topology_string: Secondary structure topology string.
      - topology_string_len: Length of the topology string.
  """
  sec_struct = calculate_secondary_struc(struct)
  topology = fold_topology(sec_struct, min_length=min_element_length)
  scores = secstruc_scores(sec_struct, min_length=min_element_length)
  return {
      'topology_string': topology['topology_string'],
      'topology_string_len': scores['topology_string_len'],
  }


def calculate_secondary_struc(prot: structure.Structure) -> list[str]:
  """Calculates 3-state secondary structure ('H', 'E', '-') for protein.

  Args:
    prot: Structure containing protein atoms.

  Returns:
    A list of 3-state secondary structure characters ('H', 'E', '-') for each
    residue in the protein.
  """
  protein = prot.filter_to_entity_type(protein=True)
  seq_length = (
      len(protein.group_by_residue.res_name) if protein.num_atoms > 0 else 0
  )
  if seq_length == 0:
    return []

  cif_string = protein.to_mmcif()
  try:
    dssp_output = mkdssp.get_dssp(cif_string)
  except RuntimeError as e:
    logging.warning('Error in DSSP: %s', e)
    return [''] * seq_length

  parse_row = False
  secstruc = []
  for row in dssp_output.splitlines():
    if parse_row:
      if len(row) < 17:
        continue
      aa = row[13:14]
      if aa == '!':
        continue
      ss_char = row[16:17]
      if ss_char in ('H', 'G', 'I', 'P'):
        simplified = 'H'
      elif ss_char in ('E', 'B'):
        simplified = 'E'
      else:
        simplified = '-'
      secstruc.append(simplified)
    elif row.startswith('  #  RESIDUE'):
      parse_row = True

  if len(secstruc) < seq_length:
    secstruc.extend([''] * (seq_length - len(secstruc)))
  elif len(secstruc) > seq_length:
    secstruc = secstruc[:seq_length]

  return secstruc


def fold_topology(
    secstruc: Sequence[str],
    min_length: int = 3,
) -> dict[str, str]:
  """Determines the fold topology given secondary structure assignment.

  Args:
    secstruc: String or sequence of DSSP secondary structure types ('H', 'E',
      '-').
    min_length: Minimum length of secondary structure element.

  Returns:
    Dictionary containing the topology string.
  """
  ss_length = 0
  topology_string = ''
  ss_rem = ''
  for ss in itertools.chain(secstruc, ('',)):
    ss_length += 1
    if ss_rem != ss:
      if ss_length >= min_length and ss_rem in ('H', 'E'):
        topology_string += ss_rem
      ss_length = 0
    ss_rem = ss

  return {
      'topology_string': topology_string,
  }


def secstruc_scores(
    secstruc: Sequence[str],
    min_length: int = 3,
) -> dict[str, int]:
  """Computes secondary structure topology string length.

  Args:
    secstruc: Sequence of 3-state secondary structure characters ('H', 'E',
      '-').
    min_length: Minimum length of secondary structure element to include in
      topology string.

  Returns:
    Dictionary containing topology_string_len.
  """
  fold_topology_result = fold_topology(secstruc, min_length=min_length)
  return {
      'topology_string_len': len(fold_topology_result['topology_string']),
  }
