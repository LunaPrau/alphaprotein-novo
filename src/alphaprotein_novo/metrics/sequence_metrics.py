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

"""Sequence-based metrics for protein design evaluation."""

import collections
from collections.abc import Sequence
import re
from typing import Any

from alphafold3 import structure
from alphaprotein_novo.constants import residue_names

MOTIF_REGEX_BLOCKLIST = (
    'WSHPQFEK',  # Strep-tag
    'EQKLISEEDL',  # c-Myc-tag
    'ENLYFQ[SGAMCH]',  # TEV protease
    'LEVLFQGP',  # 3C protease cleavage site
    'YPYDVPDYA',  # HA-tag
    'HHHHH',  # His-tag
    'DYKDDDDK',  # FLAG-tag
    'GKPIPNPLLGLDST',  # V5-tag
    'LVPRGS',  # Thrombin cleavage
)


def _blocklisted_motifs(sequence: str) -> Sequence[str]:
  """Returns a list of blocklisted motifs contained in the sequence."""
  blocklisted = []
  for pattern in MOTIF_REGEX_BLOCKLIST:
    if re.search(pattern, sequence):
      blocklisted.append(pattern)
  return blocklisted


def _max_length_consecutive_chars(string: str) -> int:
  """Returns the length of longest substring of identical characters."""
  if not string:
    return 0
  max_length = 1
  current_length = 1
  previous_char = string[0]
  for current_char in string[1:]:
    if current_char == previous_char:
      current_length += 1
      max_length = max(max_length, current_length)
    else:
      current_length = 1
    previous_char = current_char
  return max_length


def max_single_aatype_frac(aatype_sequence: str) -> float:
  """Returns the fraction of the sequence that is a single aatype."""
  if not aatype_sequence:
    return 0.0
  aatype_counts = collections.Counter(aatype_sequence)
  return max(aatype_counts.values()) / len(aatype_sequence)


def sequence_metrics(aatype_sequence: str) -> dict[str, Any]:
  """Computes sequence metrics from the amino acid sequence (assumes uppercase).

  Args:
    aatype_sequence: Amino acid sequence string.

  Returns:
    Dictionary containing:
      - seq_len: Length of the sequence.
      - max_single_aatype_frac: Maximum fraction of any single amino acid type.
      - max_consecutive_identical_aatypes: Maximum run length of identical amino
        acids.
      - num_blocklisted_motifs: Count of matching blocklisted motifs.
      - unknown_count: Count of unknown ('X') amino acids.
  """
  return {
      'seq_len': len(aatype_sequence),
      'max_single_aatype_frac': max_single_aatype_frac(aatype_sequence),
      'max_consecutive_identical_aatypes': _max_length_consecutive_chars(
          aatype_sequence
      ),
      'num_blocklisted_motifs': len(_blocklisted_motifs(aatype_sequence)),
      'unknown_count': aatype_sequence.count('X'),
  }


def sequence_metrics_from_structure(
    struct: structure.Structure,
) -> dict[str, Any]:
  """Extracts primary protein sequence from Structure and computes sequence metrics.

  Args:
    struct: Structure containing protein atoms.

  Returns:
    Dictionary containing sequence metrics (seq_len, max_single_aatype_frac,
    max_consecutive_identical_aatypes, num_blocklisted_motifs, unknown_count).
  """
  protein = struct.filter_to_entity_type(protein=True)
  seq = ''.join([
      residue_names.CCD_NAME_TO_ONE_LETTER.get(r, 'X')
      for r in protein.group_by_residue.res_name
  ])
  return sequence_metrics(seq)
