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

"""Functions for residue mapping from input structure to designed protein."""

import collections
from collections.abc import Iterator, Mapping, Sequence
import dataclasses

from absl import logging
from alphafold3 import structure
from alphafold3.constants import mmcif_names
from alphafold3.structure import mmcif
from alphaprotein_novo.data import ligand_data
import numpy as np

ResTuple = tuple[str, int]
AtomTuple = tuple[str, int, str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResidueMap:
  """Map from input structure ("source") to designed protein ("target") residues.

  Each attribute has shape (num_tokens,), where num_tokens is the number of
  protein residues plus the number of ligand atoms in the designed (target)
  protein.

  Source residue indices are in internal naming scheme, not author naming
  scheme.
  """

  source_chain_ids: np.ndarray
  source_residue_ids: np.ndarray
  source_ligand_atom_names: np.ndarray
  target_chain_ids: np.ndarray
  target_residue_ids: np.ndarray
  is_reseq_residue: np.ndarray
  use_for_residue_mapping_only: bool = True

  def __post_init__(self):
    lengths = [
        len(self.source_chain_ids),
        len(self.source_residue_ids),
        len(self.source_ligand_atom_names),
        len(self.target_chain_ids),
        len(self.target_residue_ids),
        len(self.is_reseq_residue),
    ]
    if not all(length == lengths[0] for length in lengths):
      raise ValueError('All fields must have the same length.')

  @property
  def fixed_seq_mask(self) -> np.ndarray:
    """Returns a mask indicating whether the aatype of a token is fixed."""
    if self.use_for_residue_mapping_only:
      raise ValueError(
          'fixed_seq_mask not supported when use_for_residue_mapping_only is'
          ' True, i.e. ResidueMap was constructed without an input structure.'
      )
    return np.array([
        ligand_atom_name is not None
        or (source_chain_id is not None and not is_reseq_residue)
        for ligand_atom_name, source_chain_id, is_reseq_residue in zip(
            self.source_ligand_atom_names,
            self.source_chain_ids,
            self.is_reseq_residue,
        )
    ])

  @property
  def is_ligand_mask(self) -> np.ndarray:
    """Returns a mask indicating whether a token is a ligand."""
    if self.use_for_residue_mapping_only:
      raise ValueError(
          'is_ligand_mask not supported when use_for_residue_mapping_only is'
          ' True, i.e. ResidueMap was constructed without an input structure.'
      )
    return np.array([x is not None for x in self.source_ligand_atom_names])

  def get_source_resid_to_target_resid_dict(
      self,
  ) -> dict[ResTuple, ResTuple]:
    """Returns mapping from source to target (chain_id, res_id) tuples."""
    source2target = {}
    for c1, r1, c2, r2, ligand_atom_name in zip(
        self.source_chain_ids,
        self.source_residue_ids,
        self.target_chain_ids,
        self.target_residue_ids,
        self.source_ligand_atom_names,
        strict=True,
    ):
      if c1:  # Only include fixed residues.
        if ligand_atom_name is not None:
          # Does not handle multi-residue ligands.
          # Protein.to_structure() always assigns ligands a residue index of 1.
          source2target[(str(c1), int(r1))] = (str(c2), 1)
        else:
          source2target[(str(c1), int(r1))] = (str(c2), int(r2))
    return source2target


def reindex_atom_tuples(
    atoms: Sequence[AtomTuple],
    index_map: Mapping[ResTuple, ResTuple],
) -> Sequence[AtomTuple]:
  """Reindex atom tuples between residue numbering schemes."""
  return [
      index_map[(chain_id, res_id)] + (atom_name,)
      for chain_id, res_id, atom_name in atoms
  ]


def separate_chain_id_from_res_ids(chain_segment_str: str) -> tuple[str, str]:
  """Separates (possibly multi-character) chain id from res ids.

  Args:
    chain_segment_str: Chain segment specification, eg 'AA12' or 'A5-10'.

  Returns:
    Tuple of chain id and res ids (both as strings).
  """
  if not chain_segment_str:
    raise ValueError('Chain segment must be non-empty.')
  chain_id = ''
  # i is initialized prior to loop so it remains defined if loop body doesn't
  # execute or exits early.
  i = 0
  for i, _ in enumerate(chain_segment_str):
    if chain_segment_str[i].isalpha():
      chain_id += chain_segment_str[i]
    else:
      break
  res_ids = chain_segment_str[i:]
  if any(c.isalpha() for c in res_ids):
    raise ValueError(
        'Chain segment must be of the form "AA12" or "A5-10", got'
        f' {chain_segment_str}.'
    )
  return chain_id, res_ids


def ints_from_value_or_range(value_or_range: str) -> tuple[int, ...]:
  """Converts a string e.g. '17' or '21-24' to a tuple of integers."""
  if '-' in value_or_range:
    try:
      min_i, max_i = map(int, value_or_range.split('-'))
    except ValueError as e:
      raise ValueError(
          f'Could not parse range: {value_or_range}. Range must be specified as'
          ' "min-max".'
      ) from e
    return tuple(range(min_i, max_i + 1))
  else:
    try:
      return (int(value_or_range),)
    except ValueError as e:
      raise ValueError(f'Could not parse value: {value_or_range}.') from e


def value_or_range_list_from_ints(ints: Sequence[int]) -> list[str]:
  """Converts a list of integers to a list of strings e.g. ['17', '21-24']."""
  if not ints:
    return []
  prev, minval = None, None
  value_or_range_list = []
  for i in ints:
    if prev is None:
      minval = i
    else:
      if i - prev > 1:
        maxval = prev
        if minval == maxval:
          value_or_range_list.append(f'{minval}')
        else:
          value_or_range_list.append(f'{minval}-{maxval}')
        minval = i
    prev = i

  maxval = prev
  if minval == maxval:
    value_or_range_list.append(f'{minval}')
  else:
    value_or_range_list.append(f'{minval}-{maxval}')
  return value_or_range_list


def chain_and_residue_pairs(residues: str) -> Iterator[tuple[str, int]]:
  """Yields list of chain and residue indices.

  Examples:
      "A4,A6" yields [(A, 4), (A, 6)]
      "A5-7" yields [(A, 5), (A, 6), (A, 7)]

  Args:
    residues: String specification of residues.

  Yields:
    Iterator over the (chain, index) pairs.
  """
  if not residues:
    return
  for part in residues.split(','):
    chain, indices = separate_chain_id_from_res_ids(part)
    res_indices = ints_from_value_or_range(indices)
    for index in res_indices:
      yield (chain, index)


def motif_atoms_str_to_tuples(
    motif_atoms_str: str,
) -> Sequence[AtomTuple]:
  """Convert motif atom string to a list of (chain_id, res_id, atom_name)."""
  if not motif_atoms_str:
    return []
  motif_atoms_tuples = []
  for ch_res_atom in motif_atoms_str.split(' '):
    ch_res, atom_str = ch_res_atom.split(':')
    ch, res = separate_chain_id_from_res_ids(ch_res)
    atoms = [atom for atom in atom_str.split(',') if atom]
    motif_atoms_tuples.extend([(ch, int(res), atom) for atom in atoms])
  return motif_atoms_tuples


def motif_atoms_tuples_to_str(
    motif_atoms_tuples: Sequence[AtomTuple],
) -> str:
  """Convert list of (chain_id, res_id, atom_name) to a motif atom string."""
  if not motif_atoms_tuples:
    return ''
  ch_res_dict: dict[tuple[str, int], list[str]] = {}
  for ch, res, atom in motif_atoms_tuples:
    if (ch, res) not in ch_res_dict:
      ch_res_dict[(ch, res)] = []
    ch_res_dict[(ch, res)].append(atom)
  return ' '.join([
      f'{ch}{res}:{",".join(atoms)}' for (ch, res), atoms in ch_res_dict.items()
  ])


def sample_values_with_sum_in_range(
    *,
    mins: Mapping[int, int],
    maxs: Mapping[int, int],
    rng: np.random.Generator,
    min_sum: int | None = None,
    max_sum: int | None = None,
) -> dict[int, int]:
  """Sample uniformly in the given ranges, conditioned on their total sum.

  For each i, we sample x[i] uniformly in the range mins[i] <= x[i] <= maxs[i],
  conditioned on the total sum satisfying:  min_sum <= sum_i x[i] <= max_sum.

  Args:
    mins: Minimum value for each part.
    maxs: Maximum value for each part
    rng: Numpy random number generator.
    min_sum: The minimum allowed sum. If None, set to sum of values in mins.
    max_sum: The maximum allowed sum. If None, set to sum of values in maxs.

  Returns:
    Mapping of part keys to the length for that part.

  Raises:
    ValueError: If the constraints imposed by min_sum and max_sum are
      not satisfiable given the ranges in mins, maxs.
  """
  if min_sum is None:
    min_sum = sum(mins.values())
  if max_sum is None:
    max_sum = sum(maxs.values())

  if not (0 <= min_sum <= max_sum):
    raise ValueError(f'require 0 <= min_sum ({min_sum}) <= max_sum ({max_sum})')

  min_possible = sum(mins.values())
  max_possible = sum(maxs.values())
  if min_possible > max_sum or max_possible < min_sum:
    raise ValueError(
        'Cannot sample values in range'
        f' [{min_sum}, {max_sum}]: the range of possible sample values'
        f' is [{min_possible}, {max_possible}]'
    )

  # Sample a value for each part, until we get something in range. We sample
  # in increasing flexibility of ranges, since this will reduce the expected
  # number of iterations until we get something allowable (we get increasing
  # flexibility in the range as we go on).
  keys_by_flexibility = sorted(mins.keys(), key=lambda i: maxs[i] - mins[i])

  while True:
    total_sum = 0
    lengths = {}

    for i, key in enumerate(keys_by_flexibility):
      min_val = mins[key]
      max_val = maxs[key]

      # For the last part, clip the range we sample from. This still yields
      # uniform sampling.
      if i == len(keys_by_flexibility) - 1:
        min_val = max(min_val, min_sum - total_sum)
        max_val = min(max_val, max_sum - total_sum)
        if min_val > max_val:
          # Due to previously sampled uniform variables, we are unable to sample
          # further while satisfying the total_sum constraint. Set the total_sum
          # to -1 to indicate that we failed in this iteration.
          total_sum = -1
          break

      lengths[key] = int(rng.integers(min_val, max_val + 1))
      total_sum += lengths[key]

    if min_sum <= total_sum <= max_sum:
      return lengths


def _is_segment_designable(segment_str: str) -> bool:
  """Returns whether the motif segment string represents a designable segment."""
  return segment_str[0].isnumeric()


def _is_chain_designable(chain_str: str) -> bool:
  """Returns whether the motif chain string represents a designable chain."""
  return any(
      _is_segment_designable(segment) for segment in chain_str.split(',')
  )


def get_fixed_chains(motif_str: str) -> str:
  """Gets the fixed chains from the motif string."""
  fixed_chains = [
      chain for chain in motif_str.split('/') if not _is_chain_designable(chain)
  ]
  return '/'.join(fixed_chains)


def get_designable_chains(motif_str: str) -> str:
  """Gets the designable chains from the motif string."""
  designable_chains = [
      chain for chain in motif_str.split('/') if _is_chain_designable(chain)
  ]
  return '/'.join(designable_chains)


def extract_motifs_from_designable_chains(motif_str: str) -> str:
  """Extracts the motifs from the designable chains in the motif string."""
  designable_chains = [
      chain for chain in motif_str.split('/') if _is_chain_designable(chain)
  ]
  motifs = []
  for chain in designable_chains:
    for segment in chain.split(','):
      if not _is_segment_designable(segment):
        motifs.append(segment)
  return ','.join(motifs)


def _num_designable_chains(motif_str: str) -> int:
  """Returns the number of designable chains in the motif string."""
  return sum(_is_chain_designable(chain) for chain in motif_str.split('/'))


def _is_valid_sampled_motif_str(motif_str: str) -> bool:
  """Returns whether the motif string has fixed motif placement permutations and designable segment lengths."""
  if '|' in motif_str:
    return False
  for chain in motif_str.split('/'):
    for segment in chain.split(','):
      if _is_segment_designable(segment) and '-' in segment:
        return False
  return True


def get_designable_segments_length_ranges(
    segments: list[str],
) -> dict[int, list[int]]:
  """Returns the min and max lengths for each designable segment."""
  idx_min_max_s = {}
  for i, segment in enumerate(segments):
    if _is_segment_designable(segment):
      if '-' in segment:
        a, b = segment.split('-')
        idx_min_max_s[i] = [int(a), int(b)]
      else:
        idx_min_max_s[i] = [int(segment), int(segment)]
    else:
      length = len(list(chain_and_residue_pairs(segment)))
      idx_min_max_s[i] = [length, length]
  return idx_min_max_s


def sample_designable_lengths(
    motif_str: str,
    rng: np.random.Generator,
    length_str: str | None = None,
) -> str:
  """Turns all length ranges in `motif_str` into single values, subject to overall min/max lengths in `length_str`."""
  # Allows 0 designable chains.
  if _is_valid_sampled_motif_str(motif_str):
    return motif_str

  # Assume only 1 designable chain.
  if _num_designable_chains(motif_str) != 1:
    raise ValueError('Motif must have exactly 1 designable chain.')

  # Permute motif order and substitute into placeholders on motif string.
  if '|' in motif_str:
    motifs, motif_str = motif_str.split('|')
    motifs_permuted = rng.permutation(motifs.split(','))
    motif_str = motif_str.format(*motifs_permuted)

  min_sum, max_sum = None, None
  if length_str:
    if '-' in length_str:
      min_sum, max_sum = [int(x) for x in length_str.split('-')]
    else:
      min_sum = int(length_str)
      max_sum = int(length_str)

  new_chains = []
  for chain in motif_str.split('/'):
    if _is_chain_designable(chain):
      segments = chain.split(',')
      idx_min_max_s = get_designable_segments_length_ranges(segments)
      sampled_lengths = sample_values_with_sum_in_range(
          mins={k: v1 for k, (v1, _) in idx_min_max_s.items()},
          maxs={k: v2 for k, (_, v2) in idx_min_max_s.items()},
          min_sum=min_sum,
          max_sum=max_sum,
          rng=rng,
      )

      # Replace length ranges with single lengths, don't touch fixed segments.
      for i_segment, length in sampled_lengths.items():
        if _is_segment_designable(segments[i_segment]):
          segments[i_segment] = str(length)

      new_chains.append(','.join(segments))
    else:
      new_chains.append(chain)

  return '/'.join(new_chains)


def remove_missing_residues_from_motif_string(
    motif_str: str,
    input_struct: structure.Structure,
) -> str:
  """Removes residues from a motif string that are not present in the input structure."""
  new_chains = []
  for chain in motif_str.split('/'):
    new_segments = []
    for segment in chain.split(','):
      if _is_segment_designable(segment):
        new_segments.append(segment)
      else:
        chain_id, res_id_str = separate_chain_id_from_res_ids(segment)
        motif_res_ids = ints_from_value_or_range(res_id_str)
        filtered_struct = input_struct.filter(chain_id=chain_id)
        resolved_res_ids = filtered_struct.group_by_residue.res_id
        motif_res_ids = [i for i in motif_res_ids if i in resolved_res_ids]
        new_segment = ','.join(
            [chain_id + x for x in value_or_range_list_from_ints(motif_res_ids)]
        )
        new_segments.append(new_segment)
    new_chains.append(','.join(new_segments))
  return '/'.join(new_chains)


def _is_fixed_ligand_segment(
    segment_str: str,
    input_struct: structure.Structure,
) -> bool:
  """Returns whether `segment_str` in `input_struct` represents a ligand."""
  if _is_segment_designable(segment_str):
    return False
  chain_id, _ = separate_chain_id_from_res_ids(segment_str)
  chains_to_entity_type = {
      x['chain_id']: x['chain_type'] for x in input_struct.iter_chains()
  }
  return chains_to_entity_type.get(chain_id) in mmcif_names.LIGAND_CHAIN_TYPES


def _is_fixed_protein_segment(
    segment_str: str,
    input_struct: structure.Structure,
) -> bool:
  """Returns whether `segment_str` in `input_struct` represents a protein."""
  if _is_segment_designable(segment_str):
    return False
  chain_id, _ = separate_chain_id_from_res_ids(segment_str)
  chains_to_entity_type = {
      x['chain_id']: x['chain_type'] for x in input_struct.iter_chains()
  }
  return chains_to_entity_type.get(chain_id) in mmcif_names.PEPTIDE_CHAIN_TYPES


def motif_str_to_residue_map(
    *,
    motif_str: str,
    reseq_residues: str | None,
    input_struct: structure.Structure | None = None,
) -> ResidueMap:
  """Converts a motif string to a residue map.

  Used to create a residue map, which is used for 2 purposes:
  1. Create `fixed_seq_mask`, where residues whose aatypes are to be fixed are
  set to True, as well as `is_ligand_mask`.
  2. Create a source to target residue mapping for identifying where motif
  residues are in the designed sequence.

  For #1, you need to provide the input structure and a reseq_residues string.
  This will parse ligand atoms from the input structure and set any reseq
  residues to False in `fixed_seq_mask`.

  If you only need #2, you can call this without providing an input structure
  (and optionally set reseq_residues=''). Ligand atom tokens will not be parsed
  and `is_ligand_mask` and `fixed_seq_mask` properties will trigger a
  ValueError.

  Args:
    motif_str: The motif string to convert.
    reseq_residues: Residues to resequence, e.g. "A32,A65,A103", which will not
      be set to True in `fixed_seq_mask`.
    input_struct: The input structure to use for fixed segments.

  Returns:
    A ResidueMap object containing the source to target residue mapping.
  """
  if not _is_valid_sampled_motif_str(motif_str):
    raise ValueError(
        f'Motif string {motif_str} contains designable segment length ranges or'
        ' motif placement permutations. Call sample_designable_lengths() first.'
    )
  if reseq_residues is None:
    reseq_residues = ''

  source_chain_ids = []
  source_residue_ids = []
  source_ligand_atom_names = []
  target_chain_ids = []
  target_residue_ids = []

  all_segments_designable = True

  for i_c, chain in enumerate(motif_str.split('/')):
    curr_target_residue_id = 1
    if input_struct is not None and any(
        _is_fixed_ligand_segment(segment, input_struct)
        for segment in chain.split(',')
    ):
      if ',' in chain or '-' in chain:
        raise ValueError(
            'Ligand chains on the design must contain only a single source'
            ' residue (i.e. multi-residue input ligands are not supported).'
        )
      curr_target_residue_id = 0

    for segment in chain.split(','):
      if _is_segment_designable(segment):
        length = int(segment)

        source_chain_ids += [None] * length
        source_residue_ids += [None] * length
        source_ligand_atom_names += [None] * length

        target_chain_ids += [i_c + 1] * length
        target_residue_ids += range(
            curr_target_residue_id, curr_target_residue_id + length
        )

        if length > 0:
          curr_target_residue_id = target_residue_ids[-1] + 1

      else:
        all_segments_designable = False
        if input_struct is None or _is_fixed_protein_segment(
            segment, input_struct
        ):
          chain_id, res_str = separate_chain_id_from_res_ids(segment)
          res_ids = list(ints_from_value_or_range(res_str))

          source_chain_ids += [chain_id] * len(res_ids)
          source_residue_ids += res_ids
          source_ligand_atom_names += [None] * len(res_ids)

          target_chain_ids += [i_c + 1] * len(res_ids)
          target_residue_ids += range(
              curr_target_residue_id, curr_target_residue_id + len(res_ids)
          )

          curr_target_residue_id = target_residue_ids[-1] + 1

        elif input_struct is not None and _is_fixed_ligand_segment(
            segment, input_struct
        ):
          chain_id, res_str = separate_chain_id_from_res_ids(segment)
          res_ids = list(ints_from_value_or_range(res_str))
          ligand_struct = input_struct.filter(
              chain_id=chain_id
          ).without_hydrogen()
          ligand_ccd_code = ligand_struct.res_name[0]
          ligand_ccd_atom_names = ligand_data.get_ligand_ccd_atom_names(
              ligand_ccd_code,
              chemical_components_data=ligand_struct.chemical_components_data,
              default_atom_names=list(ligand_struct.atom_name),
          )
          num_ligand_atoms = len(ligand_ccd_atom_names)
          if num_ligand_atoms != len(ligand_struct.atom_name):
            logging.warning(
                'Ligand %s has %d heavy atoms in input structure but %d in CCD.'
                ' This can be due to a covalent ligand lacking a leaving'
                ' group.',
                ligand_ccd_code,
                len(ligand_struct.atom_name),
                num_ligand_atoms,
            )

          source_chain_ids += [chain_id] * num_ligand_atoms
          source_residue_ids += res_ids * num_ligand_atoms
          source_ligand_atom_names += ligand_ccd_atom_names

          target_chain_ids += [i_c + 1] * num_ligand_atoms
          target_residue_ids += range(
              curr_target_residue_id, curr_target_residue_id + num_ligand_atoms
          )

          # Ligand atoms are given consecutive residue ids here for diffusion
          # sampling. However, when result is converted to a structure, the
          # ligand becomes a single residue with res_id = 1. This will not
          # behave correctly for chains with multiple ligands.
          curr_target_residue_id = target_residue_ids[-1] + 1

        else:
          raise ValueError(f'Segment {segment} is neither protein nor ligand.')

  # Mark which residues are resequenced, so these can be excluded from
  # fixed_seq_mask.
  is_reseq_residue = [False] * len(source_chain_ids)
  source_chain_res_ids = list(zip(source_chain_ids, source_residue_ids))
  for chain_id, res_id in chain_and_residue_pairs(reseq_residues):
    if (chain_id, res_id) in source_chain_res_ids:
      is_reseq_residue[source_chain_res_ids.index((chain_id, res_id))] = True

  return ResidueMap(
      source_chain_ids=np.array(source_chain_ids, dtype=object),
      source_residue_ids=np.array(source_residue_ids, dtype=object),
      source_ligand_atom_names=np.array(source_ligand_atom_names, dtype=object),
      target_chain_ids=np.array(
          [mmcif.int_id_to_str_id(i) for i in target_chain_ids], dtype=object
      ),
      target_residue_ids=np.array(target_residue_ids, dtype=np.int32),
      is_reseq_residue=np.array(is_reseq_residue, dtype=bool),
      use_for_residue_mapping_only=(
          (input_struct is None) and (not all_segments_designable)
      ),
  )


def source_target_dict_to_motif_str(
    source_resid_to_target_resid: Mapping[ResTuple, ResTuple],
    designable_chain_length: int,
) -> str:
  """Converts a source to target residue map to a motif string.

  This function reconstructs a motif string from a dictionary mapping source
  residues to target residues, assuming chain 'A' is the single designable
  chain with total length `designable_chain_length`.
  If a chain appears in `source_resid_to_target_resid` but its chain id is not
  'A', it is assumed to be a fully fixed chain (e.g. ligand).

  Args:
    source_resid_to_target_resid: Mapping from source (chain_id, res_id) to
      target (chain_id, res_id) for fixed residues.
    designable_chain_length: Total length of residues in chain 'A'.

  Returns:
    A motif string.
  """
  target_chain_to_source: dict[str, list[tuple[int, str, int]]] = (
      collections.defaultdict(list)
  )
  for (c1, r1), (c2, r2) in source_resid_to_target_resid.items():
    target_chain_to_source[c2].append((r2, c1, r1))

  motif_chains = []
  all_chain_ids = {'A'} | set(target_chain_to_source.keys())
  sorted_chain_ids = sorted(list(all_chain_ids))

  for chain_id in sorted_chain_ids:
    is_designable_chain = chain_id == 'A'
    chain_length = designable_chain_length if is_designable_chain else None

    if chain_id not in target_chain_to_source:
      if not is_designable_chain:
        raise ValueError(
            f'Chain {chain_id} has no fixed residues and is not the'
            ' designable_chain.'
        )
      motif_chains.append(str(chain_length))
      continue

    residues = sorted(target_chain_to_source[chain_id])

    motif_segments = []
    current_target_res = 1

    i = 0
    while i < len(residues):
      target_res, source_chain, source_res = residues[i]

      # Add designable segment if needed.
      design_len = target_res - current_target_res
      if design_len > 0:
        if not is_designable_chain:
          raise ValueError(
              f'Chain {chain_id} is not designable but has gaps in'
              ' fixed residues.'
          )
        motif_segments.append(str(design_len))

      # Start of fixed segment.
      segment_source_chain = source_chain
      segment_source_start_res = source_res
      segment_source_end_res = source_res
      segment_target_end_res = target_res

      j = i + 1
      while j < len(residues):
        next_target_res, next_source_chain, next_source_res = residues[j]
        if (
            next_target_res == segment_target_end_res + 1
            and next_source_chain == segment_source_chain
            and next_source_res == segment_source_end_res + 1
        ):
          segment_source_end_res = next_source_res
          segment_target_end_res = next_target_res
          j += 1
        else:
          break

      if segment_source_start_res == segment_source_end_res:
        motif_segments.append(
            f'{segment_source_chain}{segment_source_start_res}'
        )
      else:
        motif_segments.append(
            f'{segment_source_chain}{segment_source_start_res}-{segment_source_end_res}'
        )

      current_target_res = segment_target_end_res + 1
      i = j

    if is_designable_chain:
      # Add trailing designable segment.
      design_len = chain_length - current_target_res + 1  # pyrefly: ignore[unsupported-operation]
      if design_len > 0:
        motif_segments.append(str(design_len))

    motif_chains.append(','.join(motif_segments))

  return '/'.join(motif_chains)
