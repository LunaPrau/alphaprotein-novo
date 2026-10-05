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

"""Declarative JSON manifest describing a batch of protein design jobs.

A manifest is a JSON document listing one or more design jobs. Each job pairs
the fields of a `motif_spec.MotifSpec` with the run-level settings that vary
per design, most importantly how many designs to generate for that spec.

A document has three optional blocks around the required `designs` list:

  - `settings`: run-level values that are not per-job, such as `model_dir`.
  - `defaults`: per-job fields applied to every design that does not set them.
  - `designs`: the jobs themselves.
  - `resequence`, `folding`, `evaluation`: what the stages after generation do
    with the designs. The pipeline driver reads these and translates them into
    the stages' existing path-and-flag interfaces; the one exception is
    `folding.states`, which generation records in each design's metadata so
    that folding a design directly behaves the same as folding it through the
    driver.

Every path in a manifest is resolved relative to the manifest's own directory
unless it is absolute, so a manifest and its inputs can be relocated together.

Example:

```json
{
  "defaults": {"input_file": "motif.cif", "is_author_naming": true},
  "designs": [
    {
      "name": "kemp_eliminase",
      "motif_str": "A1,A2,A3|10-100,{},2-80,{},2-80,{},10-100/B1",
      "motif_atoms": "A1:NE2,ND1 A2:OD1 A3:ND2",
      "num_designs": 100
    }
  ]
}
```
"""

from collections.abc import Mapping, Sequence, Set
import dataclasses
import hashlib
import itertools
import json
import os
import subprocess
import sys
from typing import Any, Self

from alphaprotein_novo.data import structure_prediction_spec
from etils import epath
import pydantic

# Constant width for design numbers in filenames. Keeps `ls` output ordered
DESIGN_NUM_WIDTH = 4

# Per-job index, stable entry point for discovering everything else.
INDEX_FILENAME = 'designs.json'
METADATA_DIRNAME = 'metadata'

# Files associated with each design.
METADATA_SUFFIX = '_metadata.json'
PDB_SUFFIX = '.pdb'
CIF_SUFFIX = '.cif'
FASTA_SUFFIX = '.fa'
EVALUATION_SUFFIX = '_evaluation.json'
FOLDED_CIF_SUFFIX = '_folded.cif'
CONFIDENCES_SUFFIX = '_confidences.json'

# The resequenced (LigandMPNN-redesigned) structure.
RESEQ_CIF_SUFFIX = '_reseq.cif'

# The per-design re-indexed ground-truth motif structure.
MOTIF_CIF_SUFFIX = '_motif.cif'

# Which structure is input to AlphaFold 3.
GENERATED_FOLDING_INPUT = 'generated'
RESEQUENCED_FOLDING_INPUT = 'resequenced'
FOLDING_INPUTS = (GENERATED_FOLDING_INPUT, RESEQUENCED_FOLDING_INPUT)

# Seed used by folding when a manifest does not choose one. Unrelated to the
# generation seeds below: it perturbs structure prediction, not the design.
DEFAULT_FOLDING_SEED = 230

# First design's seed when neither `seeds` nor `seed_start` is set.
# Resuming partial batch makes the same designs as running from scratch.
DEFAULT_SEED_START = 0

# Fields of `DesignJob` that are forwarded verbatim to `MotifSpec`.
_MOTIF_SPEC_FIELDS = (
    'name',
    'description',
    'input_file',
    'motif_str',
    'motif_atoms',
    'is_author_naming',
    'reseq_residues',
    'seq_length',
    'unindexed_motif_residues',
)


# Keys holding paths, which are resolved against the manifest's directory.
_DESIGN_PATH_KEYS = ('input_file', 'partial_diffusion_input_file')
_SETTINGS_PATH_KEYS = ('model_dir', 'output_dir')
_EVALUATION_PATH_KEYS = ('reference_cif',)


def _file_digest(path: str) -> str:
  """Returns the hex SHA-256 digest of a file's contents."""
  file = epath.Path(path)
  if not file.exists():
    raise FileNotFoundError(
        f'Cannot hash design spec: input structure not found: {file}'
    )
  return hashlib.sha256(file.read_bytes()).hexdigest()


class _BaseModel(pydantic.BaseModel):
  """Base model rejecting unknown keys so typos fail loudly."""

  model_config = pydantic.ConfigDict(extra='forbid', frozen=True)


class DesignJob(_BaseModel):
  """One design problem plus the settings controlling how many to generate.

  Contains the user-facing fields of `motif_spec.MotifSpec` while avoiding JAX
  and Structure dependencies of `MotifSpec`, plus fields
  controlling the number of designs and output locations. Convert to a
  `MotifSpec` using `motif_spec.MotifSpec.from_dict(job.motif_spec_kwargs())`.

  Attributes:
    name: Unique job identifier. Also the default filename prefix, so it must be
      usable as a path component.
    input_file: Path to the mmCIF holding the motifs to condition on.
    motif_str: Motif and linker specification. See `motif_spec.MotifSpec`.
    is_author_naming: Whether residue numbering uses author naming.
    description: Free text description of the design problem.
    motif_atoms: Subset of atoms to fix within each motif residue.
    reseq_residues: Motif residues allowed to change amino acid type.
    seq_length: Optional total design length constraint.
    unindexed_motif_residues: Motif residues for unindexed conditioning. Setting
      this is what switches the spec into unindexed mode.
    num_designs: How many designs to generate for this spec. Each design
      re-parses the spec with its own seed, resampling linker lengths.
    seeds: Explicit per-design seeds, mainly for tests and for reproducing a
      specific design. Mutually exclusive with `seed_start`.
    seed_start: Seed of design 0; design i uses `seed_start + i`. Defaults to
      `DEFAULT_SEED_START`. Set it in the manifest's `defaults` block to shift
      every job at once.
    num_sampling_steps: Reverse diffusion timesteps per design.
    output_prefix: Overrides `name` as the per-design filename prefix.
    partial_diffusion_input_file: Path to an mmCIF structure to partially
      diffuse. Must be set together with `partial_diffusion_num_steps`.
    partial_diffusion_num_steps: Reverse diffusion timesteps to unroll when
      partially diffusing. Must not exceed `num_sampling_steps`.
  """

  name: str
  input_file: str
  motif_str: str
  is_author_naming: bool
  description: str = ''
  motif_atoms: str = ''
  reseq_residues: str | None = None
  seq_length: str | None = None
  unindexed_motif_residues: str | None = None

  num_designs: pydantic.PositiveInt | None = None
  seeds: tuple[int, ...] | None = None
  seed_start: int | None = None
  num_sampling_steps: pydantic.PositiveInt = 1000
  output_prefix: str | None = None
  partial_diffusion_input_file: str | None = None
  partial_diffusion_num_steps: pydantic.PositiveInt | None = None

  @pydantic.field_validator('name', 'output_prefix')
  @classmethod
  def _check_path_component(cls, value: str | None) -> str | None:
    """Rejects values that cannot be used as a single path component."""
    if value is None:
      return None
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
      raise ValueError(
          f'{value!r} must be a non-empty single path component containing no'
          ' slashes.'
      )
    return value

  @pydantic.model_validator(mode='after')
  def _check_consistency(self) -> Self:
    """Validates cross-field constraints."""
    if self.seeds is not None:
      if self.seed_start is not None:
        raise ValueError(
            f'Job {self.name!r} sets both seeds and seed_start; seed_start only'
            ' applies when seeds are generated rather than listed.'
        )
      if not self.seeds:
        raise ValueError(f'Job {self.name!r} has an empty seeds list.')
      if self.num_designs is not None and self.num_designs != len(self.seeds):
        raise ValueError(
            f'Job {self.name!r} sets num_designs={self.num_designs} but lists'
            f' {len(self.seeds)} seeds. Omit num_designs, or make them agree.'
        )
    if (self.partial_diffusion_input_file is None) != (
        self.partial_diffusion_num_steps is None
    ):
      raise ValueError(
          f'Job {self.name!r} sets only one of partial_diffusion_input_file'
          ' and partial_diffusion_num_steps; partial diffusion needs both.'
      )
    if (
        self.partial_diffusion_num_steps is not None
        and self.partial_diffusion_num_steps > self.num_sampling_steps
    ):
      raise ValueError(
          f'Job {self.name!r} has partial_diffusion_num_steps'
          f' ({self.partial_diffusion_num_steps}) greater than'
          f' num_sampling_steps ({self.num_sampling_steps}).'
      )
    return self

  @property
  def design_count(self) -> int:
    """Number of designs this job produces."""
    if self.seeds is not None:
      return len(self.seeds)
    return self.num_designs if self.num_designs is not None else 1

  @property
  def prefix(self) -> str:
    """Filename prefix for this job's designs."""
    return self.output_prefix or self.name

  def seed_for(self, design_num: int) -> int:
    """Returns the seed for a zero-based design number within this job."""
    if not 0 <= design_num < self.design_count:
      raise IndexError(
          f'Design number {design_num} out of range for job {self.name!r} with'
          f' {self.design_count} designs.'
      )
    if self.seeds is not None:
      return self.seeds[design_num]
    start = (
        self.seed_start if self.seed_start is not None else DEFAULT_SEED_START
    )
    return start + design_num

  def motif_spec_kwargs(self) -> dict[str, Any]:
    """Returns the subset of fields that construct a `MotifSpec`."""
    return {field: getattr(self, field) for field in _MOTIF_SPEC_FIELDS}

  def spec_hash(self) -> str:
    """Returns a stable hash, prefixed with its algorithm, of this job's spec.

    Only the fields that affect the generated structures are hashed. This
    includes the contents (not paths) of structure input files.

    Raises:
      FileNotFoundError: If an input structure this job references is missing.
    """
    payload: dict[str, Any] = self.motif_spec_kwargs()
    payload.pop('input_file')
    payload['input_file_sha256'] = _file_digest(self.input_file)
    payload['num_sampling_steps'] = self.num_sampling_steps
    payload['partial_diffusion_input_file_sha256'] = (
        None
        if self.partial_diffusion_input_file is None
        else _file_digest(self.partial_diffusion_input_file)
    )
    payload['partial_diffusion_num_steps'] = self.partial_diffusion_num_steps
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    return f'sha256:{hashlib.sha256(encoded).hexdigest()}'


def _merge_defaults(
    defaults: Mapping[str, Any], entries: Sequence[Any]
) -> list[Any]:
  """Returns one entry per design, with the manifest defaults merged in."""
  merged: list[Any] = []
  for index, entry in enumerate(entries):
    # Entries are already built jobs when a Manifest is constructed directly.
    if isinstance(entry, DesignJob):
      merged.append(entry)
      continue
    if not isinstance(entry, Mapping):
      raise ValueError(
          f'Design at index {index} must be a JSON object, got'
          f' {type(entry).__name__}.'
      )
    combined = {**defaults, **entry}
    # Default `seed_start` is ignored when explicit `seeds` are given.
    # Default `num_designs` will trigger a validation error if it disagrees
    # with the length of `seeds`.
    if 'seeds' in entry and 'seed_start' not in entry:
      combined.pop('seed_start', None)
    merged.append(combined)
  return merged


class RunSettings(_BaseModel):
  """Run-level settings that apply to every job in a manifest.

  These are separate from `defaults` because they are not per-job fields: a
  batch has one model and one output root, not one per design. Every setting
  here is optional and can be overridden by the corresponding command line
  flag, so a manifest stays relocatable if it omits them.

  Attributes:
    model_dir: Directory containing the generator weights. Recorded because
      different weights produce different designs, so it is provenance rather
      than mere invocation detail.
    output_dir: Root directory for outputs. Usually better left unset and passed
      as a flag, since an absolute path here stops the manifest from being
      shareable.
  """

  model_dir: str | None = None
  output_dir: str | None = None


class ResequenceSettings(_BaseModel):
  """What resequencing does with each generated design.

  Attributes:
    enabled: Whether to resequence at all. Deliberately has no default: folding
      a generated sequence and folding a resequenced one answer different
      questions, so the manifest has to say which run this is.
    temperature: LigandMPNN sampling temperature.
    num_sequences: Number of redesigned sequences to sample per generated
      backbone. When greater than 1, each redesigned sequence is written as an
      independent `<prefix>_seq<NN>` design entry.
    use_side_chain_context: Whether to pass fixed-residue side-chain atom
      coordinates to LigandMPNN (`--ligand_mpnn_use_side_chain_context`).
      Defaults to True, which improves design outcomes; set to False to
      reproduce the paper's settings.
  """

  enabled: bool
  temperature: float = 0.1
  num_sequences: pydantic.PositiveInt = 1
  use_side_chain_context: bool = True


def resequenced_prefix(
    prefix: str,
    *,
    reseq_index: int,
    num_sequences: int = 1,
) -> str:
  """Returns the output prefix for one resequenced sequence of a design."""
  if not 0 <= reseq_index < num_sequences:
    raise ValueError(
        f'reseq_index {reseq_index} out of range for'
        f' num_sequences={num_sequences}.'
    )
  return f'{prefix}_seq{reseq_index:02d}'


def _parse_states(
    states: Sequence[Mapping[str, Any]],
) -> tuple[structure_prediction_spec.StructurePredictionSpec, ...]:
  """Returns authored folding states parsed into specs.

  Args:
    states: The `folding.states` entries, as `StructurePredictionSpec.to_dict`
      writes them.

  Raises:
    ValueError: If a state is unusable, which is worth catching here because
      the alternative is a batch that generates for hours and then fails.
  """
  if not states:
    raise ValueError(
        'folding.states is empty; omit it to derive the states from the motif.'
    )
  for state in states:
    if 'name' not in state:
      raise ValueError(f'folding.states entry {dict(state)} has no name.')
  specs = tuple(
      structure_prediction_spec.StructurePredictionSpec.from_dict(state)
      for state in states
  )
  names = [spec.name for spec in specs]
  if len(set(names)) != len(names):
    raise ValueError(
        f'folding.states names a state more than once: {names}. Each state'
        ' names the files it produces, so they have to be distinct.'
    )
  return specs


class FoldingSettings(_BaseModel):
  """What structure prediction does with each design.

  Attributes:
    inputs: Which structure of each design to fold. Exactly one for now; it is a
      list so that folding several per design later does not rename the field.
      Filled in from `resequence.enabled` when the manifest leaves it out.
    seeds: Structure prediction seeds. Every folded state is predicted once per
      seed.
    states: Structures to predict for each design, as `StructurePredictionSpec`
      documents. Left unset, generation derives them from the motif, which is
      what every existing manifest relies on.
  """

  inputs: tuple[str, ...] = (GENERATED_FOLDING_INPUT,)
  seeds: tuple[int, ...] = (DEFAULT_FOLDING_SEED,)
  states: tuple[dict[str, Any], ...] | None = None

  @pydantic.field_validator('inputs')
  @classmethod
  def _check_inputs(cls, value: tuple[str, ...]):
    """Rejects folding inputs the pipeline cannot honour."""
    if len(value) != 1:
      raise ValueError(
          f'folding.inputs lists {len(value)} inputs, but folding more than'
          ' one structure per design is not implemented yet.'
      )
    unknown = sorted(set(value) - set(FOLDING_INPUTS))
    if unknown:
      raise ValueError(
          f'folding.inputs has unknown entries {unknown}; expected'
          f' {list(FOLDING_INPUTS)}.'
      )
    return value

  @pydantic.field_validator('seeds')
  @classmethod
  def _check_seeds(cls, value: tuple[int, ...]):
    """Rejects empty or duplicate seed lists."""
    if not value:
      raise ValueError('folding.seeds must list at least one seed.')
    if len(set(value)) != len(value):
      raise ValueError(
          f'folding.seeds lists a seed more than once: {list(value)}.'
      )
    return value

  @pydantic.field_validator('states')
  @classmethod
  def _check_states(cls, value: tuple[dict[str, Any], ...] | None):
    """Rejects states structure prediction could not run."""
    if value is not None:
      _parse_states(value)
    return value

  def state_specs(
      self,
  ) -> tuple[structure_prediction_spec.StructurePredictionSpec, ...] | None:
    """Returns the authored states, or None to derive them from the motif."""
    if self.states is None:
      return None
    return _parse_states(self.states)

  def input_structure(self) -> str:
    """Returns the single structure folding reads for each design."""
    return self.inputs[0]


class EvaluationSettings(_BaseModel):
  """What evaluation measures the designs against.

  Attributes:
    reference_cif: Ground truth structure to score against. Left unset, each
      design is scored against the motif recorded in its own metadata.
    suite: Optional evaluation suite name (e.g. 'kemp_eliminase',
      'serine_esterase', 'dehp_esterase', 'carbene_transfer',
      'nitrene_transfer') specifying which enzyme-specific evaluation suite to
      run.
  """

  reference_cif: str | None = None
  suite: str | None = None


class Manifest(_BaseModel):
  """A batch of design jobs.

  Build one with `Manifest.from_file` for a manifest on disk, or with
  `Manifest.model_validate` for an already-decoded document. Only `from_file`
  resolves relative input paths, since only it knows the directory they are
  relative to.

  Attributes:
    settings: Run-level settings applying to the whole batch.
    designs: The jobs to run, with the manifest `defaults` already merged in.
    resequence: What resequencing does, or None if the manifest is silent.
    folding: What structure prediction does.
    evaluation: What evaluation measures against.
  """

  settings: RunSettings = RunSettings()
  designs: tuple[DesignJob, ...]
  resequence: ResequenceSettings | None = None
  folding: FoldingSettings = FoldingSettings()
  evaluation: EvaluationSettings = EvaluationSettings()

  @classmethod
  def from_file(cls, path: epath.PathLike) -> Self:
    """Loads and validates a manifest from a JSON file.

    Relative `input_file` paths are resolved against the manifest's own
    directory, so a manifest and its inputs can be relocated together.

    Args:
      path: Path to the manifest JSON file.

    Returns:
      The validated manifest.

    Raises:
      FileNotFoundError: If the manifest does not exist.
      ValueError: If the manifest is malformed.
    """
    manifest_path = epath.Path(path)
    if not manifest_path.exists():
      raise FileNotFoundError(f'Manifest not found: {manifest_path}')
    document = json.loads(manifest_path.read_text())
    return cls.model_validate(
        _resolve_paths(document=document, base_dir=manifest_path.parent)
    )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _normalize_document(cls, data: Any) -> Any:
    """Checks the document's shape and merges `defaults` into each entry.

    Everything here is a property of the document alone, so it runs on every
    construction path and `model_validate` needs no help to be correct.
    Resolving relative paths is deliberately absent: that needs a directory to
    resolve against, which only `from_file` knows.
    """
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
      data = {'designs': list(data)}
    if not isinstance(data, Mapping):
      raise ValueError(
          'Manifest must be a JSON object or a list of design objects, got'
          f' {type(data).__name__}.'
      )
    # `defaults` is the one accepted key that is not a field: it is merged
    # into the entries below rather than kept.
    allowed = set(cls.model_fields) | {'defaults'}
    unknown = set(data) - allowed
    if unknown:
      raise ValueError(
          f'Unknown top level manifest keys: {sorted(unknown)}. Expected any'
          f' of: {sorted(allowed)}.'
      )
    raw: dict[str, Any] = dict(data)
    # Consumed rather than kept as a field: once merged into the entries it no
    # longer describes anything about the manifest.
    defaults = raw.pop('defaults', {})
    if not isinstance(defaults, Mapping):
      raise ValueError(
          'Manifest defaults must be a JSON object, got '
          f'{type(defaults).__name__}.'
      )
    unknown_defaults = set(defaults) - set(DesignJob.model_fields)
    if unknown_defaults:
      raise ValueError(
          f'Unknown keys in manifest defaults: {sorted(unknown_defaults)}.'
      )
    designs = raw.get('designs')
    if designs is None:
      raise ValueError('Manifest is missing the required "designs" key.')
    if isinstance(designs, Mapping) or not isinstance(designs, Sequence):
      raise ValueError('Manifest "designs" must be a list of design objects.')
    raw['designs'] = _merge_defaults(defaults, designs)
    folding = raw.get('folding', {})
    if isinstance(folding, Mapping):
      raw['folding'] = _default_folding_inputs(folding, raw.get('resequence'))
    return raw

  @pydantic.field_validator('designs')
  @classmethod
  def _check_designs(
      cls, value: tuple[DesignJob, ...]
  ) -> tuple[DesignJob, ...]:
    """Rejects empty manifests and duplicate output prefixes."""
    if not value:
      raise ValueError('Manifest contains no designs.')
    names: set[str] = set()
    prefixes: set[str] = set()
    for job in value:
      if job.name in names:
        raise ValueError(f'Duplicate design name {job.name!r} in manifest.')
      if job.prefix in prefixes:
        raise ValueError(
            f'Designs share an output prefix {job.prefix!r}, which would'
            ' overwrite each other.'
        )
      names.add(job.name)
      prefixes.add(job.prefix)
    return value


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedDesign:
  """A single design to generate: a job paired with a design number and seed.

  Attributes:
    job: The job this design belongs to.
    design_num: Zero-based index of this design within the job.
    seed: Seed driving both linker length sampling and diffusion noise.
  """

  job: DesignJob
  design_num: int
  seed: int

  @property
  def prefix(self) -> str:
    """Filename prefix identifying this design, e.g. `kemp_loose_0037`."""
    return f'{self.job.prefix}_{self.design_num:0{DESIGN_NUM_WIDTH}d}'

  @property
  def metadata_filename(self) -> str:
    """Name of this design's metadata file."""
    return f'{self.prefix}{METADATA_SUFFIX}'

  def is_complete(self, job_dir: epath.PathLike) -> bool:
    """Returns whether this design is complete, based on the metadata file existing."""
    return (epath.Path(job_dir) / self.metadata_filename).exists()


def expand_designs(job: DesignJob) -> list[ResolvedDesign]:
  """Returns one `ResolvedDesign` per design, ordered by design number."""
  return [
      ResolvedDesign(job=job, design_num=i, seed=job.seed_for(i))
      for i in range(job.design_count)
  ]


def _default_folding_inputs(
    folding: Mapping[str, Any], resequence: Any
) -> dict[str, Any]:
  """Returns `folding` with `inputs` set to the default for the resequence mode."""
  # If resequencing is enabled, the default input is the resequenced structure;
  # otherwise it is the generated structure.
  if 'inputs' in folding:
    return dict(folding)
  resequenced = isinstance(resequence, Mapping) and resequence.get('enabled')
  default = (
      RESEQUENCED_FOLDING_INPUT if resequenced else GENERATED_FOLDING_INPUT
  )
  return dict(folding) | {'inputs': (default,)}


def _resolve_dict(d: Any, base_dir: epath.Path, keys: Sequence[str]) -> Any:
  """Resolves relative path strings in `d` for the given `keys`."""
  if not isinstance(d, Mapping):
    return d
  out = dict(d)
  for k in keys:
    v = out.get(k)
    if isinstance(v, str) and not epath.Path(v).is_absolute():
      out[k] = str(base_dir / v)
  return out


def _resolve_paths(
    *,
    document: Mapping[str, Any] | Sequence[Any],
    base_dir: epath.Path,
) -> Mapping[str, Any] | Sequence[Any]:
  """Resolves relative paths in a manifest document against `base_dir`."""
  if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
    return [_resolve_dict(d, base_dir, _DESIGN_PATH_KEYS) for d in document]
  if not isinstance(document, Mapping):
    return document

  resolved = dict(document)
  for block, keys in (
      ('defaults', _DESIGN_PATH_KEYS),
      ('settings', _SETTINGS_PATH_KEYS),
      ('evaluation', _EVALUATION_PATH_KEYS),
  ):
    if block in resolved:
      resolved[block] = _resolve_dict(resolved[block], base_dir, keys)

  designs = resolved.get('designs')
  if isinstance(designs, Sequence) and not isinstance(designs, (str, bytes)):
    resolved['designs'] = [
        _resolve_dict(d, base_dir, _DESIGN_PATH_KEYS) for d in designs
    ]
  return resolved


def _find_with_suffix(
    directory: epath.PathLike, suffix: str, what: str
) -> list[epath.Path]:
  """Returns every file in `directory` whose name ends in `suffix`, sorted."""
  found = sorted(epath.Path(directory).glob(f'*{suffix}'))
  if not found:
    raise FileNotFoundError(
        f'No {what} found in {directory}: expected a file ending in'
        f' {suffix}. Check that this is the output directory of a'
        ' finished run, or pass the input paths explicitly.'
    )
  return found


def metadata_dir(stage_dir: epath.PathLike) -> epath.Path:
  """Returns the run-level metadata directory (`<run_root>/metadata`) for `stage_dir`."""
  return epath.Path(stage_dir).parent / METADATA_DIRNAME


def find_all_design_metadata(directory: epath.PathLike) -> list[epath.Path]:
  """Returns the metadata file of every design in `directory`, sorted."""
  return _find_with_suffix(directory, METADATA_SUFFIX, 'design')


def folded_state_paths(
    out_dir: epath.PathLike,
    sample_name: str,
    state_name: str,
    seed: int = 0,
) -> tuple[epath.Path, epath.Path]:
  """Returns the structure and confidences output paths for one state and seed."""
  out_path = epath.Path(out_dir)
  stem = f'{sample_name}_{state_name}'
  return (
      out_path / f'{stem}_folded_seed{seed}{CIF_SUFFIX}',
      out_path / f'{stem}_confidences_seed{seed}.json',
  )


def find_state_confidences(
    folded_dir: epath.PathLike,
    sample_name: str,
    state_name: str,
    existing_names: Set[str],
) -> list[epath.Path]:
  """Returns confidence JSON paths for `(sample_name, state_name)` sorted by seed index."""
  out_path = epath.Path(folded_dir)
  paths: list[epath.Path] = []
  for seed_idx in itertools.count():
    _, conf_path = folded_state_paths(
        out_path, sample_name, state_name, seed=seed_idx
    )
    if conf_path.name not in existing_names:
      break
    paths.append(conf_path)
  return paths


def af3_input_path(
    out_dir: epath.PathLike,
    sample_name: str,
    state_name: str,
) -> epath.Path:
  """Returns the AlphaFold 3 input JSON folding path for one state."""
  return epath.Path(out_dir) / f'{sample_name}_{state_name}_af3_input.json'


def design_prefix(metadata_path: epath.PathLike) -> str:
  """Returns the filename prefix shared by a design's files."""
  name = epath.Path(metadata_path).name
  if not name.endswith(METADATA_SUFFIX):
    raise ValueError(
        f'Metadata filename {name!r} does not end in {METADATA_SUFFIX}, so the'
        " prefix shared by the design's other files cannot be derived from it."
        f' Rename it to <prefix>{METADATA_SUFFIX}, or pass the input paths'
        ' explicitly.'
    )
  return name.removesuffix(METADATA_SUFFIX)


def list_dir_names(directory: epath.PathLike) -> frozenset[str]:
  """Returns the filenames in `directory` from a single directory scan."""
  dir_path = epath.Path(directory)
  try:
    return frozenset(os.listdir(dir_path))
  except OSError:
    return (
        frozenset(p.name for p in dir_path.iterdir())
        if dir_path.exists()
        else frozenset()
    )


def detect_available_gpus() -> list[str]:
  """Returns visible GPU device indices without initializing CUDA in process."""
  visible = os.environ.get('CUDA_VISIBLE_DEVICES')
  if visible is not None:
    return [
        g.strip() for g in visible.split(',') if g.strip() and g.strip() != '-1'
    ]
  try:
    res = subprocess.run(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if res.returncode == 0:
      return [line.strip() for line in res.stdout.splitlines() if line.strip()]
  except (FileNotFoundError, subprocess.SubprocessError):
    pass
  return []


def spawn_gpu_workers(
    script_path: epath.PathLike,
    base_args: Sequence[str],
    gpus: Sequence[str],
) -> None:
  """Spawns one worker subprocess per GPU in `gpus` and waits for completion."""
  num_workers = len(gpus)
  resolved_script = str(epath.Path(script_path).resolve())
  procs = []
  for w, gpu in enumerate(gpus):
    cmd = [
        sys.executable,
        resolved_script,
        *base_args,
        f'--worker_id={w}',
        f'--num_workers={num_workers}',
    ]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
    procs.append(subprocess.Popen(cmd, env=env))
  exit_codes = [p.wait() for p in procs]
  if any(rc != 0 for rc in exit_codes):
    raise RuntimeError(
        f'One or more GPU workers failed with exit codes: {exit_codes}'
    )
