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

r"""Runs the AlphaProtein Novo design pipeline end to end from one manifest.

The stages are separate programs, run in order:

  1. `run_generator.py`           generate backbones and sequences
  2. `run_ligandmpnn.py`          resequence the scaffold (optional)
  3. `run_alphafold.py`           predict structures
  4. `evaluate_design.py`         score the predictions

The manifest describes the whole campaign, including what the stages after
generation do with the designs, so a run is reproducible from one file. Where
the weights and the LigandMPNN installation live are flags instead, since they
are properties of the machine rather than of the campaign.

Example:
  python run_pipeline.py \
    --manifest=examples/kemp_eliminase/kemp_manifest.json \
    --output_dir=/tmp/kemp_campaign \
    --af3_model_dir=./models/af3_la

Each stage runs as a subprocess. That releases GPU memory between stages, lets
resequencing run under its own interpreter, and keeps every stage runnable on
its own, which is what makes `--only_stage` useful rather than a debugging
aid.
"""

from collections.abc import Sequence
import datetime
import enum
import hashlib
import json
import os
import subprocess
import sys
from typing import Any

from absl import app
from absl import flags
from absl import logging
from alphaprotein_novo.data import design_manifest
from etils import epath

# The stage programs this driver invokes sit beside it.
_PACKAGE_DIR = epath.Path(__file__).parent

# Interpreter the stages run under. The stages are plain scripts, so where this
# driver was launched without a discoverable interpreter, the one on PATH is
# what a user invoking them by hand would get.
_INTERPRETER = sys.executable or 'python3'

# Output subdirectory of each stage. The layout is stage-major because reruns
# are usually "redo folding for everything" rather than "redo everything for
# one design"; design identity lives in the tree below each of these.
_GENERATION_DIRNAME = '01_generation'
_RESEQUENCE_DIRNAME = '02_resequence'
_FOLDING_DIRNAME = '03_folded'
_EVALUATION_DIRNAME = '04_eval'


class Stage(enum.StrEnum):
  """Pipeline stages, in order of execution."""

  GENERATION = enum.auto()
  RESEQUENCE = enum.auto()
  FOLDING = enum.auto()
  EVALUATION = enum.auto()


_STAGES = tuple(Stage)

# Record of what produced a campaign, written at the root of its output.
_INDEX_FILENAME = 'pipeline_index.json'

_MANIFEST = epath.DEFINE_path(
    'manifest',
    None,
    'Path to the design manifest describing the campaign: what to generate,'
    ' and what the stages after generation do with it.',
)
_OUTPUT_DIR = epath.DEFINE_path(
    'output_dir',
    None,
    'Root directory for the campaign, overriding `settings.output_dir` in the'
    ' manifest. Every stage writes its own subdirectory, and per-stage logs go'
    ' under logs/. Required unless `settings.output_dir` is set in the'
    ' manifest.',
)
_APN_MODEL_DIR = epath.DEFINE_path(
    'apn_model_dir',
    None,
    'AlphaProtein Novo generator weights, overriding `settings.model_dir` in'
    ' the manifest.',
)
_AF3_MODEL_DIR = epath.DEFINE_path(
    'af3_model_dir',
    None,
    'AlphaFold 3 weights. Left unset, structure prediction falls back to its'
    ' own default location.',
)
_LIGANDMPNN_DIR = epath.DEFINE_path(
    'ligandmpnn_dir',
    os.environ.get('LIGANDMPNN_DIR'),
    'LigandMPNN checkout. Defaults to $LIGANDMPNN_DIR if set. Required when'
    ' the manifest sets `resequence.enabled`.',
)
_LIGANDMPNN_PYTHON = epath.DEFINE_path(
    'ligandmpnn_python',
    os.environ.get('LIGANDMPNN_PYTHON'),
    'Interpreter of the LigandMPNN environment, which is separate from the one'
    ' running this pipeline. Defaults to $LIGANDMPNN_PYTHON if set. Required'
    ' when the manifest sets `resequence.enabled`.',
)
_RESUME = flags.DEFINE_bool(
    'resume',
    True,
    'Whether to skip work already finished in --output_dir, so that an'
    ' interrupted campaign can be continued. Every stage skips per design.',
)
_FROM_STAGE = flags.DEFINE_enum(
    'from_stage',
    Stage.GENERATION,
    list(Stage),
    'First stage to run. Earlier stages are read from --output_dir instead of'
    ' being run.',
)
_ONLY_STAGE = flags.DEFINE_enum(
    'only_stage',
    None,
    list(Stage),
    'Run just this stage, reading its input from --output_dir. Nothing checks'
    ' that the manifest still describes what is already on disk, so editing'
    ' the manifest and re-running a single stage is allowed, and keeping the'
    ' two consistent is up to you.',
)


def _should_run(*, stage: Stage) -> bool:
  """Returns whether this invocation runs `stage`."""
  if _ONLY_STAGE.value is not None:
    return stage == _ONLY_STAGE.value
  return _STAGES.index(stage) >= _STAGES.index(Stage(_FROM_STAGE.value))


def _stage_argv(script: str, **stage_flags: Any) -> list[str]:
  """Returns the command running `script`, omitting flags that are unset."""
  argv = [_INTERPRETER, str(_PACKAGE_DIR / script)]
  for name, value in stage_flags.items():
    # Repeat a flag if it has multiple values, per absl convention.
    values = value if isinstance(value, list) else [value]
    argv.extend(f'--{name}={item}' for item in values if item is not None)
  return argv


def _run_stage(
    *,
    name: str,
    logs_dir: epath.Path,
    argv: Sequence[str],
) -> None:
  """Runs one stage, echoing its output and keeping a copy in its own log."""
  log_path = logs_dir / f'{name}.log'
  logging.info('Running %s: %s', name, ' '.join(argv))
  with log_path.open('w') as log_file:
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    for line in process.stdout:
      sys.stdout.write(line)
      log_file.write(line)
    return_code = process.wait()
  if return_code:
    raise RuntimeError(
        f'The {name} stage exited with code {return_code}. Its output is in'
        f' {log_path}.'
    )


def _require(path: epath.Path, *, stage: Stage, what: str) -> None:
  """Fails loudly when a stage's input is not where earlier stages leave it."""
  if not path.exists():
    raise app.UsageError(
        f'The {stage} stage reads {what} from {path}, which does not exist.'
        ' Run the earlier stages, or point --output_dir at a campaign that'
        ' has.'
    )


def _resolve_resequence(
    manifest: design_manifest.Manifest,
) -> design_manifest.ResequenceSettings:
  """Returns how this run resequences, refusing to guess when it matters."""
  if not any(
      _should_run(stage=stage) for stage in Stage if stage != Stage.GENERATION
  ):
    return design_manifest.ResequenceSettings(enabled=False)

  if manifest.resequence is None:
    raise app.UsageError(
        'The manifest has no `resequence` block, so whether these designs are'
        ' resequenced with LigandMPNN before folding is undefined. Add'
        ' {"resequence": {"enabled": true}} or {"resequence": {"enabled":'
        ' false}}: folding a generated sequence and folding a resequenced one'
        ' answer different questions, so the pipeline will not pick one.'
    )

  if manifest.resequence.enabled:
    if _LIGANDMPNN_DIR.value is None or not _LIGANDMPNN_DIR.value.exists():
      raise app.UsageError(
          'The manifest sets `resequence.enabled`, but --ligandmpnn_dir'
          f' ({_LIGANDMPNN_DIR.value}) is not a LigandMPNN checkout.'
      )
    if (
        _LIGANDMPNN_PYTHON.value is None
        or not _LIGANDMPNN_PYTHON.value.exists()
    ):
      raise app.UsageError(
          'The manifest sets `resequence.enabled`, but --ligandmpnn_python'
          f' ({_LIGANDMPNN_PYTHON.value}) is not an interpreter.'
      )

  return manifest.resequence


def _write_index(
    *,
    out_dir: epath.Path,
    manifest_path: epath.PathLike,
    manifest: design_manifest.Manifest,
    resequence_enabled: bool,
) -> None:
  """Records the settings that produced this campaign."""
  jobs = {}
  for job in manifest.designs:
    jobs[job.name] = {
        'name': job.name,
        'designs': [
            design.prefix for design in design_manifest.expand_designs(job)
        ],
    }
  manifest_sha256 = hashlib.sha256(
      epath.Path(manifest_path).read_bytes()
  ).hexdigest()
  index_path = out_dir / _INDEX_FILENAME
  previous_stages: set[str] = set()
  if _RESUME.value and index_path.exists():
    try:
      existing = json.loads(index_path.read_text())
      if existing.get('manifest_sha256') == manifest_sha256:
        previous_stages = set(existing.get('stages_run') or ())
    except json.JSONDecodeError:
      pass

  current_stages = {
      str(stage)
      for stage in Stage
      if _should_run(stage=stage)
      and (stage != Stage.RESEQUENCE or resequence_enabled)
  }
  all_stages = previous_stages | current_stages
  index = {
      'manifest_path': str(manifest_path),
      'manifest_sha256': manifest_sha256,
      'stages_run': [stage for stage in Stage if str(stage) in all_stages],
      'finished_at': datetime.datetime.now().isoformat(),
      'jobs': jobs,
  }
  index_path.write_text(json.dumps(index, indent=2) + '\n')
  logging.info('Wrote pipeline index to %s', index_path)


def run_pipeline(
    *,
    manifest_path: epath.PathLike,
    output_dir: epath.PathLike | None = None,
) -> None:
  """Runs the stages this invocation selected over the manifest's designs."""
  manifest = design_manifest.Manifest.from_file(manifest_path)
  resequence = _resolve_resequence(manifest)

  resolved_output_dir = output_dir or manifest.settings.output_dir
  if resolved_output_dir is None:
    raise app.UsageError(
        '--output_dir is required unless `settings.output_dir` is set in the'
        ' manifest.'
    )
  out_dir = epath.Path(resolved_output_dir)
  logs_dir = out_dir / 'logs'
  logs_dir.mkdir(parents=True, exist_ok=True)

  generation_root = out_dir / _GENERATION_DIRNAME
  resequence_root = out_dir / _RESEQUENCE_DIRNAME
  folding_root = out_dir / _FOLDING_DIRNAME
  folds_resequenced = (
      manifest.folding.input_structure()
      == design_manifest.RESEQUENCED_FOLDING_INPUT
  )
  sequence_root = resequence_root if folds_resequenced else generation_root

  if manifest.resequence is not None:
    if folds_resequenced and not manifest.resequence.enabled:
      raise app.UsageError(
          'The manifest specifies folding.inputs=["resequenced"], but'
          ' resequence.enabled is false. Enable resequencing or fold the'
          f' {design_manifest.GENERATED_FOLDING_INPUT} structure instead.'
      )
    if not folds_resequenced and manifest.resequence.enabled:
      logging.warning(
          'The manifest enables resequencing but folds the %s structure, so'
          ' the resequenced structures are built and never folded.',
          design_manifest.GENERATED_FOLDING_INPUT,
      )

  if _should_run(stage=Stage.GENERATION):
    _run_stage(
        name=Stage.GENERATION,
        logs_dir=logs_dir,
        argv=_stage_argv(
            'run_generator.py',
            manifest=manifest_path,
            output_dir=generation_root,
            model_dir=_APN_MODEL_DIR.value,
            resume=_RESUME.value,
        ),
    )

  if resequence.enabled and _should_run(stage=Stage.RESEQUENCE):
    _require(
        generation_root, stage=Stage.RESEQUENCE, what='the generated designs'
    )
    _run_stage(
        name=Stage.RESEQUENCE,
        logs_dir=logs_dir,
        argv=_stage_argv(
            'run_ligandmpnn.py',
            input_dir=generation_root,
            output_dir=resequence_root,
            ligandmpnn_dir=_LIGANDMPNN_DIR.value,
            python_executable=_LIGANDMPNN_PYTHON.value,
            temperature=resequence.temperature,
            num_sequences=resequence.num_sequences,
            ligand_mpnn_use_side_chain_context=int(
                resequence.use_side_chain_context
            ),
            resume=_RESUME.value,
        ),
    )

  if _should_run(stage=Stage.FOLDING):
    _require(sequence_root, stage=Stage.FOLDING, what='the sequences to fold')
    _run_stage(
        name=Stage.FOLDING,
        logs_dir=logs_dir,
        argv=_stage_argv(
            'run_alphafold.py',
            input_dir=sequence_root,
            input_structure=manifest.folding.input_structure(),
            output_dir=folding_root,
            model_dir=_AF3_MODEL_DIR.value,
            seed=list(manifest.folding.seeds),
            resume=_RESUME.value,
        ),
    )

  if _should_run(stage=Stage.EVALUATION):
    _require(folding_root, stage=Stage.EVALUATION, what='the folded structures')
    _run_stage(
        name=Stage.EVALUATION,
        logs_dir=logs_dir,
        argv=_stage_argv(
            'evaluate_design.py',
            input_dir=folding_root,
            output_dir=out_dir / _EVALUATION_DIRNAME,
            eval_reference_cif=manifest.evaluation.reference_cif,
            eval_suite=manifest.evaluation.suite,
            resume=_RESUME.value,
        ),
    )

  _write_index(
      out_dir=out_dir,
      manifest_path=manifest_path,
      manifest=manifest,
      resequence_enabled=resequence.enabled,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many positional arguments.')
  if _MANIFEST.value is None:
    raise app.UsageError('--manifest is required.')

  logging.info('AlphaProtein Novo pipeline starting.')
  run_pipeline(
      manifest_path=_MANIFEST.value,
      output_dir=_OUTPUT_DIR.value,
  )
  logging.info('AlphaProtein Novo pipeline completed successfully.')


if __name__ == '__main__':
  app.run(main)
