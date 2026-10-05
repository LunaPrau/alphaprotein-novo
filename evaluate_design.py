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

r"""Design evaluation and metric aggregation for AlphaProtein Novo.

Computes self-consistency metrics, motif ground-truth metrics, and
confidence scores across multi-state structures, outputting structured
JSON and CSV reports.

Typical usage:
  python evaluate_design.py \
      --input_dir=/tmp/folded_out \
      --output_dir=/tmp/eval_out
"""

from collections.abc import Callable, Mapping, Sequence, Set
import concurrent.futures
import csv
import dataclasses
import datetime
import functools
import itertools
import json
import os
from typing import Any, Final

from absl import app
from absl import flags
from absl import logging
from alphafold3 import structure
from alphaprotein_novo.data import design_manifest
from alphaprotein_novo.data import structure_utils
from alphaprotein_novo.metrics import carbene_metrics
from alphaprotein_novo.metrics import confidence_metrics
from alphaprotein_novo.metrics import dssp
from alphaprotein_novo.metrics import kemp_metrics
from alphaprotein_novo.metrics import ligand_metrics
from alphaprotein_novo.metrics import nitrene_metrics
from alphaprotein_novo.metrics import sequence_metrics
from alphaprotein_novo.metrics import serine_esterase_metrics
from alphaprotein_novo.metrics import structure_metrics as metrics_lib
from etils import epath
import numpy as np
from scipy import stats as scipy_stats

_INPUT_DIRS = flags.DEFINE_multi_string(
    'input_dir', None, 'Directory with the output of run_alphafold.py.'
)
_OUTPUT_DIR = epath.DEFINE_path(
    'output_dir', None, 'Directory to save evaluation results.'
)
_EVAL_REFERENCE_CIF = epath.DEFINE_path(
    'eval_reference_cif',
    None,
    'Optional path to ground-truth motif reference CIF structure.',
)
_EVAL_SUITE = flags.DEFINE_string(
    'eval_suite',
    None,
    'Optional evaluation suite name (e.g. kemp_eliminase, serine_esterase,'
    ' dehp_esterase, carbene_transfer, nitrene_transfer).',
)

# Script behavior flags. Do not affect the metrics themselves.
_RESUME = flags.DEFINE_bool(
    'resume',
    True,
    'Whether to reuse the evaluation of a design already in output_dir, as'
    ' long as the design has not been refolded since.',
)
_WORKER_ID = flags.DEFINE_integer(
    'worker_id',
    0,
    'Zero-based worker index when sharding evaluation across CPU workers.',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    1,
    'Total number of parallel workers when sharding evaluation.',
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _DesignMetadata:
  """Parsed contents of a folded design's metadata and confidence files."""

  sample_name: str
  spec_name: str
  metadata_path: epath.Path
  folded_dir: epath.Path
  designed_cif_path: epath.Path
  state_names: Sequence[str]
  seeds: tuple[int, ...] = ()
  fixed_residues: Sequence[str] = ()
  motif_cif: epath.Path | None = None
  confidence_paths: tuple[epath.Path, ...] = ()
  unindexed_num_unmatched_residues: int = 0


def _sanitize_metric_value(val: Any) -> Any:
  """Sanitizes metric values for JSON and CSV serialization."""
  if isinstance(val, (np.floating, float)):
    if np.isnan(val):
      return None
    return float(val)
  if isinstance(val, (np.integer, int)):
    return int(val)
  if isinstance(val, (np.bool_, bool)):
    return bool(val)
  if isinstance(val, np.ndarray):
    return val.tolist()
  return val


def _load_metadata(
    metadata_path: epath.Path,
    *,
    folded_dir: epath.Path,
    sample_name: str,
    existing_folded_names: Set[str],
    metadata_dict: Mapping[str, Any] | None = None,
) -> _DesignMetadata:
  """Loads a folded design's metadata and discovers its folded seeds."""
  data = (
      json.loads(metadata_path.read_text())
      if metadata_dict is None
      else metadata_dict
  )
  folding = data.get('folding') or {}
  state_names = tuple(folding.get('states') or ())

  conf_paths: list[epath.Path] = []
  first_state_paths: Sequence[epath.Path] = ()
  for idx, st_name in enumerate(state_names):
    st_paths = design_manifest.find_state_confidences(
        folded_dir, sample_name, st_name, existing_folded_names
    )
    if idx == 0:
      first_state_paths = st_paths
    conf_paths.extend(st_paths)

  first_confs: list[dict[str, Any]] = [
      json.loads(p.read_text()) for p in first_state_paths
  ]
  seeds = tuple(
      int(c['seed']) for c in first_confs if c.get('seed') is not None
  )
  if not seeds:
    raise ValueError(
        f'{metadata_path} records no folding seed for {sample_name} in'
        f' {folded_dir}, so it describes a design that has not been folded.'
        ' Point --input_dir at run_alphafold.py output.'
    )

  designed_cif = first_confs[0].get('designed_cif')
  if not designed_cif:
    raise ValueError(
        f'{metadata_path} names no designed structure for {sample_name}, so'
        ' there is nothing to evaluate the prediction against. Refold the'
        ' design with its structure present.'
    )

  motif_cif_raw = data.get('motif_cif')
  unindexed_metrics = data.get('unindexed_motif_metrics') or {}
  unindexed_unmatched = int(
      unindexed_metrics.get('unindexed_motif_num_unmatched_residues', 0) or 0
  )
  return _DesignMetadata(
      sample_name=sample_name,
      spec_name=data.get('spec_name', 'unknown_spec'),
      metadata_path=metadata_path,
      folded_dir=folded_dir,
      designed_cif_path=folded_dir / designed_cif,
      state_names=state_names,
      seeds=seeds,
      fixed_residues=data.get('fixed_residues') or (),
      motif_cif=epath.Path(motif_cif_raw) if motif_cif_raw else None,
      confidence_paths=tuple(conf_paths),
      unindexed_num_unmatched_residues=unindexed_unmatched,
  )


def _discover_folded_designs_for_metadata(
    metadata_path: epath.Path,
    folded_dir: epath.Path,
    existing_folded_names: Set[str],
) -> list[_DesignMetadata]:
  """Discovers all folded designs in `folded_dir` matching `metadata_path`."""
  parent_prefix = design_manifest.design_prefix(metadata_path)
  data = json.loads(metadata_path.read_text())
  state_names = tuple((data.get('folding') or {}).get('states') or ())
  if state_names:
    reseq_names: list[str] = []
    for reseq_idx in itertools.count():
      reseq_name = design_manifest.resequenced_prefix(
          parent_prefix,
          reseq_index=reseq_idx,
          num_sequences=reseq_idx + 1,
      )
      _, conf_seed0 = design_manifest.folded_state_paths(
          folded_dir, reseq_name, state_names[0], seed=0
      )
      if conf_seed0.name not in existing_folded_names:
        break
      reseq_names.append(reseq_name)
    if reseq_names:
      return [
          _load_metadata(
              metadata_path,
              folded_dir=folded_dir,
              sample_name=s_name,
              existing_folded_names=existing_folded_names,
              metadata_dict=data,
          )
          for s_name in reseq_names
      ]
  return [
      _load_metadata(
          metadata_path,
          folded_dir=folded_dir,
          sample_name=parent_prefix,
          existing_folded_names=existing_folded_names,
          metadata_dict=data,
      )
  ]


def _load_ground_truth_structure(
    cif_path: epath.Path | None,
) -> structure.Structure | None:
  """Loads ground truth structure from CIF and normalizes atom b-factors."""
  if cif_path is None:
    return None
  gt_struct = structure.from_mmcif(cif_path.read_text())
  if gt_struct.atom_b_factor is not None:
    b_mask = (gt_struct.atom_b_factor == 1.0).astype(np.float32)
    gt_struct = gt_struct.copy_and_update_atoms(atom_b_factor=b_mask)
  return gt_struct


def _load_designed_structure(
    cif_path: epath.Path,
    *,
    fixed_residues: Sequence[str],
) -> structure.Structure:
  """Loads designed structure and populates fixed residue motif b-factors."""
  designed_struct = structure.from_mmcif(cif_path.read_text())
  if (
      designed_struct.atom_b_factor is None
      or not np.any(designed_struct.atom_b_factor == 1)
  ) and fixed_residues:
    designed_struct = structure_utils.populate_fixed_b_factors(
        designed_struct, fixed_residues
    )
  elif designed_struct.atom_b_factor is not None:
    b_mask = (designed_struct.atom_b_factor == 1.0).astype(np.float32)
    designed_struct = designed_struct.copy_and_update_atoms(
        atom_b_factor=b_mask
    )
  return designed_struct


def _load_confidences(
    confidences_path: epath.Path,
) -> dict[str, Any]:
  """Loads and sanitizes AlphaFold confidence metrics from JSON."""
  conf_data = json.loads(confidences_path.read_text())
  confidences_by_metric: dict[str, Any] = {}
  for k, v in conf_data.items():
    if k != 'per_residue_plddt':
      confidences_by_metric[k] = _sanitize_metric_value(v)
  return confidences_by_metric


def _parse_fixed_residue_ids(
    fixed_residues: Sequence[str],
) -> list[int]:
  """Extracts integer residue IDs from fixed residue specifications.

  Unparseable specifications are skipped rather than raising, since callers use
  the result only to narrow suite metrics to candidate motif residues.

  Args:
    fixed_residues: Sequence of fixed residue identifiers.

  Returns:
    The residue IDs, in the order they appear in `fixed_residues`.
  """
  res_ids = []
  for item in fixed_residues:
    try:
      _, res_id = structure_utils.parse_residue_spec(item)
    except ValueError:
      continue
    res_ids.append(res_id)
  return res_ids


def _evaluate_kemp(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    metadata: _DesignMetadata,
    state_name: str = 'complex',
) -> dict[str, Any]:
  """Computes Kemp eliminase suite metrics."""
  del reference_structure
  state_lower = state_name.lower()
  if state_lower == 'monomer':
    return {}
  all_states_lower = {s.lower() for s in metadata.state_names}
  if state_lower != 'complex' and 'complex' in all_states_lower:
    return {}
  fixed_ids = _parse_fixed_residue_ids(metadata.fixed_residues)
  return kemp_metrics.kemp_structure_metrics(
      decoy_structure,
      motif_residues=fixed_ids if fixed_ids else None,
  )


def _evaluate_serine(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    metadata: _DesignMetadata,
    state_name: str = 'complex',
) -> dict[str, Any]:
  """Computes serine esterase suite metrics."""
  del reference_structure
  fixed_ids = _parse_fixed_residue_ids(metadata.fixed_residues)
  return serine_esterase_metrics.serine_esterase_metrics(
      decoy_structure,
      motif_residues=fixed_ids if fixed_ids else None,
      state_name=state_name,
  )


def _evaluate_carbene(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    metadata: _DesignMetadata,
    state_name: str = 'complex',
) -> dict[str, Any]:
  """Computes carbene transfer suite metrics."""
  del metadata
  if state_name.lower() == 'monomer':
    return {}
  return carbene_metrics.carbene_structure_metrics(
      decoy_structure=decoy_structure,
      reference_structure=reference_structure,
  )


def _evaluate_nitrene(
    decoy_structure: structure.Structure,
    reference_structure: structure.Structure,
    metadata: _DesignMetadata,
    state_name: str = 'complex',
) -> dict[str, Any]:
  """Computes nitrene transfer suite metrics."""
  del reference_structure
  catalytic_states = ('complex', 'heme_substrate', 'substrate')
  if not any(s.lower() in catalytic_states for s in metadata.state_names):
    raise ValueError(
        f'Design {metadata.sample_name} states {metadata.state_names} do not'
        ' include a nitrene catalytic state (complex, heme_substrate, or'
        ' substrate).'
    )
  if state_name.lower() not in catalytic_states:
    return {}
  return nitrene_metrics.nitrene_structure_metrics(
      struct=decoy_structure,
  )


_SuiteEvaluatorFn = Callable[
    [structure.Structure, structure.Structure, _DesignMetadata, str],
    dict[str, Any],
]

SUITE_EVALUATORS: Final[Mapping[str, _SuiteEvaluatorFn]] = {
    'kemp_eliminase': _evaluate_kemp,
    'serine_esterase': _evaluate_serine,
    'dehp_esterase': _evaluate_serine,
    'carbene_transfer': _evaluate_carbene,
    'nitrene_transfer': _evaluate_nitrene,
}


def _get_suite_evaluator(
    suite_name: str,
) -> _SuiteEvaluatorFn:
  """Finds suite evaluator matching `suite_name`."""
  key = suite_name.lower()
  if key not in SUITE_EVALUATORS:
    raise ValueError(
        f'Unknown evaluation suite {suite_name!r}. Valid suites:'
        f' {sorted(SUITE_EVALUATORS)}'
    )
  return SUITE_EVALUATORS[key]


def _evaluate_state(
    state_name: str,
    *,
    folded_struct: structure.Structure,
    confidences_json_path: epath.Path,
    designed_struct: structure.Structure,
    gt_struct: structure.Structure | None,
    metadata: _DesignMetadata,
    eval_suite: str | None = None,
) -> dict[str, Any]:
  """Evaluates a single state structure against design and ground truth."""
  logging.info('Evaluating state "%s"...', state_name)

  raw_confidences = _load_confidences(confidences_json_path)
  prefix = 'monomer' if state_name.lower() == 'monomer' else 'complex'

  decoy_has_ligands = (
      folded_struct.filter_to_entity_type(ligand=True).num_atoms > 0
  )
  # Only compute ligand consistency metrics for folded states with the
  # same ligand as the design. An exception is the DEHPase 'es' state, which
  # has the same atoms as the design (ti1) so will have these metrics computed.
  state_matches_design_ligand = ligand_metrics.has_matching_ligands(
      folded_struct, designed_struct
  )
  ref_struct = (
      designed_struct
      if state_matches_design_ligand
      else designed_struct.filter_to_entity_type(protein=True)
  )
  ref_gt_struct = (
      gt_struct
      if (gt_struct is None or state_matches_design_ligand)
      else gt_struct.filter_to_entity_type(protein=True)
  )

  computed_metrics = metrics_lib.structure_metrics(
      decoy_structure=folded_struct,
      reference_structure=ref_struct,
      gt_structure=ref_gt_struct,
      compute_ligand_metrics=decoy_has_ligands,
      compute_motif_metrics=bool(
          ref_gt_struct is not None
          or (
              ref_struct.atom_b_factor is not None
              and np.any(ref_struct.atom_b_factor == 1)
          )
      ),
  )
  metrics_by_name: dict[str, Any] = computed_metrics

  metrics_by_name |= dssp.secondary_structure_metrics(folded_struct)
  metrics_by_name |= confidence_metrics.calculate_confidence_metrics(
      raw_confidences, folded_struct, prefix=prefix
  )

  if eval_suite is not None:
    evaluator = _get_suite_evaluator(eval_suite)
    try:
      metrics_by_name |= evaluator(
          folded_struct, ref_struct, metadata, state_name
      )
    except (structure.MissingAtomError, ValueError) as e:
      if (
          metadata.unindexed_num_unmatched_residues > 0
          or 'No valid catalytic triad combinations found' in str(e)
          or 'No valid catalytic role combinations found' in str(e)
      ):
        logging.warning(
            'Skipping %s suite metrics for %s (%s): %s',
            eval_suite,
            metadata.sample_name,
            state_name,
            e,
        )
      else:
        raise

  return {k: _sanitize_metric_value(v) for k, v in metrics_by_name.items()}


_SEED_STATS = {
    'mean': np.mean,
    'median': np.median,
    'min': np.min,
    'max': np.max,
    'std': np.std,
}
_CIRCULAR_SEED_STATS = {
    'mean': functools.partial(scipy_stats.circmean, high=360.0, low=0.0),
    'std': functools.partial(scipy_stats.circstd, high=360.0, low=0.0),
}


def _aggregate_seed_metrics(
    per_seed_metrics: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
  """Aggregates per-seed state metrics into /seed_*, /mean, /median, /min, /max, /std."""
  all_keys: dict[str, None] = {}
  for s in seeds:
    all_keys.update(dict.fromkeys(per_seed_metrics[s]))
  aggregated: dict[str, Any] = {}
  for key in all_keys:
    vals = [_sanitize_metric_value(per_seed_metrics[s].get(key)) for s in seeds]
    first_valid = next((v for v in vals if v is not None), None)
    aggregated[key] = first_valid
    for idx, s in enumerate(seeds):
      if key in per_seed_metrics[s]:
        s_val = _sanitize_metric_value(per_seed_metrics[s][key])
        aggregated[f'{key}/seed_{idx}'] = s_val
    if not isinstance(first_valid, (int, float, type(None))) or isinstance(
        first_valid, bool
    ):
      continue
    num_vals = [float(v) for v in vals if v is not None]
    for idx, val in enumerate(vals):
      aggregated[f'{key}/seed_{idx}'] = val
    stats = (
        _CIRCULAR_SEED_STATS
        if ('angle' in key or 'dihedral' in key)
        else _SEED_STATS
    )
    for stat, fn in stats.items():
      aggregated[f'{key}/{stat}'] = float(fn(num_vals)) if num_vals else None

  return aggregated


def _write_evaluation_summary(
    *,
    summary: Mapping[str, Any],
    output_dir: epath.PathLike,
) -> None:
  """Writes evaluation summary CSV to the output directory."""
  out_dir = epath.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # One row per design, containing all scalar entries from `design['metrics']`.
  designs: Mapping[str, Mapping[str, Any]] = summary['designs']
  non_scalar_keys: set[str] = set()
  for design in designs.values():
    for k, v in design.get('metrics', {}).items():
      if isinstance(v, (list, tuple, dict, np.ndarray)):
        non_scalar_keys.add(k)
  all_metric_keys: list[str] = []
  seen_keys: set[str] = set()
  for design in designs.values():
    for k in design.get('metrics', {}):
      if k in non_scalar_keys or k in seen_keys:
        continue
      seen_keys.add(k)
      all_metric_keys.append(k)

  csv_path = out_dir / 'evaluation_summary.csv'
  with csv_path.open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['sample_name', 'spec_name'] + all_metric_keys)
    for sample_name, design in designs.items():
      metrics = design.get('metrics', {})
      row = [sample_name, design['spec_name']]
      for k in all_metric_keys:
        val = metrics.get(k)
        if val is None or isinstance(val, (list, tuple, dict, np.ndarray)):
          row.append('')
        elif isinstance(val, float) and not isinstance(val, bool):
          row.append(f'{val:.4f}')
        else:
          row.append(str(val))
      writer.writerow(row)
  logging.info('Wrote evaluation summary CSV to %s', csv_path)


def _evaluate_sampled_structure(
    *,
    designed_struct: structure.Structure,
    metadata: _DesignMetadata,
    eval_suite: str | None = None,
) -> dict[str, Any]:
  """Computes top-level sequence/ and sampled/ metrics on the designed structure."""
  results: dict[str, Any] = {}
  for k, v in sequence_metrics.sequence_metrics_from_structure(
      designed_struct
  ).items():
    results[f'sequence/{k}'] = _sanitize_metric_value(v)

  for k, v in dssp.secondary_structure_metrics(designed_struct).items():
    results[f'sampled/{k}'] = _sanitize_metric_value(v)

  results['sampled/num_missing_motif_residues'] = (
      metadata.unindexed_num_unmatched_residues
  )

  if designed_struct.filter_to_entity_type(ligand=True).num_atoms > 0:
    try:
      results['sampled/percent_ligand_bb_clashes'] = _sanitize_metric_value(
          ligand_metrics.percent_backbone_clashes(designed_struct)
      )
      results['sampled/percent_ligand_bb_clashes_1_5'] = _sanitize_metric_value(
          ligand_metrics.percent_backbone_clashes(
              designed_struct, threshold=1.5
          )
      )
      ca_neighbors, _ = metrics_lib.ligand_atom_neighbors(designed_struct)
      results['sampled/ligand_ca_neighbors_min'] = _sanitize_metric_value(
          np.min(ca_neighbors)
      )
      results['sampled/ligand_ca_neighbors_mean'] = _sanitize_metric_value(
          np.mean(ca_neighbors)
      )
    except (structure.MissingAtomError, ValueError) as e:
      logging.warning('Failed to compute sampled ligand metrics: %s', e)

    if eval_suite is not None and eval_suite.lower() == 'kemp_eliminase':
      fixed_ids = _parse_fixed_residue_ids(metadata.fixed_residues)
      try:
        results['sampled/cat_base_tip_ca_neighbors_mean'] = (
            _sanitize_metric_value(
                kemp_metrics.cat_base_tip_ca_neighbors_mean(
                    designed_struct,
                    motif_residues=fixed_ids if fixed_ids else None,
                )
            )
        )
      except (structure.MissingAtomError, ValueError) as e:
        logging.warning(
            'Failed to compute sampled kemp neighbor metrics: %s', e
        )

  return results


def _evaluate_one_design(
    *,
    metadata: _DesignMetadata,
    eval_reference_cif: epath.PathLike | None = None,
    eval_suite: str | None = None,
) -> dict[str, Any]:
  """Evaluates every folded state of one design."""
  suite_evaluator = (
      _get_suite_evaluator(eval_suite) if eval_suite is not None else None
  )

  ref_cif_path = (
      epath.Path(eval_reference_cif)
      if eval_reference_cif
      else metadata.motif_cif
  )

  gt_struct = _load_ground_truth_structure(ref_cif_path)
  designed_struct = _load_designed_structure(
      metadata.designed_cif_path, fixed_residues=metadata.fixed_residues
  )

  out_dir = metadata.folded_dir
  seeds = metadata.seeds
  metrics_by_state: dict[str, dict[str, Any]] = {}
  folded_structs_by_state: dict[str, dict[int, structure.Structure]] = {}

  for state_name in metadata.state_names:
    per_seed_metrics: dict[int, dict[str, Any]] = {}
    folded_structs_by_state[state_name] = {}
    for idx, seed in enumerate(seeds):
      folded_cif_path, confidences_json_path = (
          design_manifest.folded_state_paths(
              out_dir,
              metadata.sample_name,
              state_name,
              seed=idx,
          )
      )
      for path in (folded_cif_path, confidences_json_path):
        if not path.exists():
          raise FileNotFoundError(
              f'State "{state_name}" of design {metadata.sample_name} is'
              f' missing {path}. Refold the design with --noresume.'
          )
      folded_struct = structure.from_mmcif(folded_cif_path.read_text())
      folded_structs_by_state[state_name][seed] = folded_struct
      per_seed_metrics[seed] = _evaluate_state(
          state_name,
          folded_struct=folded_struct,
          confidences_json_path=confidences_json_path,
          designed_struct=designed_struct,
          gt_struct=gt_struct,
          metadata=metadata,
          eval_suite=eval_suite,
      )

    metrics_by_state[state_name] = _aggregate_seed_metrics(
        per_seed_metrics, seeds
    )

  multi_state_metrics_dict: dict[str, Any] = {}
  cross_state_metrics = (
      serine_esterase_metrics.compute_cross_state_ligand_metrics(
          folded_structs_by_state, designed_struct
      )
  )
  if cross_state_metrics:
    multi_state_metrics_dict.update(
        {k: _sanitize_metric_value(v) for k, v in cross_state_metrics.items()}
    )

  if suite_evaluator is _evaluate_nitrene and folded_structs_by_state:
    first_seed_states = {
        state_name: structs[seeds[0]]
        for state_name, structs in folded_structs_by_state.items()
        if seeds[0] in structs
    }
    first_seed_states['resequenced'] = designed_struct
    nitrene_ms = nitrene_metrics.compute_nitrene_multi_state_metrics(
        first_seed_states,
        reference_structure=designed_struct,
    )
    states_by_seed = {}
    for idx, s in enumerate(seeds):
      seed_states = {
          state_name: structs[s]
          for state_name, structs in folded_structs_by_state.items()
          if s in structs
      }
      seed_states['resequenced'] = designed_struct
      states_by_seed[f'seed_{idx}'] = seed_states
    nitrene_ms.update(
        nitrene_metrics.compute_nitrene_multi_seed_metrics(
            states_by_seed,
            reference_structure=designed_struct,
        )
    )
    multi_state_metrics_dict.update(
        {k: _sanitize_metric_value(v) for k, v in nitrene_ms.items()}
    )

  flat_metrics: dict[str, Any] = _evaluate_sampled_structure(
      designed_struct=designed_struct,
      metadata=metadata,
      eval_suite=eval_suite,
  )
  for state_name, state_metrics in metrics_by_state.items():
    for k, v in state_metrics.items():
      if k in multi_state_metrics_dict:
        continue
      if k.startswith(f'{state_name}/'):
        flat_metrics[k] = v
      else:
        flat_metrics[f'{state_name}/{k}'] = v
  flat_metrics.update(multi_state_metrics_dict)

  return {
      'sample_name': metadata.sample_name,
      'spec_name': metadata.spec_name,
      'designed_cif': str(metadata.designed_cif_path),
      'metadata_path': str(metadata.metadata_path),
      'metrics': flat_metrics,
  }


def _get_reusable_evaluation(
    *,
    metadata: _DesignMetadata,
    output_dir: epath.Path,
    existing_eval_names: Set[str],
) -> dict[str, Any] | None:
  """Returns the design's earlier evaluation, if it is still current.

  Args:
    metadata: The design being evaluated.
    output_dir: Directory earlier evaluations were written to.
    existing_eval_names: Pre-listed filenames in `output_dir`.

  Returns:
    The earlier record, or None if there is none or the design has been
    refolded since it was written.
  """
  record_name = f'{metadata.sample_name}{design_manifest.EVALUATION_SUFFIX}'
  if record_name not in existing_eval_names:
    return None
  record_path = output_dir / record_name
  input_mtimes = [metadata.metadata_path.stat().mtime]
  for conf_path in metadata.confidence_paths:
    if conf_path.exists():
      input_mtimes.append(conf_path.stat().mtime)
  if record_path.stat().mtime < max(input_mtimes):
    logging.info(
        '%s was refolded after it was evaluated, so evaluating it again.',
        metadata.sample_name,
    )
    return None
  try:
    return json.loads(record_path.read_text())
  except json.JSONDecodeError:
    return None


def _write_evaluation(
    *,
    design: Mapping[str, Any],
    output_dir: epath.Path,
) -> None:
  """Writes one design's evaluation, so a later run need not redo it."""
  sample_name = design['sample_name']
  record_path = output_dir / f'{sample_name}{design_manifest.EVALUATION_SUFFIX}'
  partial_path = record_path.with_suffix('.partial')
  partial_path.write_text(json.dumps(design, indent=2) + '\n')
  partial_path.replace(record_path)


def _evaluate_and_write_worker(
    task: tuple[
        _DesignMetadata,
        epath.PathLike | None,
        str | None,
        epath.Path | None,
    ],
) -> tuple[str, dict[str, Any]]:
  """Evaluates one design and writes its JSON record in a worker process."""
  metadata, eval_reference_cif, eval_suite, out_dir = task
  design = _evaluate_one_design(
      metadata=metadata,
      eval_reference_cif=eval_reference_cif,
      eval_suite=eval_suite,
  )
  if out_dir is not None:
    _write_evaluation(design=design, output_dir=out_dir)
  return metadata.sample_name, design


def evaluate_designs(
    *,
    input_dirs: Sequence[epath.PathLike] = (),
    output_dir: epath.PathLike | None = None,
    eval_reference_cif: epath.PathLike | None = None,
    eval_suite: str | None = None,
    resume: bool = False,
    worker_id: int = 0,
    num_workers: int = 1,
) -> dict[str, Any]:
  """Evaluates every design and returns one summary covering all of them."""
  if eval_suite is not None:
    _get_suite_evaluator(eval_suite)

  all_metadata: list[_DesignMetadata] = []
  seen_names: set[str] = set()

  for in_dir_like in input_dirs:
    in_dir = epath.Path(in_dir_like)
    existing_folded_names = design_manifest.list_dir_names(in_dir)
    meta_root = design_manifest.metadata_dir(in_dir)
    meta_paths = design_manifest.find_all_design_metadata(meta_root)
    if len(meta_paths) > 4:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(16, len(meta_paths))
      ) as pool:
        discovered_lists = list(
            pool.map(
                functools.partial(
                    _discover_folded_designs_for_metadata,
                    folded_dir=in_dir,
                    existing_folded_names=existing_folded_names,
                ),
                meta_paths,
            )
        )
    else:
      discovered_lists = [
          _discover_folded_designs_for_metadata(
              mp,
              folded_dir=in_dir,
              existing_folded_names=existing_folded_names,
          )
          for mp in meta_paths
      ]
    for dm_list in discovered_lists:
      for dm in dm_list:
        if dm.sample_name in seen_names:
          raise ValueError(
              f'Two designs are both named {dm.sample_name!r}, the second'
              f' being {dm.metadata_path}.'
          )
        seen_names.add(dm.sample_name)
        all_metadata.append(dm)

  out_dir = None if output_dir is None else epath.Path(output_dir)
  existing_eval_names: frozenset[str] = frozenset()
  if out_dir is not None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_eval_names = design_manifest.list_dir_names(out_dir)

  selected_metadata: list[_DesignMetadata] = [
      metadata
      for design_idx, metadata in enumerate(all_metadata)
      if (design_idx % num_workers) == worker_id
  ]

  reused_by_name: dict[str, Any] = {}
  pending_metadata: list[_DesignMetadata] = []
  for metadata in selected_metadata:
    if resume and out_dir is not None:
      design = _get_reusable_evaluation(
          metadata=metadata,
          output_dir=out_dir,
          existing_eval_names=existing_eval_names,
      )
      if design is not None:
        if num_workers == 1:
          logging.info('Reusing the evaluation of %s.', metadata.sample_name)
          reused_by_name[metadata.sample_name] = design
        continue
    pending_metadata.append(metadata)

  evaluated_by_name: dict[str, Any] = {}
  cpu_workers = min(24, os.cpu_count() or 1, len(pending_metadata))
  use_process_pool = (
      num_workers == 1
      and len(pending_metadata) > 4
      and cpu_workers > 1
      and getattr(_evaluate_one_design, '__module__', '') == __name__
  )
  if use_process_pool:
    logging.info(
        'Evaluating %d pending design(s) across %d CPU worker(s).',
        len(pending_metadata),
        cpu_workers,
    )
    tasks = [
        (m, eval_reference_cif, eval_suite, out_dir) for m in pending_metadata
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cpu_workers
    ) as pool:
      for sample_name, design in pool.map(_evaluate_and_write_worker, tasks):
        evaluated_by_name[sample_name] = design
  else:
    for metadata in pending_metadata:
      design = _evaluate_one_design(
          metadata=metadata,
          eval_reference_cif=eval_reference_cif,
          eval_suite=eval_suite,
      )
      if out_dir is not None:
        _write_evaluation(design=design, output_dir=out_dir)
      evaluated_by_name[metadata.sample_name] = design

  designs: dict[str, Any] = {}
  for metadata in selected_metadata:
    if metadata.sample_name in reused_by_name:
      designs[metadata.sample_name] = reused_by_name[metadata.sample_name]
    elif metadata.sample_name in evaluated_by_name:
      designs[metadata.sample_name] = evaluated_by_name[metadata.sample_name]

  summary: dict[str, Any] = {
      'evaluation_time': datetime.datetime.now().isoformat(),
      'designs': designs,
  }
  if out_dir is not None and num_workers == 1:
    _write_evaluation_summary(summary=summary, output_dir=out_dir)

  return summary


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many positional arguments.')

  if not _INPUT_DIRS.value:
    raise app.UsageError(
        'Folding output not specified. Please specify --input_dir.'
    )

  input_dirs = [epath.Path(d) for d in _INPUT_DIRS.value]
  out_dir = epath.Path(_OUTPUT_DIR.value or input_dirs[0])

  evaluate_designs(
      input_dirs=input_dirs,
      output_dir=out_dir,
      eval_reference_cif=_EVAL_REFERENCE_CIF.value,
      eval_suite=_EVAL_SUITE.value,
      resume=_RESUME.value,
      worker_id=_WORKER_ID.value,
      num_workers=_NUM_WORKERS.value,
  )
  logging.info('Design evaluation completed.')


if __name__ == '__main__':
  app.run(main)
