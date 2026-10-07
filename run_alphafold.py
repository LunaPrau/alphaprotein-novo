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

r"""AlphaFold 3 structure prediction runner for AlphaProtein Novo designs.

Executes structure prediction for designed protein sequences across multiple
states (e.g. apo monomer, bound complex, covalent intermediate, etc).

Typical workflow:
  1. Generate designs with AlphaProtein Novo:
     python run_generator.py --output_dir=/tmp/samples

  2. Optional: Resequence with LigandMPNN:
     python run_ligandmpnn.py \
       --input_dir=/tmp/samples \
       --ligandmpnn_dir=/path/to/LigandMPNN \
       --python_executable=/path/to/conda/envs/ligandmpnn/bin/python \
       --output_dir=/tmp/samples

  3. Predict structures with AlphaFold 3:
     python run_alphafold.py \
       --input_dir=/tmp/samples \
       --output_dir=/tmp/folded_out \
       --model_dir=/path/to/af3_la_weights

Predictions run against the AlphaFold 3 Leaving Atom (AF3-LA) weights with
`--fix_standalone_glycans=true`.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence, Set
import dataclasses
import functools
import itertools
import json
import logging as python_logging
import os
from typing import Any
import warnings

from absl import app
from absl import flags
from absl import logging
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.cpp import cif_dict
from alphafold3.data import featurisation
from alphafold3.data.tools import rdkit_utils
from alphafold3.model import features
from alphafold3.model import mmcif_metadata
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model.components import utils
from alphaprotein_novo.data import design_manifest
from alphaprotein_novo.data import structure_prediction_spec
from alphaprotein_novo.data import structure_utils
from etils import epath
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
from rdkit import Chem as rd_chem


class _ExpectedFoldingLogFilter(python_logging.Filter):
  """Hide routine padding and unsupported GLU kernel fallback diagnostics."""

  def filter(self, record: python_logging.LogRecord) -> bool:
    path = record.pathname.replace('\\', '/')
    if (
        '/tokamax/_src/ops/gated_linear_unit/api.py' in path
        and record.getMessage() == 'Failed to run implementation'
        and record.exc_info
        and isinstance(record.exc_info[1], NotImplementedError)
        and str(record.exc_info[1]).startswith('Not supported on ')
    ):
      # Tokamax tries another implementation (including XLA). If none works,
      # its ExceptionGroup still propagates and remains visible to the caller.
      return False
    if (
        record.levelno == python_logging.INFO
        and path.endswith('/alphafold3/data/pipeline.py')
        and record.getMessage().startswith('Got bucket size ')
    ):
      return False
    return True


logging.get_absl_logger().addFilter(_ExpectedFoldingLogFilter())
warnings.filterwarnings(
    'ignore',
    message=r'backend and device argument on jit is deprecated\..*',
    category=DeprecationWarning,
    module=r'run_alphafold|__main__',
)
warnings.filterwarnings(
    'ignore',
    message=r'Explicitly requested dtype int64 requested in broadcasted_iota.*',
    category=UserWarning,
    module=r'alphafold3\.model\.network\.featurization',
)

_DEFAULT_AF3_MODEL_DIR = epath.Path(__file__).parent / 'models/af3_la'

StateByName = Mapping[str, structure_prediction_spec.StructurePredictionSpec]


# Input / Output flags.
_INPUT_DIR = epath.DEFINE_path(
    'input_dir',
    None,
    'Path to a directory holding one or more designs, as written by'
    ' run_generator.py or run_ligandmpnn.py.',
)
_INPUT_STRUCTURE = flags.DEFINE_enum(
    'input_structure',
    design_manifest.GENERATED_FOLDING_INPUT,
    list(design_manifest.FOLDING_INPUTS),
    'Which structure to fold: the generated or resequenced structure.',
)
_OUTPUT_DIR = epath.DEFINE_path(
    'output_dir',
    epath.Path('folded_outputs'),
    'Output directory where folded structures and confidences will be written.',
)
_MODEL_DIR = epath.DEFINE_path(
    'model_dir',
    None,
    'Path to AlphaFold 3 model parameters directory.',
)
_SEED = flags.DEFINE_multi_integer(
    'seed',
    [design_manifest.DEFAULT_FOLDING_SEED],
    'RNG seed(s) for AlphaFold 3. May be specified multiple times (e.g.'
    ' --seed=0 --seed=1). Every input is folded with every seed.',
)
_FIX_STANDALONE_GLYCANS = flags.DEFINE_bool(
    'fix_standalone_glycans',
    True,
    'Whether to keep leaving atoms on glycan ligands that are not bonded to'
    ' anything ("standalone" glycans).',
)

# Script behavior flags. Do not affect the predictions themselves.
_RESUME = flags.DEFINE_bool(
    'resume',
    True,
    'Whether to skip designs already folded into output_dir, allowing an'
    ' interrupted batch to be continued.',
)
_WORKER_ID = flags.DEFINE_integer(
    'worker_id',
    0,
    'Zero-based worker index when sharding designs across multiple GPUs.',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    1,
    'Total number of parallel workers when sharding designs across GPUs.',
)


def load_fasta_sequence(fasta_path: epath.PathLike) -> str:
  """Loads redesigned protein sequence from FASTA file."""
  path = epath.Path(fasta_path)
  if not path.exists():
    raise FileNotFoundError(f'FASTA file not found at: {path}')

  lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
  if not lines:
    raise ValueError(f'Empty FASTA file: {path}')

  sequences = []
  current_seq: list[str] = []
  for line in lines:
    if line.startswith('>'):
      if current_seq:
        sequences.append(''.join(current_seq))
        current_seq = []
    else:
      current_seq.append(line)
  if current_seq:
    sequences.append(''.join(current_seq))

  if len(sequences) != 1:
    raise ValueError(
        f'Expected exactly 1 sequence in FASTA file {path}, but found'
        f' {len(sequences)}'
    )
  return sequences[0]


def _smiles_to_ccd_cif(ccd_code: str, smiles: str, seed: int) -> str:
  """Converts a `(ccd_code, smiles)` pair into a CCD mmCIF block string."""
  mol = rd_chem.MolFromSmiles(smiles)
  if mol is None:
    raise ValueError(f'Invalid SMILES string for {ccd_code!r}: {smiles!r}')
  mol = rd_chem.AddHs(mol)
  mol = rdkit_utils.assign_atom_names_from_graph(mol)
  conformer = rdkit_utils.get_random_conformer(
      mol=mol, random_seed=seed, max_iterations=None, logging_name=ccd_code
  )
  if conformer is not None:
    mol.AddConformer(conformer)
  cif = dict(
      rdkit_utils.mol_to_ccd_cif(mol, component_id=ccd_code, pdbx_smiles=smiles)
  )
  for key in (
      '_chem_comp.name',
      '_chem_comp.type',
      '_chem_comp.mon_nstd_parent_comp_id',
      '_chem_comp.pdbx_synonyms',
      '_chem_comp.formula',
      '_chem_comp.formula_weight',
  ):
    cif[key] = ['non-polymer' if key == '_chem_comp.type' else '?']
  cif['_pdbx_chem_comp_descriptor.comp_id'] = [ccd_code]
  cif['_pdbx_chem_comp_descriptor.type'] = ['SMILES_CANONICAL']
  cif['_pdbx_chem_comp_descriptor.program'] = ['RDKit']
  cif['_pdbx_chem_comp_descriptor.program_version'] = ['?']
  cif['_pdbx_chem_comp_descriptor.descriptor'] = [smiles]
  return cif_dict.CifDict(cif).to_string()


def build_af3_input_for_state(
    state: structure_prediction_spec.StructurePredictionSpec,
    sequence: str,
    seeds: Sequence[int] = (design_manifest.DEFAULT_FOLDING_SEED,),
) -> folding_input.Input:
  """Constructs an AlphaFold 3 Input dataclass for a StructurePredictionSpec."""
  _validate_seeds(seeds)
  rng_seeds = [int(s) for s in seeds]

  protein_chain = folding_input.ProteinChain(
      id='A',
      sequence=sequence,
      ptms=[],
      unpaired_msa='',
      paired_msa='',
      templates=[],
  )
  chains: list[folding_input.ProteinChain | folding_input.Ligand] = [
      protein_chain
  ]
  user_ccd_blocks: dict[str, str] = {}

  for ligand_spec in state.ligands:
    if ligand_spec.smiles is not None:
      # AF3 featurization generates a fresh RDKit conformer on the fly for each
      # rng_seed; the user_ccd conformer only seeds 3D stereochemistry/fallback.
      user_ccd_blocks[ligand_spec.ccd_code] = _smiles_to_ccd_cif(
          ligand_spec.ccd_code, ligand_spec.smiles, seed=rng_seeds[0]
      )
    chains.append(
        folding_input.Ligand(
            id=ligand_spec.id,
            ccd_ids=[ligand_spec.ccd_code],
            description=ligand_spec.name,
        )
    )

  bonds = None
  if state.covalent_bonds:
    bonds = [
        (
            (b.atom_1[0], b.atom_1[1], b.atom_1[2]),
            (b.atom_2[0], b.atom_2[1], b.atom_2[2]),
        )
        for b in state.covalent_bonds
    ]

  return folding_input.Input(
      name=state.name,
      chains=chains,
      rng_seeds=rng_seeds,
      bonded_atom_pairs=bonds,
      user_ccd='\n'.join(user_ccd_blocks.values()) or None,
  )


def make_model_config(
    *,
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> model.Model.Config:
  """Returns an AlphaFold 3 model config with specified settings."""
  config = model.Model.Config()
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  config.return_distogram = return_distogram
  return config


class ModelRunner:
  """Helper class to run AlphaFold 3 structure prediction."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: epath.PathLike,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = epath.Path(model_dir)

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch):
      return model.Model(self._model_config)(batch)

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params
    )

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )
    result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result = dict(result)
    identifier = self.model_params['__meta__']['__identifier__'].tobytes()
    result['__identifier__'] = identifier
    return result

  def extract_inference_results(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> list[model.InferenceResult]:
    """Extracts inference results from model outputs."""
    return list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )


_DEFAULT_BUCKETS = (
    256,
    512,
    768,
)


@functools.lru_cache(maxsize=8)
def _get_ccd(user_ccd: str | None) -> chemical_components.Ccd:
  """Returns a cached Chemical Component Dictionary instance."""
  return chemical_components.Ccd(user_ccd=user_ccd)


def predict_structure_for_state(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = _DEFAULT_BUCKETS,
    fix_standalone_glycans: bool = True,
) -> list[model.InferenceResult]:
  """Runs featurization and model inference for a single state."""
  ccd = _get_ccd(fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      ccd=ccd,
      buckets=buckets,
      fix_standalone_glycans=fix_standalone_glycans,
      verbose=True,
  )
  all_results: list[model.InferenceResult] = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key)
    inference_results = model_runner.extract_inference_results(
        batch=example, result=result, target_name=fold_input.name
    )
    all_results.extend(inference_results)
  return all_results


def _ranking_confidence(result: model.InferenceResult) -> float:
  """Returns the ranking confidence score.

  Ranks AlphaFold 3 predictions by `ranking_confidence`, which equals:
    - 0.8 * ipTM + 0.2 * pTM for multimer complexes (>1 chain)
    - pTM for single-chain monomers
  unlike `ranking_score`, which applies additional disorder and clash penalties.

  Args:
    result: Inference result containing predicted structure and confidence
      metadata.

  Returns:
    Ranking confidence scalar (higher is better).
  """
  if 'ranking_confidence' in result.metadata:
    return float(result.metadata['ranking_confidence'])
  ptm = float(result.metadata.get('predicted_tm_score', 0.0))
  iptm = float(result.metadata.get('interface_predicted_tm_score', 0.0))
  if iptm > 0.0:
    return 0.8 * iptm + 0.2 * ptm
  return ptm


def _structure_filename(prefix: str, input_structure: str) -> str:
  """Returns the design's structure filename that `input_structure` selects."""
  if input_structure == design_manifest.RESEQUENCED_FOLDING_INPUT:
    return f'{prefix}{design_manifest.RESEQ_CIF_SUFFIX}'
  return f'{prefix}{design_manifest.CIF_SUFFIX}'


def _resolve_designs(
    *,
    input_dir: epath.PathLike | None = None,
    input_structure: str = design_manifest.GENERATED_FOLDING_INPUT,
) -> list[tuple[epath.Path, epath.Path, epath.Path | None]]:
  """Returns `(fasta_path, metadata_path, cif_path)` for every design in `input_dir`."""
  if input_dir is None:
    raise ValueError('Input directory must be specified via --input_dir.')
  in_dir = epath.Path(input_dir)
  meta_root = design_manifest.metadata_dir(in_dir)
  metadata_files = design_manifest.find_all_design_metadata(meta_root)
  in_dir_names = design_manifest.list_dir_names(in_dir)

  resolved: list[tuple[epath.Path, epath.Path, epath.Path | None]] = []
  for design_metadata in metadata_files:
    parent_prefix = design_manifest.design_prefix(design_metadata)
    if input_structure == design_manifest.RESEQUENCED_FOLDING_INPUT:
      fastas: list[epath.Path] = []
      for reseq_idx in itertools.count():
        reseq_prefix = design_manifest.resequenced_prefix(
            parent_prefix,
            reseq_index=reseq_idx,
            num_sequences=reseq_idx + 1,
        )
        fasta_name = f'{reseq_prefix}{design_manifest.FASTA_SUFFIX}'
        if fasta_name not in in_dir_names:
          break
        fastas.append(in_dir / fasta_name)
      if not fastas:
        raise FileNotFoundError(
            f'No resequenced FASTA files found for {parent_prefix} in {in_dir}.'
        )
    else:
      fastas = [in_dir / f'{parent_prefix}{design_manifest.FASTA_SUFFIX}']

    for fasta_path in fastas:
      if fasta_path.name not in in_dir_names:
        raise FileNotFoundError(f'FASTA sequence file not found: {fasta_path}')
      cif_name = _structure_filename(fasta_path.stem, input_structure)
      candidate_cif = in_dir / cif_name
      if cif_name in in_dir_names:
        resolved_cif: epath.Path | None = candidate_cif
      else:
        if input_structure == design_manifest.RESEQUENCED_FOLDING_INPUT:
          raise FileNotFoundError(
              f'No resequenced structure at {candidate_cif}.'
          )
        logging.warning(
            'No %s structure at %s, so this run records no reference structure'
            ' and later evaluation has nothing to compare against.',
            design_manifest.GENERATED_FOLDING_INPUT,
            candidate_cif,
        )
        resolved_cif = None
      resolved.append((fasta_path, design_metadata, resolved_cif))
  return resolved


def _select_attention_implementation(device: jax.Device) -> str:
  """Chooses a Tokamax attention backend supported by `device`.

  AlphaFold 3 defaults to its Triton attention kernel, which needs NVIDIA
  compute capability 8.0 (Ampere) or newer. Use chunked XLA attention for
  older CUDA GPUs, and regular XLA attention for CPU and other JAX platforms.
  """
  if device.platform != 'gpu':
    return 'xla'

  compute_capability = getattr(device, 'compute_capability', None)
  try:
    supports_triton = float(compute_capability) >= 8.0
  except (TypeError, ValueError):
    supports_triton = False
  return 'triton' if supports_triton else 'xla_chunked'


def _build_model_runner(model_dir: epath.PathLike | None) -> ModelRunner:
  """Returns an AlphaFold 3 model runner reading weights from `model_dir`."""
  model_path = epath.Path(
      _DEFAULT_AF3_MODEL_DIR if model_dir is None else model_dir
  )
  if not model_path.exists():
    raise FileNotFoundError(f'AF3 Model directory not found: {model_path}')
  logging.info('Initializing AlphaFold 3 ModelRunner from %s...', model_path)
  devices = jax.local_devices()
  device = devices[0] if devices else jax.devices()[0]
  config = make_model_config()
  config.global_config.flash_attention_implementation = (
      _select_attention_implementation(device)
  )
  logging.info(
      'Using AF3 %s attention on %s (platform=%s, compute_capability=%s).',
      config.global_config.flash_attention_implementation,
      device.device_kind,
      device.platform,
      getattr(device, 'compute_capability', 'n/a'),
  )
  return ModelRunner(
      config=config,
      device=device,
      model_dir=model_path,
  )


def _validate_seeds(seeds: Sequence[int]) -> None:
  """Rejects empty or duplicate seed sequences."""
  if not seeds:
    raise ValueError('At least one folding seed is required.')
  if len(set(seeds)) != len(seeds):
    raise ValueError(f'Folding seeds must be distinct, got {list(seeds)}.')


def is_folded(
    out_dir: epath.PathLike,
    sample_name: str,
    *,
    state_names: Iterable[str],
    seeds: Sequence[int] = (design_manifest.DEFAULT_FOLDING_SEED,),
    existing_out_names: Set[str],
) -> bool:
  """Returns whether `out_dir` already holds a finished fold of the design."""
  _validate_seeds(seeds)
  out_dir = epath.Path(out_dir)
  state_list = list(state_names)
  if not state_list:
    return False

  expected_seeds = tuple(int(s) for s in seeds)
  existing_first = design_manifest.find_state_confidences(
      out_dir, sample_name, state_list[0], existing_out_names
  )
  if not existing_first:
    return False

  folded_seeds_list: list[int] = []
  for conf_path in existing_first:
    try:
      conf_data = json.loads(conf_path.read_text())
    except json.JSONDecodeError:
      return False
    seed_val = conf_data.get('seed')
    if seed_val is None:
      return False
    folded_seeds_list.append(int(seed_val))
  folded_seeds = tuple(folded_seeds_list)

  if folded_seeds != expected_seeds[: len(folded_seeds)]:
    raise ValueError(
        f'{sample_name} was already folded with seeds {list(folded_seeds)},'
        f' but this run uses seeds {list(expected_seeds)}. Pass --noresume to'
        ' refold, or --seed to match.'
    )
  if len(folded_seeds) < len(expected_seeds):
    return False

  for state_name in state_list:
    for idx, s in enumerate(expected_seeds):
      cif_path, conf_path = design_manifest.folded_state_paths(
          out_dir, sample_name, state_name, seed=idx
      )
      if (
          cif_path.name not in existing_out_names
          or conf_path.name not in existing_out_names
      ):
        return False
      try:
        conf_data = json.loads(conf_path.read_text())
      except json.JSONDecodeError:
        return False
      if conf_data.get('seed') != s:
        return False
  return True


def _fold_one_design(
    *,
    resolved_fasta: epath.Path,
    resolved_metadata: epath.Path,
    resolved_cif: epath.Path | None,
    sample_name: str,
    state_by_name: StateByName,
    input_structure: str,
    out_dir: epath.Path,
    model_runner: ModelRunner,
    seeds: Sequence[int],
    fix_standalone_glycans: bool = True,
    existing_out_names: Set[str],
) -> None:
  """Folds every state and seed of one design and writes its outputs."""
  sequence = load_fasta_sequence(resolved_fasta)
  logging.info(
      'Loaded redesigned sequence (length %d) from %s',
      len(sequence),
      resolved_fasta,
  )
  logging.info('Loaded metadata from %s', resolved_metadata)
  if resolved_cif:
    logging.info('Resolved reference designed structure: %s', resolved_cif)

  logging.info(
      'Executing structure prediction for %d states (%s) across %d seed(s): %s',
      len(state_by_name),
      list(state_by_name.keys()),
      len(seeds),
      list(seeds),
  )

  for state_name in state_by_name:
    af3_json_path = design_manifest.af3_input_path(
        out_dir, sample_name, state_name
    )
    if af3_json_path.name in existing_out_names:
      af3_json_path.unlink(missing_ok=True)
    for idx in itertools.count():
      folded_cif_path, confidences_path = design_manifest.folded_state_paths(
          out_dir, sample_name, state_name, seed=idx
      )
      state_files = (folded_cif_path, confidences_path)
      present = [p for p in state_files if p.name in existing_out_names]
      if not present and idx >= len(seeds):
        break
      for stale in present:
        stale.unlink(missing_ok=True)

  for state_name, state in state_by_name.items():
    logging.info(
        'Building AF3 input for state "%s" (seeds=%s)...',
        state_name,
        list(seeds),
    )
    af3_input = build_af3_input_for_state(
        state=state,
        sequence=sequence,
        seeds=seeds,
    )
    af3_json_path = design_manifest.af3_input_path(
        out_dir, sample_name, state_name
    )
    af3_json_path.write_text(af3_input.to_json())
    logging.info('Saved AF3 input JSON to %s', af3_json_path)

    for idx, s in enumerate(seeds):
      folded_cif_path, confidences_path = design_manifest.folded_state_paths(
          out_dir, sample_name, state_name, seed=idx
      )

      logging.info(
          'Predicting structure for state "%s" (seed=%d)...', state_name, s
      )
      results = predict_structure_for_state(
          fold_input=dataclasses.replace(af3_input, rng_seeds=[int(s)]),
          model_runner=model_runner,
          fix_standalone_glycans=fix_standalone_glycans,
      )
      if not results:
        raise RuntimeError(
            f'No inference results returned for state "{state_name}"'
        )

      top_sample = max(results, key=_ranking_confidence)
      struct = top_sample.predicted_structure
      # The folded structure carries both AP Novo and AF3 notices.
      folded_cif_path.write_text(
          structure_utils.add_license_and_terms_of_use_header(
              mmcif_metadata.add_legal_comment(struct.to_mmcif())
          )
      )
      plddt = (
          float(np.mean(struct.atoms_table.b_factor))
          if struct.atoms_table.size > 0
          else 0.0
      )
      ptm = float(top_sample.metadata.get('predicted_tm_score', 0.0))
      iptm = float(top_sample.metadata.get('interface_predicted_tm_score', 0.0))
      ranking_confidence = _ranking_confidence(top_sample)
      confidences: dict[str, Any] = {
          'seed': int(s),
          'input_structure': input_structure,
          'designed_cif': (
              os.path.relpath(resolved_cif, out_dir) if resolved_cif else None
          ),
          'plddt': plddt,
          'ptm': ptm,
          'iptm': iptm,
          'ranking_confidence': ranking_confidence,
          'per_atom_plddt': struct.atoms_table.b_factor.tolist(),
      }
      if 'predicted_distance_error' in top_sample.metadata:
        confidences['predicted_distance_error'] = float(
            top_sample.metadata['predicted_distance_error']
        )
      if 'chain_pair_pde_mean' in top_sample.metadata:
        pde_mean = top_sample.metadata['chain_pair_pde_mean']
        confidences['chain_pair_pde_mean'] = (
            pde_mean.tolist() if hasattr(pde_mean, 'tolist') else pde_mean
        )
      confidences_path.write_text(json.dumps(confidences, indent=2) + '\n')
      logging.info(
          'State "%s" (seed=%d) folded: pLDDT=%.2f, pTM=%.2f, ipTM=%.2f,'
          ' ranking_confidence=%.2f',
          state_name,
          s,
          plddt,
          ptm,
          iptm,
          ranking_confidence,
      )


def run_alphafold_for_states(
    *,
    input_dir: epath.PathLike | None = None,
    input_structure: str = design_manifest.GENERATED_FOLDING_INPUT,
    output_dir: epath.PathLike,
    model_dir: epath.PathLike | None = None,
    seeds: Sequence[int] = (design_manifest.DEFAULT_FOLDING_SEED,),
    model_runner: ModelRunner | None = None,
    resume: bool = False,
    fix_standalone_glycans: bool = True,
    worker_id: int = 0,
    num_workers: int = 1,
) -> epath.Path:
  """Executes AlphaFold 3 structure prediction for every resolved design."""
  _validate_seeds(seeds)
  out_dir = epath.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  existing_out_names = design_manifest.list_dir_names(out_dir)

  designs = _resolve_designs(
      input_dir=input_dir,
      input_structure=input_structure,
  )
  logging.info(
      'Folding %d design(s) (worker %d/%d).',
      len(designs),
      worker_id,
      num_workers,
  )

  folded = 0
  skipped = 0
  metadata_cache: dict[epath.Path, dict[str, Any]] = {}
  for design_idx, (
      resolved_fasta,
      resolved_metadata,
      resolved_cif,
  ) in enumerate(designs):
    if (design_idx % num_workers) != worker_id:
      continue

    metadata = metadata_cache.get(resolved_metadata)
    if metadata is None:
      metadata = json.loads(resolved_metadata.read_text())
      metadata_cache[resolved_metadata] = metadata
    sample_name = resolved_fasta.stem
    specs = (metadata.get('folding') or {}).get('states')
    if not specs:
      raise ValueError(
          f'No folding states found in metadata: {resolved_metadata}'
      )
    state_by_name = {
        name: structure_prediction_spec.StructurePredictionSpec.from_dict(spec)
        for name, spec in specs.items()
    }

    if resume and is_folded(
        out_dir,
        sample_name,
        state_names=state_by_name,
        seeds=seeds,
        existing_out_names=existing_out_names,
    ):
      logging.info('Skipping %s, which is already folded.', sample_name)
      skipped += 1
      continue

    if model_runner is None:
      model_runner = _build_model_runner(model_dir)

    _fold_one_design(
        resolved_fasta=resolved_fasta,
        resolved_metadata=resolved_metadata,
        resolved_cif=resolved_cif,
        sample_name=sample_name,
        state_by_name=state_by_name,
        input_structure=input_structure,
        out_dir=out_dir,
        model_runner=model_runner,
        seeds=seeds,
        fix_standalone_glycans=fix_standalone_glycans,
        existing_out_names=existing_out_names,
    )
    folded += 1

  logging.info('Folding completed: %d folded, %d skipped.', folded, skipped)
  return out_dir


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many positional arguments.')

  logging.info(
      'AlphaFold 3 Structure Prediction Runner starting (worker %d/%d).',
      _WORKER_ID.value,
      _NUM_WORKERS.value,
  )
  if _NUM_WORKERS.value == 1 and _INPUT_DIR.value is not None:
    _validate_seeds(_SEED.value)
    out_dir = epath.Path(_OUTPUT_DIR.value)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_out_names = design_manifest.list_dir_names(out_dir)
    designs = _resolve_designs(
        input_dir=_INPUT_DIR.value,
        input_structure=_INPUT_STRUCTURE.value,
    )
    if not _RESUME.value or not existing_out_names:
      pending_count = len(designs)
    else:
      pending_count = 0
      metadata_cache: dict[epath.Path, dict[str, Any]] = {}
      for resolved_fasta, resolved_metadata, _ in designs:
        metadata = metadata_cache.get(resolved_metadata)
        if metadata is None:
          metadata = json.loads(resolved_metadata.read_text())
          metadata_cache[resolved_metadata] = metadata
        sample_name = resolved_fasta.stem
        specs = (metadata.get('folding') or {}).get('states')
        if not specs:
          raise ValueError(
              f'No folding states found in metadata: {resolved_metadata}'
          )
        if is_folded(
            out_dir,
            sample_name,
            state_names=specs,
            seeds=_SEED.value,
            existing_out_names=existing_out_names,
        ):
          continue
        pending_count += 1

    if pending_count > 1:
      gpus = design_manifest.detect_available_gpus()
      if len(gpus) > 1:
        num_workers = min(len(gpus), pending_count)
        logging.info(
            'Sharding %d pending fold design(s) across %d GPU worker(s): %s',
            pending_count,
            num_workers,
            gpus[:num_workers],
        )
        base_args = [
            f'--input_dir={_INPUT_DIR.value}',
            f'--input_structure={_INPUT_STRUCTURE.value}',
            f'--output_dir={out_dir}',
            f'--resume={_RESUME.value}',
            f'--fix_standalone_glycans={_FIX_STANDALONE_GLYCANS.value}',
        ]
        if _MODEL_DIR.value is not None:
          base_args.append(f'--model_dir={_MODEL_DIR.value}')
        for s in _SEED.value:
          base_args.append(f'--seed={s}')
        design_manifest.spawn_gpu_workers(
            script_path=__file__,
            base_args=base_args,
            gpus=gpus[:num_workers],
        )
        logging.info(
            'AlphaFold 3 multi-GPU structure prediction completed across %d'
            ' worker(s).',
            num_workers,
        )
        return

  run_alphafold_for_states(
      input_dir=_INPUT_DIR.value,
      input_structure=_INPUT_STRUCTURE.value,
      output_dir=_OUTPUT_DIR.value,
      model_dir=_MODEL_DIR.value,
      seeds=_SEED.value,
      resume=_RESUME.value,
      fix_standalone_glycans=_FIX_STANDALONE_GLYCANS.value,
      worker_id=_WORKER_ID.value,
      num_workers=_NUM_WORKERS.value,
  )
  logging.info(
      'AlphaFold 3 Structure Prediction Runner completed successfully.'
  )


if __name__ == '__main__':
  app.run(main)
