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

"""AlphaProtein Novo structure and sequence design script.

Demonstrates the end-to-end pipeline:
1. Defines an enzyme design specification.
2. Parses the specification into a ParsedMotifSpec with sampled lengths.
3. Featurizes the parsed specification into a DiffusionInput.
4. Initializes the ProteinDenoiser model and samples from the reverse diffusion
   trajectory.
5. Runs validation assertions on the resulting DiffusionOutput.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import json
from typing import Any

from absl import app
from absl import flags
from absl import logging
from alphafold3 import structure
from alphaprotein_novo.data import design_manifest
from alphaprotein_novo.data import motif_spec
from alphaprotein_novo.data import structure_prediction_spec
from alphaprotein_novo.data import structure_utils
from alphaprotein_novo.data import unindexed_motif
from alphaprotein_novo.model import config
from alphaprotein_novo.model import denoiser as denoiser_lib
from alphaprotein_novo.model import params as params_lib
from alphaprotein_novo.model import sampling
from alphaprotein_novo.model import types as types_lib
from etils import epath
import haiku as hk
import jax
import numpy as np

# Default location for model weights, relative to the repository root.
_DEFAULT_MODEL_DIR = epath.Path(__file__).parent / 'models/apnovo_generator'

# Example manifest, suggested if no manifest is specified.
_EXAMPLE_MANIFEST = (
    epath.Path(__file__).parent / 'examples/kemp_eliminase/kemp_manifest.json'
)

_MANIFEST = epath.DEFINE_path(
    'manifest',
    None,
    'Required. Path to the JSON design manifest describing the batch to'
    ' generate.',
)
_MODEL_DIR = epath.DEFINE_path(
    'model_dir',
    None,
    'Directory containing model weights (.bin.zst). Overrides'
    ' settings.model_dir in the manifest, allowing a manifest to be shared'
    ' between users with different local paths. Defaults to'
    f' {_DEFAULT_MODEL_DIR} if neither is set.',
)
_OUTPUT_DIR = epath.DEFINE_path(
    'output_dir',
    None,
    'Output directory for generated structures. Overrides settings.output_dir'
    ' in the manifest.',
)

# Script behavior flags. Do not affect the designs themselves. Not in manifest.
_RESUME = flags.DEFINE_bool(
    'resume',
    True,
    'Whether to skip designs whose outputs already exist in output_dir,'
    ' allowing an interrupted batch to be continued.',
)
_RETURN_ALL_DATA = flags.DEFINE_bool(
    'return_all_data',
    False,
    'Whether to return full unrolled trajectory diagnostics.',
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


def build_spec(job: design_manifest.DesignJob) -> motif_spec.MotifSpec:
  """Returns the MotifSpec describing the design problem posed by `job`."""
  spec = motif_spec.MotifSpec(**job.motif_spec_kwargs())

  if not spec.input_file or not epath.Path(spec.input_file).exists():
    raise FileNotFoundError(f'Input CIF file not found at: {spec.input_file}')

  if job.partial_diffusion_input_file is not None:
    pd_file = epath.Path(job.partial_diffusion_input_file)
    if not pd_file.exists():
      raise FileNotFoundError(
          f'Partial diffusion input file not found: {pd_file}'
      )
    logging.info('Loading partial diffusion structure from %s...', pd_file)
    pd_struct = structure.from_mmcif(pd_file.read_text()).without_hydrogen()
    pd_struct = pd_struct.copy_and_update_globals(
        bioassembly_data=None
    ).generate_bioassembly()
    spec = dataclasses.replace(spec, partial_diffusion_input_struct=pd_struct)

  return spec


def parse_and_prepare_diffusion_input(
    spec: motif_spec.MotifSpec,
    seed: int,
    global_config: config.GlobalConfig,
) -> types_lib.DiffusionInput:
  """Parses a MotifSpec and converts it into a DiffusionInput.

  Args:
    spec: Unparsed MotifSpec defining the design problem.
    seed: Random seed used to sample linker lengths from interval ranges.
    global_config: Model global configuration specifying vocabulary and bounds.

  Returns:
    A DiffusionInput dataclass ready for input to the Denoiser.
  """
  logging.info('Parsing MotifSpec with seed %d...', seed)
  parsed_spec = spec.parsed(seed=seed)
  logging.info('Converting ParsedMotifSpec to DiffusionInput...')
  max_num_res_unindexed = (
      global_config.max_num_res_unindexed
      if global_config.max_num_res_unindexed is not None
      else 128
  )
  return parsed_spec.to_diffusion_input(
      max_num_res=512,
      max_num_res_unindexed=max_num_res_unindexed,
  )


def resolve_params(model_dir: epath.PathLike) -> hk.Params:
  """Returns the Haiku model parameters loaded from `model_dir`."""
  # Parameter loading dominates startup cost, so a batch run resolves
  # parameters once and passes them to every call of `run_sampling`.
  if not model_dir:
    raise ValueError('model_dir must be specified.')
  if not epath.Path(model_dir).exists():
    raise FileNotFoundError(f'Model directory not found: {model_dir}')
  logging.info('Loading model parameters from %s...', model_dir)
  return params_lib.get_model_haiku_params(model_dir)


@functools.lru_cache(maxsize=8)
def _get_compiled_trajectory(
    sampling_steps: int,
    num_partial_diffusion_steps: int | None,
):
  """Returns a JIT-compiled diffusion trajectory function cached by step count."""
  model_config = config.get_model_config()
  global_config = config.get_global_config()
  sampling_config = model_config.sampling_config.model_copy(
      update={
          'num_timesteps': sampling_steps,
          'num_partial_diffusion_steps': num_partial_diffusion_steps,
      }
  )
  denoiser = denoiser_lib.Denoiser(
      config=model_config,
      global_config=global_config,
  )

  @jax.jit
  def _compiled_trajectory(initial_dinput, sample_rng, params):
    apply_fn = lambda x: denoiser.apply(params, x)  # pyrefly: ignore[bad-argument-type]
    return sampling.sample_trajectory(
        initial_input=initial_dinput,
        rng=sample_rng,
        apply_fn=apply_fn,  # pyrefly: ignore[bad-argument-type]
        config=sampling_config,
    )

  return sampling_config, _compiled_trajectory


def run_sampling(
    dinput: types_lib.DiffusionInput,
    model_dir: epath.PathLike,
    sampling_steps: int,
    seed: int,
    return_all_data: bool,
    num_partial_diffusion_steps: int | None = None,
    params: hk.Params | None = None,
) -> tuple[
    types_lib.DiffusionOutput,
    types_lib.DiffusionInput,
    Mapping[str, Any],
]:
  """Constructs the denoiser and runs the diffusion sampling trajectory.

  Args:
    dinput: Preprocessed DiffusionInput conditioning data.
    model_dir: Directory containing model parameter files. Ignored when `params`
      is supplied.
    sampling_steps: Number of reverse diffusion steps to unroll.
    seed: Seed for diffusion sampling.
    return_all_data: Whether to unroll trajectory without scan for full aux.
    num_partial_diffusion_steps: Optional number of reverse diffusion timesteps
      to unroll in partial diffusion mode.
    params: Optional preloaded parameters, avoiding a reload per design.

  Returns:
    Tuple of (DiffusionOutput, preprocessed DiffusionInput, aux dict).

  Raises:
    ValueError: If model_dir is empty and no params are supplied.
    FileNotFoundError: If model_dir does not exist and no params are supplied.
  """
  rng = jax.random.PRNGKey(seed)
  _, sample_rng = jax.random.split(rng)

  if params is None:
    params = resolve_params(model_dir)

  logging.info('Executing diffusion sampling (%d timesteps)...', sampling_steps)
  if not return_all_data:
    sampling_config, compiled_traj = _get_compiled_trajectory(
        sampling_steps, num_partial_diffusion_steps
    )
    preproc_rng, traj_rng = jax.random.split(sample_rng)
    initial_dinput = sampling.preprocess_diffusion_input(
        dinput,
        preproc_rng,
        zero_all_nonfixed=sampling_config.num_partial_diffusion_steps is None,
    )
    sampling.validate_diffusion_input(
        initial_dinput,
        sampling_config,
        validate_centering=True,
    )
    diffusion_output, _ = compiled_traj(
        initial_dinput, traj_rng, {'params': params}
    )
    sampled_protein, postprocess_aux = sampling.postprocess_diffusion_output(
        diffusion_output.protein,
        initial_dinput,
        use_discrete_aatype=sampling_config.use_discrete_aatype,
        prune_unsupported_atoms=sampling_config.prune_unsupported_atoms,
    )
    output = dataclasses.replace(diffusion_output, protein=sampled_protein)
    aux = {
        'virtual_atom_discrepancy_count': postprocess_aux[
            'virtual_atom_discrepancy_count'
        ],
        'aatype_discrepancy_count': postprocess_aux['aatype_discrepancy_count'],
        'unsupported_atom_count': postprocess_aux['unsupported_atom_count'],
        'discrete_fallback_count': postprocess_aux['discrete_fallback_count'],
        'discrete_aatype': postprocess_aux['discrete_aatype'],
        'saa_aatype': postprocess_aux['saa_aatype'],
    }
    return output, initial_dinput, aux

  model_config = config.get_model_config()
  global_config = config.get_global_config()
  sampling_config = model_config.sampling_config.model_copy(
      update={
          'num_timesteps': sampling_steps,
          'num_partial_diffusion_steps': num_partial_diffusion_steps,
      }
  )

  denoiser = denoiser_lib.Denoiser(
      config=model_config,
      global_config=global_config,
  )

  output, initial_dinput, aux = sampling.sample(
      initial_dinput=dinput,
      rng=sample_rng,
      denoiser=denoiser,
      sampling_config=sampling_config,
      params={'params': params},  # pyrefly: ignore[bad-argument-type, bad-assignment]
      return_all_data=return_all_data,
  )
  return output, initial_dinput, aux


def validate_output(
    output: types_lib.DiffusionOutput,
    dinput: types_lib.DiffusionInput,
    aux: Mapping[str, Any],
) -> None:
  """Runs basic validation assertions on the sampled output.

  Args:
    output: Sampled DiffusionOutput from sampling_pipeline.sample.
    dinput: Preprocessed initial DiffusionInput.
    aux: Auxiliary metrics dictionary returned by sampling.
  """
  logging.info('Validating sampling outputs...')

  # 1. Output structure checks.
  assert isinstance(
      output, types_lib.DiffusionOutput
  ), f'Expected DiffusionOutput, got {type(output)}'
  assert isinstance(
      output.protein, types_lib.Protein
  ), f'Expected Protein, got {type(output.protein)}'

  # 2. Protein coordinate checks.
  atom_pos = np.asarray(output.protein.atom_positions)
  assert (
      atom_pos.ndim == 3
  ), f'Expected 3D positions array, got {atom_pos.shape}'
  assert (
      atom_pos.shape[-1] == 3
  ), f'Expected 3 spatial dimensions, got {atom_pos.shape[-1]}'
  assert np.all(
      np.isfinite(atom_pos)
  ), 'Sampled atom positions contain non-finite values (NaN / Inf)'

  # 3. Amino acid type sequence checks.
  aatype = np.asarray(output.protein.aatype)
  assert aatype.ndim == 1, f'Expected 1D aatype array, got {aatype.shape}'
  assert aatype.shape[0] == atom_pos.shape[0], (
      f'Length mismatch between aatype ({aatype.shape[0]}) and positions'
      f' ({atom_pos.shape[0]})'
  )

  # 4. Logits shape assertions.
  aatype_logits = np.asarray(output.aatype_logits)
  assert (
      aatype_logits.ndim == 2
  ), f'Expected 2D aatype logits, got {aatype_logits.shape}'
  assert (
      aatype_logits.shape[0] == aatype.shape[0]
  ), 'Sequence length mismatch between logits and aatype'

  # 5. Auxiliary diagnostics checks.
  assert (
      'virtual_atom_discrepancy_count' in aux
  ), 'Missing virtual_atom_discrepancy_count in aux dict'
  assert (
      'aatype_discrepancy_count' in aux
  ), 'Missing aatype_discrepancy_count in aux dict'

  logging.info(
      'Validation passed: %d residues sampled, aatype_discrepancy=%s,'
      ' virtual_atom_discrepancy=%s, discrete_fallback=%s,'
      ' unsupported_atoms=%s',
      int(np.sum(np.asarray(dinput.protein.sequence_mask))),
      aux['aatype_discrepancy_count'],
      aux['virtual_atom_discrepancy_count'],
      aux.get('discrete_fallback_count'),
      aux.get('unsupported_atom_count'),
  )


def derive_folding_states(
    struct: structure.Structure,
) -> tuple[structure_prediction_spec.StructurePredictionSpec, ...]:
  """Returns the structures to predict for a design, read off the design itself.

  A design is always worth predicting on its own, and again with whatever
  ligands it was generated around, so that the two can be compared.

  Args:
    struct: The generated structure, ligands included.

  Raises:
    ValueError: If a ligand chain holds more than one residue.
  """
  ligands = []
  ligand_struct = struct.filter(chain_type='non-polymer')
  for chain_id, res_names in ligand_struct.chain_res_name_sequence().items():
    if len(res_names) > 1:
      raise ValueError(
          f'Multiple residues found on ligand chain {chain_id}: {res_names}.'
          ' Only single-residue ligands are supported.'
      )
    if res_names:
      ligands.append(
          structure_prediction_spec.LigandSpec(
              id=chain_id, ccd_code=res_names[0]
          )
      )

  states = [
      structure_prediction_spec.StructurePredictionSpec(
          name='monomer', ligands=()
      )
  ]
  if ligands:
    states.append(
        structure_prediction_spec.StructurePredictionSpec(
            name='complex', ligands=ligands
        )
    )
  else:
    logging.warning(
        'No complex ligands found in designed structure; skipping complex'
        ' structure prediction spec.'
    )
  return tuple(states)


def _remap_folding_states(
    folding_states: Sequence[structure_prediction_spec.StructurePredictionSpec],
    spec: motif_spec.MotifSpec,
    seed: int,
    unindexed_result: (
        unindexed_motif.UnindexedMotifPreparationResult | None
    ) = None,
) -> tuple[structure_prediction_spec.StructurePredictionSpec, ...]:
  """Remaps protein residue and ligand chain IDs in `folding_states` from motif to design numbering."""
  if not any(state.covalent_bonds or state.ligands for state in folding_states):
    return tuple(folding_states)

  parsed_spec = spec if spec.is_parsed else spec.parsed(seed=seed)
  if unindexed_result is not None and unindexed_result.residue_map is not None:
    res_map = unindexed_result.residue_map
  else:
    res_map = parsed_spec.residue_map
  if res_map is None:
    return tuple(folding_states)

  source2target = res_map.get_source_resid_to_target_resid_dict()
  if parsed_spec.input_struct is not None:
    auth2internal, _ = motif_spec.get_author_chain_residue_index_map(
        parsed_spec.input_struct
    )
    for auth_key, internal_key in auth2internal.items():
      if internal_key in source2target:
        source2target.setdefault(auth_key, source2target[internal_key])

  ligand_chain_map = {
      str(c1): str(c2)
      for c1, c2 in zip(
          res_map.source_chain_ids[res_map.is_ligand_mask],
          res_map.target_chain_ids[res_map.is_ligand_mask],
      )
  }

  def _remap_atom(
      atom: tuple[str, int, str], ligand_ids: set[str], state_name: str
  ) -> tuple[str, int, str]:
    chain_id, res_id, atom_name = atom
    if (chain_id, res_id) in source2target:
      new_ch, new_res = source2target[(chain_id, res_id)]
      return (str(new_ch), int(new_res), atom_name)
    if chain_id in ligand_ids:
      return atom
    raise ValueError(
        f'Covalent bond protein residue {chain_id}{res_id} ({atom_name}) in'
        f' folding state {state_name!r} was not found in conditioned motif'
        f' residues {sorted(source2target.keys())}.'
    )

  remapped_states = []
  for state in folding_states:
    if not state.covalent_bonds and not state.ligands:
      remapped_states.append(state)
      continue
    candidate_ids = [ligand_chain_map.get(l.id, l.id) for l in state.ligands]
    if len(set(candidate_ids)) == len(candidate_ids):
      remapped_ligands = tuple(
          l.model_copy(update={'id': new_id}) if new_id != l.id else l
          for l, new_id in zip(state.ligands, candidate_ids)
      )
    else:
      remapped_ligands = state.ligands
    ligand_ids = {l.id for l in remapped_ligands}
    remapped_bonds = tuple(
        structure_prediction_spec.BondSpec(
            atom_1=_remap_atom(bond.atom_1, ligand_ids, state.name),
            atom_2=_remap_atom(bond.atom_2, ligand_ids, state.name),
        )
        for bond in state.covalent_bonds
    )
    remapped_states.append(
        state.model_copy(
            update={
                'ligands': remapped_ligands,
                'covalent_bonds': remapped_bonds,
            }
        )
    )
  return tuple(remapped_states)


def save_outputs(
    output: types_lib.DiffusionOutput,
    dinput: types_lib.DiffusionInput,
    spec: motif_spec.MotifSpec,
    seed: int,
    output_dir: epath.PathLike,
    unindexed_result: (
        unindexed_motif.UnindexedMotifPreparationResult | None
    ) = None,
    output_prefix: str = 'sample',
    design_num: int | None = None,
    num_partial_diffusion_steps: int | None = None,
    num_sampling_steps: int | None = None,
    model_dir: epath.PathLike | None = None,
    folding_states: (
        Sequence[structure_prediction_spec.StructurePredictionSpec] | None
    ) = None,
) -> None:
  """Saves generated structures and LigandMPNN metadata to output directory.

  Args:
    output: Sampled DiffusionOutput.
    dinput: Preprocessed initial DiffusionInput.
    spec: The MotifSpec this design was generated from.
    seed: Seed that produced this design.
    output_dir: Directory to write into.
    unindexed_result: Optional result of unindexed motif preparation.
    output_prefix: Filename prefix identifying this design.
    design_num: Zero-based design index within its job, if part of a batch.
    num_partial_diffusion_steps: Partial diffusion steps used, if any.
    num_sampling_steps: Reverse diffusion timesteps used, recorded because it
      changes the generated structure.
    model_dir: Directory the weights were loaded from, recorded for the same
      reason.
    folding_states: Structures for folding to predict, recorded in the metadata.
      Left unset, they are derived from the design.
  """
  out_path = epath.Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  meta_path = design_manifest.metadata_dir(out_path)
  meta_path.mkdir(parents=True, exist_ok=True)

  if unindexed_result is not None:
    struct = unindexed_result.sampled_struct
    fixed_seq_mask = unindexed_result.fixed_seq_mask
    reference_motif_struct = unindexed_result.reference_motif_struct
  else:
    struct = output.protein.to_structure(b_factors=dinput.fixed_atom_mask)
    fixed_seq_mask = np.asarray(dinput.fixed_seq_mask)
    if (
        dinput.crop_cond_atom_positions is not None
        and dinput.crop_cond_aatype is not None
        and np.any(np.asarray(dinput.fixed_atom_mask))
    ):
      reference_motif_struct = motif_spec.get_motif_gtstruct(dinput)
    else:
      reference_motif_struct = None

  # Save ProDy-compatible PDB file (for LigandMPNN).
  pdb_path = out_path / f'{output_prefix}{design_manifest.PDB_SUFFIX}'
  structure_utils.save_structure_to_pdb(struct, pdb_path)
  logging.info('Saved PDB structure to %s', pdb_path)

  # Save mmCIF file.
  cif_path = out_path / f'{output_prefix}{design_manifest.CIF_SUFFIX}'
  cif_path.write_text(
      structure_utils.add_license_and_terms_of_use_header(struct.to_mmcif())
  )
  logging.info('Saved mmCIF structure to %s', cif_path)

  # Save per-design ground-truth motif mmCIF file (in target residue indexing).
  motif_cif_path = None
  if reference_motif_struct is not None:
    motif_cif_path = (
        out_path / f'{output_prefix}{design_manifest.MOTIF_CIF_SUFFIX}'
    )
    motif_cif_path.write_text(reference_motif_struct.to_mmcif())
    logging.info('Saved ground-truth motif mmCIF to %s', motif_cif_path)

  # Save FASTA sequence.
  protein_struct = struct.filter_out(chain_type='non-polymer')
  chain_seqs = protein_struct.chain_single_letter_sequence()
  if len(chain_seqs) > 1:
    raise ValueError(
        'Multiple protein chains found in designed structure:'
        f' {list(chain_seqs.keys())}. Only single-chain proteins are supported.'
    )
  if chain_seqs:
    protein_seq = next(iter(chain_seqs.values()))
    fasta_path = out_path / f'{output_prefix}{design_manifest.FASTA_SUFFIX}'
    fasta_path.write_text(
        structure_utils.format_fasta_with_terms_of_use(
            output_prefix, protein_seq
        )
    )
    logging.info('Saved FASTA sequence to %s', fasta_path)

  # Extract fixed residue indices and write the metadata. The ligand mask is
  # passed so that ligand tokens, which are always fixed, are not reported as
  # residues of the redesigned protein chain.
  fixed_residues_list, fixed_residues_str = (
      structure_utils.extract_fixed_residues(
          fixed_seq_mask=fixed_seq_mask,
          sequence_mask=np.asarray(dinput.protein.sequence_mask),
          is_ligand_mask=(
              None
              if dinput.protein.is_ligand_mask is None
              else np.asarray(dinput.protein.is_ligand_mask)
          ),
      )
  )
  if folding_states is None:
    folding_states = derive_folding_states(struct)
  else:
    folding_states = _remap_folding_states(
        folding_states,
        spec=spec,
        seed=seed,
        unindexed_result=unindexed_result,
    )

  folding_metadata: dict[str, Any] = {
      'states': {state.name: state.to_dict() for state in folding_states},
  }

  metadata: dict[str, Any] = {
      'spec_name': spec.name,
      'seed': seed,
      'num_residues': int(np.sum(np.asarray(dinput.protein.sequence_mask))),
      'is_partial_diffusion': spec.is_partial_diffusion,
      'fixed_residues': fixed_residues_list,
      'fixed_residues_flag': fixed_residues_str,
      'motif_cif': (
          str(motif_cif_path) if motif_cif_path is not None else spec.input_file
      ),
      'folding': folding_metadata,
  }
  if design_num is not None:
    metadata['design_num'] = design_num
  if num_sampling_steps is not None:
    metadata['num_sampling_steps'] = num_sampling_steps
  if model_dir is not None:
    metadata['model_dir'] = str(model_dir)
  if spec.is_partial_diffusion:
    metadata['num_partial_diffusion_steps'] = num_partial_diffusion_steps
  if unindexed_result is not None:
    metadata['unindexed_motif_metrics'] = unindexed_result.metrics

  metadata_path = (
      meta_path / f'{output_prefix}{design_manifest.METADATA_SUFFIX}'
  )
  metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
  logging.info('Saved sample metadata to %s', metadata_path)


def write_job_index(
    job: design_manifest.DesignJob,
    output_dir: epath.PathLike,
    designs: Sequence[design_manifest.ResolvedDesign],
    completed_design_nums: set[int] | None = None,
) -> None:
  """Writes the index mapping design numbers to seeds and filenames.

  This is the only fixed filename in an output directory, and therefore the
  stable entry point for discovering the descriptively named design files.

  Args:
    job: The job being written.
    output_dir: Directory holding generated designs.
    designs: All designs belonging to this job.
    completed_design_nums: Optional set of design numbers known to be complete
      in memory, avoiding repeated filesystem `exists()` checks.
  """
  out_path = epath.Path(output_dir)
  index_path = out_path / design_manifest.INDEX_FILENAME
  existing_jobs: dict[str, Any] = {}
  if index_path.exists():
    try:
      existing = json.loads(index_path.read_text())
      if isinstance(existing.get('jobs'), dict):
        existing_jobs = dict(existing['jobs'])
      elif isinstance(existing.get('job_name'), str):
        existing_jobs[existing['job_name']] = existing
    except ValueError:
      pass

  def _design_status(design: design_manifest.ResolvedDesign) -> str:
    if completed_design_nums is not None:
      is_done = design.design_num in completed_design_nums
    else:
      is_done = design.is_complete(out_path)
    return 'complete' if is_done else 'pending'

  job_entry = {
      'job_name': job.name,
      'prefix': job.prefix,
      'num_designs': job.design_count,
      'spec_hash': job.spec_hash(),
      'designs': [
          {
              'design_num': design.design_num,
              'seed': design.seed,
              'prefix': design.prefix,
              'metadata_filename': design.metadata_filename,
              'status': _design_status(design),
          }
          for design in designs
      ],
  }
  existing_jobs[job.name] = job_entry
  index = {**job_entry, 'jobs': existing_jobs}
  index_path.write_text(json.dumps(index, indent=2) + '\n')


def check_resume_compatible(
    job: design_manifest.DesignJob, output_dir: epath.PathLike
) -> None:
  """Fails if an existing output directory was produced by a different spec."""
  index_path = epath.Path(output_dir) / design_manifest.INDEX_FILENAME
  if not index_path.exists():
    return
  existing = json.loads(index_path.read_text())
  if isinstance(existing.get('jobs'), dict):
    entry = existing['jobs'].get(job.name)
    if entry is None:
      return
  elif existing.get('job_name') in (None, job.name):
    entry = existing
  else:
    return

  previous_hash = entry.get('spec_hash')
  current_hash = job.spec_hash()
  if previous_hash is not None and previous_hash != current_hash:
    raise ValueError(
        f'Output directory {output_dir} holds designs generated from a'
        f' different spec for job {job.name!r} (hash {previous_hash} on disk,'
        f' but {current_hash} in the manifest). Use a fresh output_dir, or pass'
        ' --noresume to regenerate from scratch.'
    )


def check_resumed_seed(
    design: design_manifest.ResolvedDesign, output_dir: epath.PathLike
) -> None:
  """Fails if a finished design on disk was generated with a different seed."""
  metadata_path = epath.Path(output_dir) / design.metadata_filename
  previous_seed = json.loads(metadata_path.read_text())['seed']
  if previous_seed != design.seed:
    raise ValueError(
        f'Design {design.design_num} of job {design.job.name!r} already exists'
        f' in {output_dir} with seed {previous_seed}, but the manifest now'
        f' gives it seed {design.seed}. Use a fresh output_dir, or pass'
        ' --noresume to regenerate from scratch.'
    )


def resolve_run_settings(
    settings: design_manifest.RunSettings,
) -> tuple[epath.Path, epath.Path | None]:
  """Returns the model directory and the output root, which may be None."""
  # Precedence is flag, then manifest `settings`, then built-in default.
  model_dir = _MODEL_DIR.value or settings.model_dir or _DEFAULT_MODEL_DIR
  output_root = _OUTPUT_DIR.value or settings.output_dir
  return (
      epath.Path(model_dir),
      epath.Path(output_root) if output_root else None,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many positional arguments.')
  if _MANIFEST.value is None:
    raise app.UsageError(
        'No --manifest given. Run the bundled Kemp eliminase'
        f' example with:\n\n  --manifest={_EXAMPLE_MANIFEST}.'
    )

  manifest = design_manifest.Manifest.from_file(_MANIFEST.value)
  model_dir, output_root = resolve_run_settings(manifest.settings)
  metadata_root = (
      design_manifest.metadata_dir(output_root)
      if output_root is not None
      else None
  )
  logging.info(
      'Loaded manifest %s describing %d job(s); model_dir=%s, output_dir=%s,'
      ' metadata_dir=%s (worker %d/%d).',
      _MANIFEST.value,
      len(manifest.designs),
      model_dir,
      output_root,
      metadata_root,
      _WORKER_ID.value,
      _NUM_WORKERS.value,
  )

  if output_root is not None:
    output_root.mkdir(parents=True, exist_ok=True)
  if metadata_root is not None:
    metadata_root.mkdir(parents=True, exist_ok=True)
    if not _RESUME.value and _NUM_WORKERS.value == 1:
      (metadata_root / design_manifest.INDEX_FILENAME).unlink(missing_ok=True)

  if _NUM_WORKERS.value == 1:
    total_pending = 0
    for job in manifest.designs:
      designs = design_manifest.expand_designs(job)
      if not _RESUME.value:
        for design in designs:
          if metadata_root is not None:
            (metadata_root / design.metadata_filename).unlink(missing_ok=True)
          if output_root is not None:
            for suffix in (
                design_manifest.PDB_SUFFIX,
                design_manifest.CIF_SUFFIX,
                design_manifest.FASTA_SUFFIX,
                design_manifest.MOTIF_CIF_SUFFIX,
            ):
              (output_root / f'{design.prefix}{suffix}').unlink(missing_ok=True)
      if metadata_root is not None and _RESUME.value:
        check_resume_compatible(job, metadata_root)
      build_spec(job)
      for design in designs:
        if (
            _RESUME.value
            and metadata_root is not None
            and design.is_complete(metadata_root)
        ):
          check_resumed_seed(design, metadata_root)
        else:
          total_pending += 1

    if (
        output_root is not None
        and metadata_root is not None
        and total_pending > 1
    ):
      gpus = design_manifest.detect_available_gpus()
      if len(gpus) > 1:
        num_workers = min(len(gpus), total_pending)
        logging.info(
            'Sharding %d pending design(s) across %d GPU worker(s): %s',
            total_pending,
            num_workers,
            gpus[:num_workers],
        )
        for job in manifest.designs:
          write_job_index(
              job, metadata_root, design_manifest.expand_designs(job)
          )
        design_manifest.spawn_gpu_workers(
            script_path=__file__,
            base_args=[
                f'--manifest={_MANIFEST.value}',
                f'--model_dir={model_dir}',
                f'--output_dir={output_root}',
                '--resume=True',
                f'--return_all_data={_RETURN_ALL_DATA.value}',
            ],
            gpus=gpus[:num_workers],
        )
        for job in manifest.designs:
          write_job_index(
              job, metadata_root, design_manifest.expand_designs(job)
          )
        logging.info(
            'AlphaProtein Novo multi-GPU generation completed across %d'
            ' worker(s).',
            num_workers,
        )
        return

  global_config = config.get_global_config()
  folding_states = manifest.folding.state_specs()

  params = None
  num_generated = 0
  num_skipped = 0
  design_idx = 0

  for job in manifest.designs:
    designs = design_manifest.expand_designs(job)
    partial_diffusion_steps = job.partial_diffusion_num_steps

    spec = build_spec(job)
    logging.info(
        'Job %s (%s): generating %d design(s).',
        job.name,
        spec.description or 'no description',
        len(designs),
    )

    completed_design_nums: set[int] = set()
    if _RESUME.value and metadata_root is not None:
      for design in designs:
        if design.is_complete(metadata_root):
          completed_design_nums.add(design.design_num)

    for design in designs:
      my_turn = (design_idx % _NUM_WORKERS.value) == _WORKER_ID.value
      design_idx += 1
      if not my_turn:
        continue

      if (
          _RESUME.value
          and metadata_root is not None
          and design.design_num in completed_design_nums
      ):
        if _NUM_WORKERS.value == 1:
          check_resumed_seed(design, metadata_root)
        logging.info(
            'Skipping design %d of job %s; %s already exists.',
            design.design_num,
            job.name,
            design.metadata_filename,
        )
        num_skipped += 1
        continue

      # Re-parsing per design resamples the linker lengths.
      dinput = parse_and_prepare_diffusion_input(
          spec=spec,
          seed=design.seed,
          global_config=global_config,
      )
      logging.info(
          'Design %d (seed %d): prepared DiffusionInput with %d padded'
          ' residues.',
          design.design_num,
          design.seed,
          dinput.protein.aatype.shape[0],
      )

      if params is None:
        params = resolve_params(model_dir)

      output, initial_dinput, aux = run_sampling(
          dinput=dinput,
          model_dir=model_dir,
          sampling_steps=job.num_sampling_steps,
          seed=design.seed,
          return_all_data=_RETURN_ALL_DATA.value,
          num_partial_diffusion_steps=partial_diffusion_steps,
          params=params,
      )

      unindexed_result = None
      if spec.is_unindexed:
        logging.info(
            'Spec is unindexed; executing automatic geometric motif matching...'
        )
        predicted_logits = (
            np.asarray(output.predicted_motif_indices_mask_logits)
            if output.predicted_motif_indices_mask_logits is not None
            else None
        )
        unindexed_result = unindexed_motif.prepare_unindexed_motif_sample(
            sampled_struct=output.protein.to_structure(),
            spec=spec,
            unindexed_motif_data=initial_dinput,
            predicted_motif_indices_mask_logits=predicted_logits,
        )
        logging.info(
            'Unindexed motif matched: insertion_rmsd=%.3f,'
            ' unmatched_residues=%d, matched_residues=%s',
            unindexed_result.metrics.get(
                'unindexed_motif_insertion_rmsd', float('nan')
            ),
            unindexed_result.metrics.get(
                'unindexed_motif_num_unmatched_residues', 0
            ),
            unindexed_result.metrics.get(
                'unindexed_motif_matched_residues', ''
            ),
        )

      validate_output(output=output, dinput=initial_dinput, aux=aux)

      if output_root is not None:
        save_outputs(
            output=output,
            dinput=initial_dinput,
            spec=spec,
            seed=design.seed,
            output_dir=output_root,
            unindexed_result=unindexed_result,
            output_prefix=design.prefix,
            design_num=design.design_num,
            num_partial_diffusion_steps=partial_diffusion_steps,
            num_sampling_steps=job.num_sampling_steps,
            model_dir=model_dir,
            folding_states=folding_states,
        )
        completed_design_nums.add(design.design_num)
        assert metadata_root is not None
        if _NUM_WORKERS.value == 1:
          write_job_index(job, metadata_root, designs, completed_design_nums)
      num_generated += 1

  logging.info(
      'AlphaProtein Novo completed: %d design(s) generated, %d skipped.',
      num_generated,
      num_skipped,
  )


if __name__ == '__main__':
  app.run(main)
