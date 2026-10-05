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

"""Sampling pipeline, diffusion trajectory execution, and noise processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, TypeAlias

from alphaprotein_novo.model import config as config_lib
from alphaprotein_novo.model import denoiser as denoiser_lib
from alphaprotein_novo.model import types as types_lib
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import autonotebook

ApplyFn: TypeAlias = Callable[
    [types_lib.DiffusionInput], types_lib.DenoiserOutput
]
PredictFn: TypeAlias = Callable[
    [jax.Array, types_lib.DiffusionInput],
    tuple[jax.Array, types_lib.DenoiserOutput],
]

_MASK_TOKEN_ID: int = 64  # Mask token for discrete sequence diffusion


def expand_dims_to(source: Any, target: jax.Array) -> jax.Array:
  """Expands trailing dimensions of array source to match target's ndim."""
  if isinstance(source, (int, float)) or (
      hasattr(source, 'shape') and not source.shape
  ):
    return source  # pyrefly: ignore[bad-return]
  source = jnp.asarray(source)
  diff = target.ndim - source.ndim
  if diff > 0:
    source = jnp.expand_dims(source, axis=tuple(range(-diff, 0)))
  return source


# ==============================================================================
# Timesteps
# ==============================================================================


def compute_lognormal_timesteps(
    num_timesteps: int,
    tmin: float = 0.01,
    tmax: float = 100.0,
    lognormal_mean: float = 1.7,
    lognormal_std: float = 2.0,
) -> jax.Array:
  """Computes log-normal diffusion timesteps grid from tmax down to tmin."""
  tau_min = 0.5 + 0.5 * jax.scipy.special.erf(
      (jnp.log(tmin) - lognormal_mean) / lognormal_std / jnp.sqrt(2.0)
  )
  tau_max = 0.5 + 0.5 * jax.scipy.special.erf(
      (jnp.log(tmax) - lognormal_mean) / lognormal_std / jnp.sqrt(2.0)
  )
  uniform_grid = jnp.linspace(tau_max, tau_min, num_timesteps)
  return jnp.exp(
      jax.scipy.special.ndtri(uniform_grid) * lognormal_std + lognormal_mean
  )


def get_timesteps(
    num_timesteps: int,
    timesteps_config: config_lib.TimestepsConfig,
) -> jax.Array:
  """Returns the timesteps for diffusion."""
  if timesteps_config.timesteps_name == 'log_normal':
    return compute_lognormal_timesteps(
        num_timesteps=num_timesteps,
        **timesteps_config.timesteps_kwargs,
    )
  raise ValueError(f'Unknown timesteps: {timesteps_config.timesteps_name}')


# ==============================================================================
# Sequence Alpha Schedule
# ==============================================================================


def masked_sequence_alpha(
    t: jax.Array,
    start_t: float = 0.93,
    tau: float = 0.3,
    start: float = -4.0,
    end: float = 3.0,
    lognormal_mean: float = 1.7,
    lognormal_std: float = 2.0,
) -> jax.Array:
  """Converts diffusion timestep t to unmask probability alpha(t)."""
  t_norm = 0.5 + 0.5 * jax.scipy.special.erf(
      (jnp.log(t) - lognormal_mean) / lognormal_std / jnp.sqrt(2.0)
  )
  t_scaled = (t_norm / start_t).clip(0.0, 1.0)
  v_start = jax.nn.sigmoid(start / tau)
  v_end = jax.nn.sigmoid(end / tau)
  x = t_scaled * (end - start) + start
  val_at_x = jax.nn.sigmoid(x / tau)
  return (v_end - val_at_x) / (v_end - v_start)


# ==============================================================================
# Logit Processing
# ==============================================================================


def _process_logits_for_sampling(
    logits: jax.Array,
    temp: float = 1.0,
    top_p: float = 1.0,
    excluded_labels: jax.Array | None = None,
) -> jax.Array:
  """Process logits for sampling with temperature and nucleus filtering."""
  sorted_logits = jax.lax.sort(logits, is_stable=False)
  sorted_probs = jax.nn.softmax(sorted_logits)
  threshold_idx = jnp.argmax(jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
  threshold_largest_logits = jnp.take_along_axis(
      sorted_logits, threshold_idx[..., None], axis=-1
  )
  assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
  mask = logits >= threshold_largest_logits

  logits /= jnp.maximum(temp, 1e-12)

  # Mask to indicate labels to be excluded from sampling.
  anti_mask = 1 - mask
  if excluded_labels is not None:
    anti_mask = anti_mask.at[..., excluded_labels].set(1)
  logits += anti_mask * -1e12

  return logits


# ==============================================================================
# Single-Step Diffusion Solvers
# ==============================================================================


def edm_stochastic_step(
    t_index: int,
    key: jax.Array,
    timesteps: jax.Array,
    xt: jax.Array,
    predict_fn: PredictFn,
    dinput: types_lib.DiffusionInput,
    config: config_lib.StochasticSamplingAlgorithmConfig,
) -> tuple[jax.Array, types_lib.DenoiserOutput]:
  """Performs one stochastic EDM diffusion step on continuous coordinates."""
  noise_key, _, conditioning_key = jax.random.split(key, 3)
  t = timesteps[t_index]
  sigma = t
  gamma = config.s_churn / len(timesteps)
  gamma = (
      jnp.min(jnp.array([gamma, config.gamma_max]))
      * (sigma > config.s_t_min)
      * (sigma < config.s_t_max)
  )
  sigma_new = sigma * (1 + gamma)
  t_new = sigma_new

  state = dataclasses.replace(dinput, t_struct=t_new)
  sigma_diff = jnp.sqrt(sigma_new**2 - sigma**2)
  noise = jax.random.normal(noise_key, xt.shape)
  xt = xt + sigma_diff * noise

  # Fixed crop conditioning on target positions
  assert state.crop_cond_atom_positions is not None
  noised_target_xt = state.crop_cond_atom_positions + t_new * jax.random.normal(
      conditioning_key, state.crop_cond_atom_positions.shape
  )
  xt = jnp.where(state.fixed_atom_mask[..., None], noised_target_xt, xt)

  # Euler step
  x0, denoiser_output = predict_fn(xt, state)
  alpha_t_new = jnp.ones_like(t_new)
  delta = (xt - x0) * alpha_t_new / jnp.maximum(sigma_new, 1e-6)
  t_next = timesteps[t_index + 1]
  dt = t_next - t_new
  xt += config.step_scale * delta * dt

  return xt, denoiser_output


def _get_sequence_alpha_kwargs(
    process_config: config_lib.ProcessConfig,
) -> dict[str, Any]:
  """Extracts kwargs supported by masked_sequence_alpha from ProcessConfig."""
  allowed = {
      'start_t',
      'tau',
      'start',
      'end',
      'lognormal_mean',
      'lognormal_std',
  }
  return {
      k: v for k, v in process_config.process_kwargs.items() if k in allowed
  }


def masked_sequence_step(
    t_index: int,
    key: jax.Array,
    timesteps: jax.Array,
    xt: jax.Array,
    dinput: types_lib.DiffusionInput,
    denoiser_output: types_lib.DenoiserOutput,
    config: config_lib.MaskedSamplingAlgorithmConfig,
    sequence_process: config_lib.ProcessConfig | None = None,
) -> tuple[jax.Array, types_lib.DenoiserOutput]:
  """Performs one masked discrete diffusion reverse step on amino acid tokens."""
  key, _, _ = jax.random.split(key, 3)
  t = timesteps[t_index]
  s = timesteps[t_index + 1]

  alpha_kwargs = (
      _get_sequence_alpha_kwargs(sequence_process)
      if sequence_process is not None
      else {}
  )
  alpha_t = masked_sequence_alpha(t, **alpha_kwargs)
  alpha_s = masked_sequence_alpha(s, **alpha_kwargs)
  alpha_s = jnp.where(t_index + 1 < len(timesteps), alpha_s, 1.0)

  assert denoiser_output.denoised_aatype_logits is not None
  logits = denoiser_output.denoised_aatype_logits[
      ..., : config.num_designable_tokens
  ]

  # How many tokens to unmask.
  log_unmask_prob = jnp.where(
      alpha_t == 0.0,
      -jnp.inf,
      jnp.log(alpha_s - alpha_t) - jnp.log1p(-alpha_t),
  )

  log_mean_preds = jax.nn.log_softmax(logits / config.temperature)
  log_probs_vocab = log_unmask_prob + log_mean_preds
  log_probs_mask = jnp.ones(logits.shape[:-1] + (1,)) * jnp.log1p(
      -jnp.exp(log_unmask_prob)
  )
  log_probs = jnp.concatenate([log_probs_vocab, log_probs_mask], axis=-1)

  processed_logits = _process_logits_for_sampling(log_probs)
  sampled_xs = jax.random.categorical(key, processed_logits, axis=-1)
  sampled_xs = jnp.where(
      sampled_xs == (processed_logits.shape[-1] - 1),
      _MASK_TOKEN_ID,
      sampled_xs,
  )

  assert dinput.fixed_seq_mask is not None
  assert dinput.crop_cond_aatype is not None
  free_mask = dinput.protein.sequence_mask * (1.0 - dinput.fixed_seq_mask)
  is_mask = (xt == _MASK_TOKEN_ID) * free_mask
  unmasked_xs = jnp.where(is_mask, sampled_xs, xt)
  xs = unmasked_xs
  xs = xs * dinput.protein.sequence_mask
  xs = jnp.where(dinput.fixed_seq_mask, dinput.crop_cond_aatype, xs)

  return xs, denoiser_output


# ==============================================================================
# Predict Helpers & Trajectory Stepping
# ==============================================================================


def get_struct_predict_fn(predict_fn: ApplyFn) -> PredictFn:
  """Wraps apply_fn into a coordinate prediction function."""

  def struct_predict_fn(
      xt: jax.Array,
      state: types_lib.DiffusionInput,
  ) -> tuple[jax.Array, types_lib.DenoiserOutput]:
    state = dataclasses.replace(
        state,
        protein=dataclasses.replace(state.protein, atom_positions=xt),
    )
    denoiser_output = predict_fn(state)
    return denoiser_output.denoised_coords, denoiser_output

  return struct_predict_fn


def step(
    dinput: types_lib.DiffusionInput,
    t_index: int,
    struct_timesteps: jax.Array,
    seq_timesteps: jax.Array,
    key: jax.Array,
    predict_fn: ApplyFn,
    config: config_lib.SamplingConfig,
) -> tuple[types_lib.DiffusionInput, types_lib.DenoiserOutput]:
  """Executes one synchronized structure + sequence diffusion step."""
  struct_key, seq_key = jax.random.split(key)

  dinput = dinput.replace(
      t_struct=struct_timesteps[t_index],
      t_seq=seq_timesteps[t_index],
  )

  xt_struct, denoiser_output = edm_stochastic_step(
      t_index=t_index,
      key=struct_key,
      timesteps=struct_timesteps,
      xt=dinput.protein.atom_positions,
      predict_fn=get_struct_predict_fn(predict_fn),
      dinput=dinput,
      config=config.structure_sampling_algorithm,
  )

  xt_seq, _ = masked_sequence_step(
      t_index=t_index,
      key=seq_key,
      timesteps=seq_timesteps,
      xt=dinput.protein.aatype,
      dinput=dinput,
      denoiser_output=denoiser_output,
      config=config.sequence_sampling_algorithm,
      sequence_process=config.sequence_process,
  )

  self_cond_coords = (
      denoiser_output.denoised_coords * dinput.protein.atom_mask[..., None]
  )
  assert dinput.self_cond_aatype is not None
  self_cond_aatype = jnp.zeros_like(dinput.self_cond_aatype)

  dinput_out = dataclasses.replace(
      dinput,
      protein=dataclasses.replace(
          dinput.protein,
          atom_positions=xt_struct,
          aatype=xt_seq,
      ),
      self_cond_atom_positions=self_cond_coords,
      self_cond_aatype=self_cond_aatype,
  )
  return dinput_out, denoiser_output


# ==============================================================================
# Initialization & Preprocessing
# ==============================================================================


def _split_rng_for_consistent_denoising(
    rng: jax.Array,
    num_denoising_steps: int,
) -> jax.Array:
  """Split `rng` key consistently with increasing trajectory length."""
  seeds = jnp.arange(num_denoising_steps)
  keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(rng, seeds)
  return jnp.flip(keys, axis=0)


def preprocess_diffusion_input(
    dinput: types_lib.DiffusionInput,
    rng: jax.Array,
    zero_all_nonfixed: bool,
) -> types_lib.DiffusionInput:
  """Preprocesses training input for the diffusion model."""
  saa_coords, saa_atom_mask = denoiser_lib.regular_to_super_all_atom(
      atom_positions=dinput.protein.atom_positions,
      atom_mask=dinput.protein.atom_mask,
      is_ligand_mask=dinput.protein.is_ligand_mask,
  )

  (
      saa_coords,
      center,
      rot,
      translation,
  ) = denoiser_lib.apply_so3_data_augmentation(
      key=rng,
      dense_atom_positions=saa_coords,
      dense_atom_mask=saa_atom_mask,
      fixed_atom_mask=dinput.fixed_atom_mask,
  )

  if zero_all_nonfixed:
    assert dinput.fixed_atom_mask is not None
    saa_coords = saa_coords * dinput.fixed_atom_mask[..., None]

  assert dinput.fixed_atom_mask is not None
  assert dinput.fixed_seq_mask is not None
  crop_cond_coords = saa_coords * dinput.fixed_atom_mask[..., None]
  crop_cond_aatype = dinput.protein.aatype * dinput.fixed_seq_mask
  crop_cond_aatype = crop_cond_aatype.astype(jnp.int32)

  dinput = denoiser_lib.transform_unindexed_motif_coords(
      dinput,
      center,
      rot,
      translation,
  )

  loss_target_protein = dataclasses.replace(
      dinput.protein,
      atom_positions=saa_coords,
      atom_mask=saa_atom_mask,
  )

  return dataclasses.replace(
      dinput,
      protein=loss_target_protein,
      crop_cond_atom_positions=crop_cond_coords,
      crop_cond_aatype=crop_cond_aatype,
  )


def postprocess_diffusion_output(
    sampled_protein: types_lib.Protein,
    initial_dinput: types_lib.DiffusionInput,
    use_discrete_aatype: bool = True,
    prune_unsupported_atoms: bool = False,
) -> tuple[types_lib.Protein, dict[str, Any]]:
  """Parse super-all-atom sampled protein.

  Args:
    sampled_protein: Protein sampled by the diffusion trajectory.
    initial_dinput: Preprocessed initial diffusion input.
    use_discrete_aatype: Whether to emit the sequence sampled by the discrete
      track rather than the one inferred from the virtual atom clouds.
    prune_unsupported_atoms: Whether to drop side chain atoms required by the
      emitted residue types whose virtual atom stayed on the placeholder atom.

  Returns:
    A tuple of (parsed Protein, aux diagnostics dict).
  """
  sampled_protein = dataclasses.replace(
      sampled_protein,
      atom_positions=jnp.nan_to_num(
          sampled_protein.atom_positions, posinf=1e3, neginf=-1e3
      ),
  )

  sampled_protein = dataclasses.replace(
      initial_dinput.protein,
      atom_positions=sampled_protein.atom_positions,
      aatype=sampled_protein.aatype,
  )

  sampled_protein, aux = denoiser_lib.parse_super_all_atom_prot(
      sampled_protein,
      threshold=1.0,
      placeholder_atom='CA',
      enforce_input_aatype_for_fixed_residues=True,
      fixed_aatype_mask=initial_dinput.fixed_seq_mask,
      diffusion_input_aatype=initial_dinput.protein.aatype,
      use_discrete_aatype=use_discrete_aatype,
      prune_unsupported_atoms=prune_unsupported_atoms,
  )
  return sampled_protein, dataclasses.asdict(aux)


def validate_diffusion_input(
    diffusion_input: types_lib.DiffusionInput,
    config: config_lib.SamplingConfig,
    validate_centering: bool,
) -> None:
  """Validates initial diffusion input."""
  if validate_centering:
    assert diffusion_input.fixed_atom_mask is not None
    nonzero_atom_mask = (
        diffusion_input.protein.atom_mask
        if config.num_partial_diffusion_steps is not None
        else diffusion_input.fixed_atom_mask * diffusion_input.protein.atom_mask
    )
    centered_atom_positions, _ = denoiser_lib.center_positions(
        atom_positions=diffusion_input.protein.atom_positions,
        centering_atom_mask=diffusion_input.fixed_atom_mask
        * diffusion_input.protein.atom_mask,
        nonzero_atom_mask=nonzero_atom_mask,
    )
    if not np.allclose(
        centered_atom_positions,
        diffusion_input.protein.atom_positions,
        atol=1e-3,
    ):
      raise ValueError('Diffusion input is not centered around fixed motif.')


def _init_sampling(
    initial_input: types_lib.DiffusionInput,
    rng: jax.Array,
    config: config_lib.SamplingConfig,
):
  """Initializes timesteps, random keys, and noise states for trajectory execution."""
  assert rng.shape == (2,)
  struct_timesteps = get_timesteps(
      config.num_timesteps, config.structure_timesteps
  )
  seq_timesteps = get_timesteps(config.num_timesteps, config.sequence_timesteps)
  _, traj_key = jax.random.split(rng)
  traj_keys = _split_rng_for_consistent_denoising(
      traj_key, config.num_timesteps
  )
  if config.num_partial_diffusion_steps is not None:
    start_t_idx = config.num_timesteps - config.num_partial_diffusion_steps
  else:
    start_t_idx = 0

  # Sample initial continuous coordinates noise
  start_struct_t = struct_timesteps[start_t_idx]
  alpha_t = expand_dims_to(
      jnp.ones_like(start_struct_t), initial_input.protein.atom_positions
  )
  sigma_t = expand_dims_to(start_struct_t, initial_input.protein.atom_positions)
  xt_struct = (
      alpha_t * initial_input.protein.atom_positions
      + sigma_t
      * jax.random.normal(
          traj_keys[start_t_idx], initial_input.protein.atom_positions.shape
      )
  )

  # Sample initial discrete sequence noise
  if config.num_partial_diffusion_steps is not None:
    alpha_kwargs = _get_sequence_alpha_kwargs(config.sequence_process)
    p_unmask = masked_sequence_alpha(seq_timesteps[start_t_idx], **alpha_kwargs)
    xt_seq = jnp.where(
        jax.random.bernoulli(
            traj_keys[start_t_idx],
            p_unmask,
            initial_input.protein.aatype.shape,
        ),
        initial_input.protein.aatype,
        _MASK_TOKEN_ID,
    )
  else:
    xt_seq = jnp.full(initial_input.protein.aatype.shape, _MASK_TOKEN_ID)

  xt_seq = xt_seq * initial_input.protein.sequence_mask
  if (
      initial_input.fixed_seq_mask is not None
      and initial_input.crop_cond_aatype is not None
  ):
    xt_seq = jnp.where(
        initial_input.fixed_seq_mask,
        initial_input.crop_cond_aatype,
        xt_seq,
    )

  initial_input = dataclasses.replace(
      initial_input,
      protein=dataclasses.replace(
          initial_input.protein,
          atom_positions=xt_struct,
          aatype=xt_seq,
      ),
  )

  return traj_keys, initial_input, struct_timesteps, seq_timesteps, start_t_idx


# ==============================================================================
# Trajectory Solvers
# ==============================================================================


def sample_trajectory(
    initial_input: types_lib.DiffusionInput,
    rng: jax.Array,
    apply_fn: ApplyFn,
    config: config_lib.SamplingConfig,
    middle_t_indices: jax.Array | None = None,
) -> tuple[types_lib.DiffusionOutput, types_lib.DiffusionInput]:
  """Runs the scan-based diffusion trajectory."""
  traj_keys, initial_input, struct_timesteps, seq_timesteps, start_t_idx = (
      _init_sampling(initial_input, rng, config)
  )

  num_middle_inputs = 0 if middle_t_indices is None else len(middle_t_indices)
  middle_dinputs = jax.tree.map(
      lambda x: jnp.zeros((num_middle_inputs,) + x.shape, dtype=x.dtype),
      initial_input,
  )

  def _do_replace(middle_dinputs, dinput, idx):
    return jax.tree.map(
        lambda middle_dinputs, dinput: jax.lax.dynamic_update_index_in_dim(
            operand=middle_dinputs,
            update=dinput,
            index=idx,
            axis=0,
        ),
        middle_dinputs,
        dinput,
    )

  def _do_not_replace(middle_dinputs, dinput, idx):
    del dinput, idx
    return middle_dinputs

  def sample_step(carry, x):
    t_index, key = x
    dinput_curr, middle_dinputs_curr = carry
    dinput_next, _ = step(
        dinput=dinput_curr,
        t_index=t_index,
        struct_timesteps=struct_timesteps,
        seq_timesteps=seq_timesteps,
        key=key,
        predict_fn=apply_fn,
        config=config,
    )

    if middle_t_indices is not None:
      idx = jnp.where(middle_t_indices == t_index, size=1, fill_value=-1)[0][0]
      middle_dinputs_curr = jax.lax.cond(
          idx >= 0,
          _do_replace,
          _do_not_replace,
          middle_dinputs_curr,
          dinput_next,
          idx,
      )
    return (dinput_next, middle_dinputs_curr), None

  carry_init = (initial_input, middle_dinputs)
  xs = [
      jnp.array(list(range(start_t_idx, config.num_timesteps))),
      traj_keys[start_t_idx:],
  ]
  carry_output, _ = jax.lax.scan(
      f=sample_step,
      init=carry_init,
      xs=xs,
  )

  final_dinput, middle_dinputs = carry_output

  denoiser_output = apply_fn(final_dinput)
  denoised_coords = denoiser_output.denoised_coords
  sampling_output = types_lib.DiffusionOutput(
      protein=dataclasses.replace(
          final_dinput.protein,
          atom_positions=denoised_coords,
      ),
      aatype_logits=denoiser_output.denoised_aatype_logits,  # pyrefly: ignore[bad-argument-type]
      predicted_motif_indices_mask_logits=denoiser_output.predicted_motif_indices_mask_logits,
  )

  assert initial_input.fixed_atom_mask is not None
  assert initial_input.crop_cond_atom_positions is not None
  final_coords = jnp.where(
      initial_input.fixed_atom_mask[..., None],
      initial_input.crop_cond_atom_positions,
      sampling_output.protein.atom_positions,
  )
  sampling_output = dataclasses.replace(
      sampling_output,
      protein=dataclasses.replace(
          sampling_output.protein,
          atom_positions=final_coords,
      ),
  )
  return sampling_output, middle_dinputs


def sample_trajectory_no_scan(
    initial_input: types_lib.DiffusionInput,
    rng: jax.Array,
    apply_fn: ApplyFn,
    config: config_lib.SamplingConfig,
    show_progress_bar: bool = True,
    return_aux: bool = False,
) -> tuple[types_lib.DiffusionOutput, dict[str, Any]]:
  """Runs the unrolled Python loop diffusion trajectory (debugging/aux)."""
  traj_keys, dinput, struct_timesteps, seq_timesteps, start_t_idx = (
      _init_sampling(initial_input, rng, config)
  )

  progress_bar = (
      autonotebook.tqdm(total=config.num_timesteps - start_t_idx)
      if show_progress_bar
      else None
  )

  aux = {
      'dinput': [dinput],
      'denoiser_output': [],
      'aatype_discrepancy_count': [],
  }

  for t_idx in range(start_t_idx, config.num_timesteps):
    dinput, denoiser_output = step(
        dinput=dinput,
        t_index=t_idx,
        struct_timesteps=struct_timesteps,
        seq_timesteps=seq_timesteps,
        key=traj_keys[t_idx],
        predict_fn=apply_fn,
        config=config,
    )
    if return_aux:
      aux['dinput'].append(dinput)
      aux['denoiser_output'].append(denoiser_output)
      assert denoiser_output.denoised_aatype_logits is not None
      argmax_aatype = jnp.argmax(
          denoiser_output.denoised_aatype_logits, axis=-1
      )
      aux['aatype_discrepancy_count'].append(
          jnp.sum(argmax_aatype != dinput.protein.aatype)
      )
    if progress_bar:
      progress_bar.update(1)

  denoiser_output = apply_fn(dinput)
  denoised_coords = denoiser_output.denoised_coords
  sampling_output = types_lib.DiffusionOutput(
      protein=dataclasses.replace(
          dinput.protein,
          atom_positions=denoised_coords,
      ),
      aatype_logits=denoiser_output.denoised_aatype_logits,  # pyrefly: ignore[bad-argument-type]
      predicted_motif_indices_mask_logits=denoiser_output.predicted_motif_indices_mask_logits,
  )

  assert initial_input.fixed_atom_mask is not None
  assert initial_input.crop_cond_atom_positions is not None
  final_coords = jnp.where(
      initial_input.fixed_atom_mask[..., None],
      initial_input.crop_cond_atom_positions,
      sampling_output.protein.atom_positions,
  )
  sampling_output = dataclasses.replace(
      sampling_output,
      protein=dataclasses.replace(
          sampling_output.protein,
          atom_positions=final_coords,
      ),
  )
  return sampling_output, aux


# ==============================================================================
# Main Sample API
# ==============================================================================


def sample(
    initial_dinput: types_lib.DiffusionInput,
    rng: jax.Array | int,
    denoiser: denoiser_lib.Denoiser,
    sampling_config: config_lib.SamplingConfig,
    params: hk.Params | None = None,
    return_all_data: bool = False,
) -> tuple[
    types_lib.DiffusionOutput,
    types_lib.DiffusionInput,
    Mapping[str, Any],
]:
  """Samples a protein structure and sequence from the diffusion model.

  Args:
    initial_dinput: Initial input containing target protein and conditioning.
    rng: JAX PRNGKey or integer seed.
    denoiser: The Linen Denoiser module.
    sampling_config: Configuration parameterizing sampling schedules and steps.
    params: Model parameters (weights) dictionary.
    return_all_data: If True, uses unrolled trajectory and returns full aux
      info.

  Returns:
    A tuple of (final DiffusionOutput, preprocessed initial DiffusionInput, aux
    dict).
  """
  if isinstance(rng, int):
    rng = jax.random.PRNGKey(rng)
  preproc_rng, sample_rng = jax.random.split(rng)

  apply_fn = lambda x: denoiser.apply(params, x)  # pyrefly: ignore[bad-argument-type]

  initial_dinput = preprocess_diffusion_input(
      initial_dinput,
      preproc_rng,
      # When doing partial diffusion, we don't want to zero out all
      # non-fixed atoms at the start of sampling, since this would erase
      # the input sample used for partial diffusion.
      zero_all_nonfixed=sampling_config.num_partial_diffusion_steps is None,
  )
  jax.debug.callback(
      validate_diffusion_input,
      initial_dinput,
      sampling_config,
      validate_centering=True,
  )
  sample_kwargs = dict(
      initial_input=initial_dinput,
      rng=sample_rng,
      apply_fn=apply_fn,
      config=sampling_config,
  )

  sample_aux = {}
  if return_all_data:
    diffusion_output, sample_aux = sample_trajectory_no_scan(  # pyrefly: ignore[not-iterable]
        **sample_kwargs, return_aux=return_all_data  # pyrefly: ignore[bad-argument-type]
    )
  else:
    diffusion_output, _ = sample_trajectory(**sample_kwargs)  # pyrefly: ignore[bad-argument-type]

  sampled_protein, postprocess_aux = postprocess_diffusion_output(
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
  if return_all_data:
    aux = aux | sample_aux | postprocess_aux
  return output, initial_dinput, aux
