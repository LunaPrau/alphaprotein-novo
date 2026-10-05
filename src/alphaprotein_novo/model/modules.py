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

"""Neural network primitives, layers, and transformer modules in Flax."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import numbers
from typing import Final, Literal

from alphaprotein_novo.model import config as config_lib
from alphaprotein_novo.model import types as types_lib
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tokamax

# pylint: disable=invalid-name
ArrayT = types_lib.ArrayT
Bool = types_lib.Bool
Float = types_lib.Float
Int = types_lib.Int
L = types_lib.L
typed = types_lib.typed


TRUNCATED_NORMAL_STDDEV_FACTOR: Final[np.ndarray] = np.asarray(
    0.87962566103423978, dtype=np.float32
)


def linear(
    input_dims: int,
    output_dims: int,
    inputs: jax.Array,
    weights: jax.Array,
    bias: jax.Array | None,
    transpose_weights: bool,
    precision: jax.lax.DotAlgorithmPreset | jax.lax.Precision,
) -> jax.Array:
  """Linear forward pass."""
  in_letters = 'abcde'[:input_dims]
  out_letters = 'hijkl'[:output_dims]
  if transpose_weights:
    equation = f'...{in_letters}, {out_letters}{in_letters}->...{out_letters}'
  else:
    equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

  weights = weights.astype(inputs.dtype)
  bias = bias.astype(inputs.dtype) if bias is not None else None
  output = jnp.einsum(equation, inputs, weights, precision=precision)
  if bias is not None:
    # Cast to FP32 to ensure there are no BF16 reductions in the backward pass
    # due to the bias broadcast.
    output = output.astype(jnp.float32) + bias.astype(jnp.float32)
    output = output.astype(inputs.dtype)
  return output


def get_initializer_scale(
    initializer_name: str,
    input_shape: tuple[int, ...],
) -> nn.initializers.Initializer:
  """Get initializer for weights."""
  if initializer_name == 'zeros':
    return nn.initializers.constant(value=0.0, dtype=jnp.float32)
  stddev = get_initializer_stddev(initializer_name, input_shape)
  return nn.initializers.truncated_normal(stddev=stddev, dtype=jnp.float32)


def get_initializer_stddev(
    initializer_name: str, input_shape: tuple[int, ...]
) -> jax.Array:
  """Get standard deviation for initializer."""
  # fan-in scaling
  noise_scale = 1.0
  for channel_dim in input_shape:
    noise_scale /= channel_dim
  if initializer_name == 'relu':
    noise_scale *= 2

  stddev = np.sqrt(noise_scale)
  # Adjust stddev for truncation.
  stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
  return jnp.asarray(stddev, dtype=jnp.float32)


class Linear(nn.Module):
  """Linear module.

  This differs from the standard Flax linear in a few ways:
    * It allows us to apply the linear to more than just the last dimension by
      specifying `num_input_dims`
    * Outputs can be specified via an integer or a tuple (e.g. if we want to
      project then reshape)
    * Initializers are specified via strings
    * It allows us to explicitly specify which dimension of the input will map
      to the tpu sublane/lane dimensions
    * It allows for a device-independent precision specification.

  Attributes:
    num_output: number of output channels. Can be tuple when outputting multiple
      dimensions.
    global_config: Optional global configuration.
    initializer: What initializer to use, should be one of {'linear', 'relu',
      'zeros'}.
    num_input_dims: Number of dimensions from the end to project.
    use_bias: Whether to include trainable bias (True by default).
    bias_init: Value used to initialize bias.
    precision: The precision to use for the matrix multiplication.
    transpose_weights: decides whether weights have shape [input, output] or
      [output, input], True means [output, input], this is helpful to avoid
      padding on the tensors holding the weights.
  """

  num_output: int | Sequence[int]
  global_config: config_lib.GlobalConfig | None = None
  initializer: str = 'linear'
  num_input_dims: int = 1
  use_bias: bool | None = None
  bias_init: float = 0.0
  precision: (
      Literal['default', 'high', 'highest']
      | jax.lax.DotAlgorithmPreset
      | jax.lax.Precision
  ) = 'default'
  transpose_weights: bool = False

  def setup(self):
    if isinstance(self.num_output, numbers.Integral):
      self.output_shape = (self.num_output,)
    else:
      self.output_shape = tuple(self.num_output)  # pyrefly: ignore[bad-argument-type]

    self.num_output_dims = len(self.output_shape)

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Connects Module.

    Args:
      inputs: Tensor of shape [..., num_channel]

    Returns:
      output of shape [..., num_output]
    """
    weights, bias = self._get_params(inputs)
    if isinstance(self.precision, str):
      prec = get_explicit_precision(self.precision)
    else:
      prec = self.precision

    return linear(
        input_dims=self.num_input_dims,
        output_dims=self.num_output_dims,
        inputs=inputs,
        weights=weights,
        bias=bias,
        transpose_weights=self.transpose_weights,
        precision=prec,
    )

  def _get_params(
      self,
      inputs: jax.Array,
  ) -> tuple[jax.Array, jax.Array | None]:
    if self.num_input_dims > 0:
      in_shape = inputs.shape[-self.num_input_dims :]
    else:
      in_shape = ()

    weight_init = get_initializer_scale(self.initializer, in_shape)

    if self.transpose_weights:
      weight_shape = self.output_shape + in_shape
      weights = self.param('weights', weight_init, weight_shape)
    else:
      weight_shape = in_shape + self.output_shape
      weights = self.param('weights', weight_init, weight_shape)

    weights = weights.astype(inputs.dtype)

    use_bias = (
        self.global_config is not None
        if self.use_bias is None
        else self.use_bias
    )
    if use_bias:
      bias = self.param(
          'bias',
          nn.initializers.constant(value=self.bias_init, dtype=jnp.float32),
          self.output_shape,
      )
      bias = bias.astype(inputs.dtype)
    else:
      bias = None
    return weights, bias


def get_explicit_precision(
    precision: Literal['default', 'high', 'highest'],
) -> jax.lax.DotAlgorithmPreset | jax.lax.Precision:
  """Returns the precision enum corresponding to the precision."""
  match precision:
    case 'default':
      return jax.lax.Precision.DEFAULT
    case 'high':
      return jax.lax.Precision.HIGH
    case 'highest':
      return jax.lax.Precision.HIGHEST
    case _:
      raise ValueError(f'Unknown {precision=}.')


def LayerNorm(
    global_config: config_lib.GlobalConfig,
    *,
    eps=1e-5,
    axis=-1,
    name=None,
):
  """Layer normalization module."""
  del global_config

  return _LayerNorm(
      axis=axis,
      create_scale=True,
      create_offset=True,
      eps=eps,
      use_fast_variance=True,
      scale_init=nn.initializers.constant(1.0),
      offset_init=nn.initializers.constant(0.0),
      param_axis=axis,
      name=name,
  )


def adaptive_layernorm(
    global_config: config_lib.GlobalConfig, x, single_cond, name
):
  """Adaptive LayerNorm."""
  # Adopted from Scalable Diffusion Models with Transformers
  # https://arxiv.org/abs/2212.09748
  del global_config
  if single_cond is None:
    x = _LayerNorm(
        name=f'{name}layer_norm',
        axis=-1,
        create_scale=True,
        create_offset=True,
        use_fast_variance=True,
    )(x)
  else:
    x = _LayerNorm(
        name=f'{name}layer_norm',
        axis=-1,
        create_scale=False,
        create_offset=False,
        use_fast_variance=True,
    )(x)
    single_cond = _LayerNorm(
        name=f'{name}single_cond_layer_norm',
        axis=-1,
        create_scale=True,
        create_offset=True,
        use_fast_variance=True,
    )(single_cond)
    single_scale = Linear(
        x.shape[-1],
        initializer='zeros',
        use_bias=True,
        precision=get_explicit_precision('default'),
        name=f'{name}single_cond_scale',
    )(single_cond)
    single_bias = Linear(
        x.shape[-1],
        initializer='zeros',
        precision=get_explicit_precision('default'),
        name=f'{name}single_cond_bias',
    )(single_cond)
    x = jax.nn.sigmoid(single_scale) * x + single_bias
  return x


def dgram_from_positions(positions, *, num_bins=15, min_bin=0.2, max_bin=1.5):
  """Compute distogram from amino acid positions.

  Args:
    positions: (num_res, 3) Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches everything
      larger than `max_bin`.

  Returns:
    Distogram with the specified number of bins.
  """

  def squared_difference(x, y):
    return jnp.square(x - y)

  lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
  lower_breaks = jnp.square(lower_breaks)
  upper_breaks = jnp.concatenate(
      [
          lower_breaks[1:],
          jnp.array([1e8], dtype=jnp.float32),
      ],
      axis=-1,
  )
  dist2 = jnp.sum(
      squared_difference(
          jnp.expand_dims(positions, axis=-2),  # [num_res, 1, 3]
          jnp.expand_dims(positions, axis=-3),  # [1, num_res, 3]
      ),
      axis=-1,
      keepdims=True,
  )  # [num_res, num_res, 1]

  dgram = (dist2 > lower_breaks).astype(jnp.float32) * (
      dist2 < upper_breaks
  ).astype(jnp.float32)
  return dgram


class _LayerNorm(nn.LayerNorm):
  """Flax reimplementation of Haiku LayerNorm module.

  Equivalent to nn.LayerNorm but with an extra 'upcast' option that casts
  (b)float16 inputs to float32 before computing the layer norm, and then casts
  the output back to the input type.

  In the previous Haiku version, learnable parameter shapes were always vectors
  rather than possibly higher-rank tensors. Flax LayerNorm doesn't support
  passing a custom scale and bias, so this is removed here.
  """

  def __init__(
      self,
      axis: int,
      create_scale: bool,
      create_offset: bool,
      eps: float = 1e-5,  # NOTE: Default value in nn.LayerNorm is 1e-6.
      scale_init: nn.initializers.Initializer = nn.initializers.ones,
      offset_init: nn.initializers.Initializer = nn.initializers.zeros,
      use_fast_variance: bool = False,
      name: str | None = None,
      param_axis: int = -1,
  ):
    super().__init__(
        epsilon=eps,
        dtype=None,
        param_dtype=jnp.float32,
        use_bias=create_offset,
        use_scale=create_scale,
        bias_init=offset_init,
        scale_init=scale_init,
        reduction_axes=axis,
        feature_axes=param_axis,
        name=name,
        use_fast_variance=use_fast_variance,
    )

  def __call__(
      self, x: jnp.ndarray, mask: jnp.ndarray | None = None
  ) -> jnp.ndarray:
    dtype = x.dtype
    is_16bit = x.dtype in [jnp.bfloat16, jnp.float16]
    if is_16bit:
      x = x.astype(jnp.float32)

    out = super().__call__(x, mask=mask)

    if is_16bit:
      out = out.astype(dtype)

    return out


def with_namescope(f):
  """Add haiku names to functions, so it's easier to profile them."""
  return functools.wraps(f)(jax.named_call(f, name=f.__qualname__))


class SelfAttention(nn.Module):
  """Multihead self-attention."""

  config: config_lib.AttentionConfig
  global_config: config_lib.GlobalConfig

  @with_namescope
  @typed
  @nn.compact
  def __call__(
      self,
      act: Float[ArrayT, '*B L C'],
      mask: Int[ArrayT, '*B L'],
      single_cond: (
          Float[ArrayT, 'L C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
      pair_logits: Float[ArrayT, '*B H L L'],
  ) -> Float[ArrayT, '*B L C']:
    """Multihead self-attention.

    Args:
      act: Input activations.
      mask: Mask that indicates valid residues.
      single_cond: Per-residue conditioning information.
      pair_logits: Pair logits to add to inner product logits.

    Returns:
      Output activations.
    """
    assert len(mask.shape) == len(act.shape) - 1, f'{mask.shape}, {act.shape}'

    num_channel = act.shape[-1]

    act = adaptive_layernorm(
        self.global_config,
        act,
        single_cond=single_cond,
        name='input_layer_norm',
    )

    # Fetch attention dimensions from the config.
    num_head = self.config.num_head
    key_dim = self.config.attn_dim
    value_dim = self.config.attn_dim

    # Casting here reduces memory bandwidth as the outputs of Linear layers
    # are will be bfloat16.
    if self.global_config.use_bf16:
      act_maybe_bf16 = act.astype(jnp.bfloat16)
      bias_maybe_bf16 = pair_logits.astype(jnp.bfloat16)
    else:
      act_maybe_bf16 = act
      bias_maybe_bf16 = pair_logits

    q = Linear(
        [num_head, key_dim],
        self.global_config,
        use_bias=True,
        name='q_projection',
    )(act_maybe_bf16)
    k = Linear(
        [num_head, key_dim],
        self.global_config,
        use_bias=False,
        name='k_projection',
    )(act_maybe_bf16)
    v = Linear(
        [num_head, value_dim],
        self.global_config,
        use_bias=False,
        name='v_projection',
    )(act_maybe_bf16)

    weighted_avg = tokamax.dot_product_attention(
        query=q,
        key=k,
        value=v,
        bias=bias_maybe_bf16,
        mask=mask[..., None, None, :].astype(jnp.bool_),
        scale=key_dim ** (-0.5),
        implementation='xla',
    ).astype(jnp.float32)

    gate_logits = Linear(
        [num_head, value_dim],
        self.global_config,
        bias_init=1.0,
        initializer='zeros',
        name='gating_query',
    )(act)
    weighted_avg *= jax.nn.sigmoid(gate_logits)

    output = Linear(
        num_channel,
        self.global_config,
        initializer='zeros',
        num_input_dims=2,
        name='output_projection',
    )(weighted_avg)

    return output


class CrossAttention(nn.Module):
  """Multihead cross-attention."""

  config: config_lib.AttentionConfig
  global_config: config_lib.GlobalConfig

  @with_namescope
  @typed
  @nn.compact
  def __call__(
      self,
      queries: Float[ArrayT, '*B L_q C'],
      keys: Float[ArrayT, '*B L_k D'],
      queries_mask: Int[ArrayT, '*B L_q'],
      keys_mask: Int[ArrayT, '*B L_k'],
      single_cond_queries: (
          Float[ArrayT, 'L_q C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
      single_cond_keys: (
          Float[ArrayT, 'L_k C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
  ) -> Float[ArrayT, '*B L_q C']:
    """Multihead cross attention.

    Args:
      queries: Query activations.
      keys: Key activations.
      queries_mask: Mask that indicates valid residues on query.
      keys_mask: Mask that indicates valid residues on keys.
      single_cond_queries: Per-residue conditioning information for query.
      single_cond_keys: Per-residue conditioning information for keys.

    Returns:
      Output activations.
    """
    num_channel = queries.shape[-1]

    queries_act = adaptive_layernorm(
        self.global_config,
        queries,
        single_cond=single_cond_queries,
        name='queries_layer_norm',
    )

    keys_act = adaptive_layernorm(
        self.global_config,
        keys,
        single_cond=single_cond_keys,
        name='keys_layer_norm',
    )

    # Fetch attention dimensions from the config.
    num_head = self.config.num_head
    key_dim = self.config.attn_dim
    value_dim = self.config.attn_dim

    # Casting here reduces memory bandwidth as the outputs of Linear layers
    # will be bfloat16.
    if self.global_config.use_bf16:
      queries_act_maybe_bf16 = queries_act.astype(jnp.bfloat16)
      keys_act_maybe_bf16 = keys_act.astype(jnp.bfloat16)
    else:
      queries_act_maybe_bf16 = queries_act
      keys_act_maybe_bf16 = keys_act

    q = Linear(
        [num_head, key_dim],
        self.global_config,
        use_bias=True,
        name='q_projection',
    )(queries_act_maybe_bf16)
    k = Linear(
        [num_head, key_dim],
        self.global_config,
        use_bias=False,
        name='k_projection',
    )(keys_act_maybe_bf16)
    v = Linear(
        [num_head, value_dim],
        self.global_config,
        use_bias=False,
        name='v_projection',
    )(keys_act_maybe_bf16)

    weighted_avg = tokamax.dot_product_attention(
        query=q,
        key=k,
        value=v,
        bias=None,
        mask=keys_mask[..., None, None, :].astype(jnp.bool_),
        scale=key_dim ** (-0.5),
        implementation='xla',
    ).astype(jnp.float32)

    gate_logits = Linear(
        [num_head, value_dim],
        self.global_config,
        bias_init=1.0,
        initializer='zeros',
        name='gating_query',
    )(queries_act)
    weighted_avg *= jax.nn.sigmoid(gate_logits)

    output = Linear(
        num_channel,
        self.global_config,
        initializer='zeros',
        num_input_dims=2,
        name='output_projection',
    )(weighted_avg)

    return output


class Transition(nn.Module):
  """Transition block for transformer."""

  config: config_lib.TransitionConfig
  global_config: config_lib.GlobalConfig

  @with_namescope
  @typed
  @nn.compact
  def __call__(
      self,
      act: Float[ArrayT, '*B L C'],
      single_cond: (
          Float[ArrayT, 'L C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
  ) -> Float[ArrayT, '*B L C']:
    """Transition module.

    Args:
      act: Input activations.
      single_cond: Per-residue conditioning information.

    Returns:
      Output activations.
    """
    nc = act.shape[-1]

    num_intermediate = int(nc * self.config.num_intermediate_factor)

    act = adaptive_layernorm(
        self.global_config,
        act,
        single_cond=single_cond,
        name='input_layer_norm',
    )

    act = Linear(
        num_intermediate * 2,
        global_config=self.global_config,
        initializer='relu',
        name='transition1',
    )(act)
    a, b = jnp.split(act, 2, axis=-1)
    c = jax.nn.swish(a) * b

    act = Linear(
        nc,
        global_config=self.global_config,
        initializer='zeros',
        name='transition2',
    )(c)

    return act


class TransformerBlockWithCrossAttention(nn.Module):
  """Transformer block with cross-attention and skip connection."""

  config: config_lib.TransformerConfig
  global_config: config_lib.GlobalConfig

  @with_namescope
  @typed
  @nn.compact
  def __call__(
      self,
      queries: Float[ArrayT, '*B L_q C'],
      keys: Float[ArrayT, '*B L_k D'],
      queries_mask: Int[ArrayT, '*B L_q'],
      keys_mask: Int[ArrayT, '*B L_k'],
      single_cond_queries: (
          Float[ArrayT, 'L_q C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
      single_cond_keys: (
          Float[ArrayT, 'L_k C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
      pair_logits: Float[ArrayT, '*B H L_q L_q'],
      keys_pair_logits: Float[ArrayT, '*B H L_k L_k'],
  ) -> Float[ArrayT, '*B L C']:
    act = queries + SelfAttention(
        self.config.attention,
        self.global_config,
    )(
        queries,
        queries_mask,
        single_cond_queries,
        pair_logits,
    )
    keys_act = keys + SelfAttention(
        self.config.attention,
        self.global_config,
    )(
        keys,
        keys_mask,
        single_cond_keys,
        keys_pair_logits,
    )
    act += CrossAttention(
        self.config.attention,
        self.global_config,
    )(
        queries=act,
        keys=keys_act,
        queries_mask=queries_mask,
        keys_mask=keys_mask,
        single_cond_queries=single_cond_queries,
        single_cond_keys=single_cond_keys,
    )
    act += Transition(
        self.config.transition,
        self.global_config,
    )(
        act,
        single_cond_queries,
    )
    return act


class TransformerWithCrossAttention(nn.Module):
  """Sequence transformer with cross-attention."""

  config: config_lib.TransformerConfig
  global_config: config_lib.GlobalConfig

  @with_namescope
  @typed
  @nn.compact
  def __call__(
      self,
      queries: Float[ArrayT, '*B L_q C'],
      keys: Float[ArrayT, '*B L_k D'],
      queries_mask: Int[ArrayT, '*B L_q'],
      keys_mask: Int[ArrayT, '*B L_k'],
      single_cond_queries: (
          Float[ArrayT, 'L_q C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
      single_cond_keys: (
          Float[ArrayT, 'L_k C_single_cond'] | Float[ArrayT, '1 C_single_cond']
      ),
      pair_cond: Float[ArrayT, '*B L_q L_q H'],
      keys_pair_cond: Float[ArrayT, '*B L_k L_k H'],
  ) -> Float[ArrayT, 'L C']:
    """Transforms residue activations.

    Args:
      queries: Input queries for cross-attention.
      keys: Input keys for cross-attention.
      queries_mask: Mask that indicates valid residues on query.
      keys_mask: Mask that indicates valid residues on keys.
      single_cond_queries: Per-residue sequence conditioning information for
        query.
      single_cond_keys: Per-residue sequence conditioning information for keys.
      pair_cond: Pairwise residue conditioning information.
      keys_pair_cond: Pairwise residue conditioning information for keys.

    Returns:
      Transformed activations.
    """
    c = self.config
    gc = self.global_config

    pair_act = adaptive_layernorm(
        self.global_config,
        pair_cond,
        single_cond=single_cond_queries,
        name='pair_input_layer_norm',
    )
    keys_pair_act = adaptive_layernorm(
        self.global_config,
        keys_pair_cond,
        single_cond=single_cond_keys,
        name='keys_pair_input_layer_norm',
    )

    if c.share_pair_logits_all_blocks:
      num_output = (c.attention.num_head,)
      transpose_axes = (2, 0, 1)
      pair_logits_list_fn = lambda logits: [logits] * c.num_blocks
    else:
      num_output = (c.num_blocks, c.attention.num_head)
      transpose_axes = (2, 3, 0, 1)
      pair_logits_list_fn = lambda logits: jnp.unstack(logits, axis=0)

    pair_logits = Linear(
        num_output=num_output,
        use_bias=True,
        global_config=gc,
        name='pair_logits_projection',
    )(pair_act)
    pair_logits = jnp.transpose(pair_logits, transpose_axes)

    keys_pair_logits = Linear(
        num_output=(c.attention.num_head,),
        use_bias=True,
        global_config=gc,
        name='keys_pair_logits_projection',
    )(keys_pair_act)
    keys_pair_logits = jnp.transpose(keys_pair_logits, (2, 0, 1))

    act = queries
    for block_id, pair_logits_i in enumerate(pair_logits_list_fn(pair_logits)):
      # Explicit module naming matches the Flax Linen submodule scopes in
      # trained checkpoints where nn.remat was previously used.
      act = TransformerBlockWithCrossAttention(
          c,
          gc,
          name=f'CheckpointTransformerBlockWithCrossAttention_{block_id}',
      )(
          act,
          keys,
          queries_mask,
          keys_mask,
          single_cond_queries,
          single_cond_keys,
          pair_logits_i,
          keys_pair_logits,
      )
    return act
