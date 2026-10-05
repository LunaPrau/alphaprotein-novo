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

"""Consolidated configuration classes for AlphaProtein Novo."""

import abc
from typing import Any, Literal, Self
import pydantic


class BaseConfig(pydantic.BaseModel):
  """Base config class for AlphaProtein Novo models and modules."""

  model_config = pydantic.ConfigDict(
      extra='forbid',
      validate_assignment=True,
      arbitrary_types_allowed=True,
  )


class GlobalConfig(BaseConfig):
  """Global config for the experiment.

  Attributes:
    num_protein_tokens: The number of protein tokens.
    num_ligand_tokens: The number of ligand tokens.
    max_num_res_unindexed: The maximum number of residues to add unindexed
      features for.
    use_bf16: Whether to use bfloat16. If False, will use float32 instead in
      places where we would normally use bfloat16.
  """

  num_protein_tokens: pydantic.NonNegativeInt = 21
  num_ligand_tokens: pydantic.NonNegativeInt = 43
  max_num_res_unindexed: pydantic.NonNegativeInt | None = 128
  use_bf16: bool = True

  @pydantic.computed_field
  @property
  def num_vocab_tokens(self) -> int:
    """Vocabulary size."""
    return self.num_protein_tokens + self.num_ligand_tokens + 1


class ProcessConfig(BaseConfig):
  """Config for the process."""

  process_name: str
  process_kwargs: dict[str, Any]


class SamplingAlgorithmConfig(BaseConfig, abc.ABC):
  """Base config for a sampling algorithm."""

  name: str


class StochasticSamplingAlgorithmConfig(SamplingAlgorithmConfig):
  """Stochastic sampler."""

  name: Literal['StochasticSamplingAlgorithmConfig'] = pydantic.Field(
      default='StochasticSamplingAlgorithmConfig'
  )

  s_churn: pydantic.NonNegativeFloat = 65.0
  s_t_min: pydantic.NonNegativeFloat = 0.1
  s_t_max: pydantic.NonNegativeFloat = float('inf')
  gamma_max: pydantic.NonNegativeFloat = 0.41421356237309515
  step_scale: pydantic.NonNegativeFloat = 4.0
  sigma_data: pydantic.NonNegativeFloat = 14.4

  @pydantic.field_validator('s_t_max', mode='before')
  @classmethod
  def _validate_s_t_max(cls, value: None | float) -> float:
    if value is None:
      return float('inf')
    return value


class TimestepsConfig(BaseConfig):
  """Timesteps config."""

  timesteps_name: str
  timesteps_kwargs: dict[str, Any]


class DiscreteSamplingAlgorithmConfig(SamplingAlgorithmConfig, abc.ABC):
  """Discrete sampler."""

  name: Literal['DiscreteSamplingAlgorithmConfig'] = pydantic.Field(
      default='DiscreteSamplingAlgorithmConfig'
  )
  temperature: float = 0.2
  num_designable_tokens: int = 20


class MaskedSamplingAlgorithmConfig(DiscreteSamplingAlgorithmConfig):
  """Masked diffusion sampler."""

  name: Literal['MaskedSamplingAlgorithmConfig'] = pydantic.Field(
      default='MaskedSamplingAlgorithmConfig'
  )


SequenceSamplingAlgorithm = MaskedSamplingAlgorithmConfig
StructureSamplingAlgorithm = StochasticSamplingAlgorithmConfig


def _default_structure_process() -> ProcessConfig:
  return ProcessConfig(
      process_name='edm',
      process_kwargs={},
  )


def _default_sequence_process() -> ProcessConfig:
  return ProcessConfig(
      process_name='masked',
      process_kwargs={
          'end': 3.0,
          'lognormal_mean': 1.7,
          'lognormal_std': 2.0,
          'mask_token': 64,
          'start': -4.0,
          'start_t': 0.93,
          't_is_in_0_1': False,
          'tau': 0.3,
          'vocab_size': 20,
      },
  )


def _default_structure_timesteps() -> TimestepsConfig:
  return TimestepsConfig(
      timesteps_name='log_normal',
      timesteps_kwargs={
          'lognormal_mean': 1.7,
          'lognormal_std': 2.0,
          'tmax': 100.0,
          'tmin': 0.01,
      },
  )


def _default_sequence_timesteps() -> TimestepsConfig:
  return TimestepsConfig(
      timesteps_name='log_normal',
      timesteps_kwargs={
          'lognormal_mean': 1.7,
          'lognormal_std': 2.0,
          'tmax': 100.0,
          'tmin': 0.01,
      },
  )


class SamplingConfig(BaseConfig):
  """Configuration for sampling from the model."""

  structure_process: ProcessConfig = pydantic.Field(
      default_factory=_default_structure_process
  )
  sequence_process: ProcessConfig = pydantic.Field(
      default_factory=_default_sequence_process
  )
  num_timesteps: int = 1000
  structure_timesteps: TimestepsConfig = pydantic.Field(
      default_factory=_default_structure_timesteps
  )
  sequence_timesteps: TimestepsConfig = pydantic.Field(
      default_factory=_default_sequence_timesteps
  )
  sequence_sampling_algorithm: SequenceSamplingAlgorithm = pydantic.Field(
      default_factory=MaskedSamplingAlgorithmConfig
  )
  structure_sampling_algorithm: StructureSamplingAlgorithm = pydantic.Field(
      default_factory=StochasticSamplingAlgorithmConfig
  )
  num_partial_diffusion_steps: pydantic.PositiveInt | None = None
  # Emit the sequence sampled by the discrete track rather than the one
  # inferred from the collapse of the super-all-atom virtual atom clouds.
  use_discrete_aatype: bool = True
  # Drop side chain atoms that the emitted residue type requires but whose
  # virtual atom stayed on the placeholder atom.
  prune_unsupported_atoms: bool = False

  @pydantic.model_validator(mode='after')
  def _validate_num_steps(self: Self) -> Self:
    if (
        self.num_partial_diffusion_steps is not None
        and self.num_partial_diffusion_steps > self.num_timesteps
    ):
      raise ValueError(
          'num_partial_diffusion_steps must be less than or equal to the'
          ' number of timesteps.'
      )
    return self


class AttentionConfig(BaseConfig):
  """Config for the attention."""

  attn_dim: int = 32
  num_head: int = 8


class TransitionConfig(BaseConfig):
  """Config for the transition."""

  num_intermediate_factor: int = 2


class TransformerConfig(BaseConfig):
  """Config for the transformer."""

  num_blocks: int = 32
  attention: AttentionConfig = pydantic.Field(default_factory=AttentionConfig)
  transition: TransitionConfig = pydantic.Field(
      default_factory=TransitionConfig
  )
  share_pair_logits_all_blocks: bool = True


class ConditioningConfig(BaseConfig):
  """Config for the conditioning."""

  pair_channel: int = 16
  seq_channel: int = 192
  max_relative_idx: int = 64


class DenoiserConfig(BaseConfig):
  """Config for the denoiser."""

  num_channel: int = 512
  conditioning: ConditioningConfig = pydantic.Field(
      default_factory=ConditioningConfig
  )
  transformer: TransformerConfig = pydantic.Field(
      default_factory=TransformerConfig
  )


class ModelBaseConfig(BaseConfig):
  """Config for the model base."""

  sampling_config: SamplingConfig = pydantic.Field(
      default_factory=SamplingConfig
  )
  denoiser: DenoiserConfig = pydantic.Field(default_factory=DenoiserConfig)


def get_model_config() -> ModelBaseConfig:
  """Returns the ModelBaseConfig for AlphaProtein Novo."""
  return ModelBaseConfig()


def get_global_config() -> GlobalConfig:
  """Returns the GlobalConfig for AlphaProtein Novo."""
  return GlobalConfig()


def get_sampling_config() -> SamplingConfig:
  """Returns sampling config."""
  return SamplingConfig()
