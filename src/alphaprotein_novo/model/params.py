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

"""Model param loading for AlphaProtein Novo.

Reads model weights from the binary record format (scope/name/dtype/shape/data)
compressed with zstandard. Compatible with single files (model.bin.zst) and
sharded files (model.bin.zst.0, model.bin.zst.1, ...).

The binary record format is identical to that used by alphafold3.
"""

from __future__ import annotations

import bisect
import collections
from collections.abc import Iterator, Sequence
import contextlib
import io
import os
import re
import struct
import sys
from typing import Any, IO, NamedTuple

from etils import epath
import haiku as hk
import jax.numpy as jnp
import numpy as np
import zstandard


class Record(NamedTuple):
  """A single parameter record (scope, name, array)."""

  scope: str
  name: str
  arr: np.ndarray


class RecordError(Exception):
  """Error reading a record."""


def encode_record(*, scope: str, name: str, arr: np.ndarray) -> bytes:
  """Encodes a single haiku param as bytes.

  Format (all little-endian):
    header:  5 x int32 = (scope_len, name_len, dtype_len, ndim, data_len)
    payload: scope_bytes + name_bytes + dtype_bytes + shape_ints + array_data

  Args:
    scope: The scope of the parameter.
    name: The name of the parameter.
    arr: The parameter array.

  Returns:
    The encoded parameter as bytes.
  """
  scope_bytes = scope.encode('utf-8')
  name_bytes = name.encode('utf-8')
  dtype_bytes = str(arr.dtype).encode('utf-8')
  shape = arr.shape
  arr = np.ascontiguousarray(arr)
  if sys.byteorder == 'big':
    arr = arr.byteswap()
  arr_buffer = arr.tobytes('C')
  header = struct.pack(
      '<5i',
      len(scope_bytes),
      len(name_bytes),
      len(dtype_bytes),
      len(shape),
      len(arr_buffer),
  )
  shape_bytes = struct.pack(f'{len(shape)}i', *shape)
  return (
      header + scope_bytes + name_bytes + dtype_bytes + shape_bytes + arr_buffer
  )


def _read_record(stream: IO[bytes]) -> Record | None:
  """Reads a record encoded by `encode_record` from a byte stream."""
  header_size = struct.calcsize('<5i')
  header = stream.read(header_size)
  if not header:
    return None
  if len(header) < header_size:
    raise RecordError(f'Incomplete header: {len(header)=} < {header_size=}')
  scope_len, name_len, dtype_len, shape_len, arr_buffer_len = struct.unpack(
      '<5i', header
  )
  fmt = f'<{scope_len}s{name_len}s{dtype_len}s{shape_len}i'
  payload_size = struct.calcsize(fmt) + arr_buffer_len
  payload = stream.read(payload_size)
  if len(payload) < payload_size:
    raise RecordError(f'Incomplete payload: {len(payload)=} < {payload_size=}')
  scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
  scope = scope.decode('utf-8')
  name = name.decode('utf-8')
  dtype = dtype.decode('utf-8')
  arr = np.frombuffer(payload[-arr_buffer_len:], dtype=dtype)
  arr = np.reshape(arr, shape)
  if sys.byteorder == 'big':
    arr = arr.byteswap()
  return Record(scope=scope, name=name, arr=arr)


def read_records(stream: IO[bytes]) -> Iterator[Record]:
  """Fully reads the contents of a byte stream."""
  while (record := _read_record(stream)) is not None:
    yield record


class _MultiFileIO(io.RawIOBase):
  """A file-like object that presents a concatenated view of multiple files."""

  def __init__(self, files: Sequence[epath.PathLike]):
    self._files = [epath.Path(file) for file in files]
    self._stack = contextlib.ExitStack()
    self._handles = [
        self._stack.enter_context(file.open('rb')) for file in self._files
    ]
    self._sizes = []
    for handle in self._handles:
      handle.seek(0, os.SEEK_END)
      self._sizes.append(handle.tell())
    self._length = sum(self._sizes)
    self._offsets = [0]
    for s in self._sizes[:-1]:
      self._offsets.append(self._offsets[-1] + s)
    self._abspos = 0
    self._relpos = (0, 0)

  def _abs_to_rel(self, pos: int) -> tuple[int, int]:
    idx = bisect.bisect_right(self._offsets, pos) - 1
    return idx, pos - self._offsets[idx]

  def close(self):
    self._stack.close()

  @property
  def closed(self) -> bool:
    return all(handle.closed for handle in self._handles)

  def fileno(self) -> int:
    return -1

  def readable(self) -> bool:
    return True

  def tell(self) -> int:
    return self._abspos

  def seek(self, pos: int, whence: int = os.SEEK_SET, /):
    match whence:
      case os.SEEK_SET:
        pass
      case os.SEEK_CUR:
        pos += self._abspos
      case os.SEEK_END:
        pos = self._length - pos
      case _:
        raise ValueError(f'Invalid whence: {whence}')
    self._abspos = pos
    self._relpos = self._abs_to_rel(pos)

  # pyrefly: ignore[bad-override]
  def readinto(self, buffer: bytearray | memoryview[int], /) -> int:
    b = buffer
    result = 0
    mem = memoryview(b)
    while mem:
      file_handle = self._handles[self._relpos[0]]
      file_handle.seek(self._relpos[1])
      if hasattr(file_handle, 'readinto'):
        count = file_handle.readinto(mem)
      else:
        data = file_handle.read(len(mem))
        count = len(data)
        mem[:count] = data

      result += count
      self._abspos += count
      self._relpos = self._abs_to_rel(self._abspos)
      mem = mem[count:]
      if self._abspos == self._length:
        break
    return result


@contextlib.contextmanager
def open_for_reading(
    model_files: Sequence[epath.PathLike],
    *,
    is_compressed: bool,
):
  """Opens one or more model files for reading, optionally decompressing."""
  with contextlib.closing(_MultiFileIO(model_files)) as f:
    if is_compressed:
      buffered = io.BufferedReader(f)
      yield zstandard.ZstdDecompressor().stream_reader(buffered)
    else:
      yield f


def _match_model(
    paths: Sequence[epath.Path],
    pattern: re.Pattern[str],
) -> dict[str, list[epath.Path]]:
  """Match files in a directory with a pattern, and group by model name."""
  models = collections.defaultdict(list)
  for path in paths:
    match = pattern.fullmatch(path.name)
    if match:
      models[match.group('model_name')].append(path)
  return {k: sorted(v) for k, v in models.items()}


def select_model_files(
    model_dir: epath.PathLike,
    model_name: str | None = None,
) -> tuple[list[epath.Path], bool]:
  """Select the model files from a model directory."""
  model_dir = epath.Path(model_dir)
  if model_dir.exists():
    files = [file for file in model_dir.iterdir() if file.is_file()]
  else:
    files = []

  for pattern, is_compressed in (
      (r'(?P<model_name>.*)\.[0-9]+\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin\.zst\.[0-9]+$', True),
      (r'(?P<model_name>.*)\.[0-9]+\.bin$', False),
      (r'(?P<model_name>.*)\.bin\.[0-9]+$', False),
      (r'(?P<model_name>.*)\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin$', False),
  ):
    models = _match_model(files, re.compile(pattern))
    if model_name is not None:
      if model_name in models:
        return models[model_name], is_compressed
    else:
      if models:
        if len(models) > 1:
          raise RuntimeError(f'Multiple models matched in {model_dir}')
        _, model_files = models.popitem()
        return model_files, is_compressed
  raise FileNotFoundError(f'No models matched in {model_dir}')


def _unflatten_params(
    flat_params: dict[str, dict[str, jnp.Array]],
) -> hk.Params:
  """Reconstructs nested Haiku params from flat scope/name representation.

  Converts {'ModuleA/SubModule': {'w': arr}} back to
  {'ModuleA': {'SubModule': {'w': arr}}}.

  Args:
    flat_params: A flattened dictionary of Haiku parameters.

  Returns:
    A nested dictionary of Haiku parameters.
  """
  nested: dict[str, Any] = {}
  for scope, name_dict in flat_params.items():
    parts = scope.split('/')
    current = nested
    for part in parts:
      current = current.setdefault(part, {})
    current.update(name_dict)
  return nested


def get_model_haiku_params(
    model_dir: epath.PathLike,
    model_name: str | None = None,
) -> hk.Params:
  """Get the Haiku parameters from a model directory.

  Args:
    model_dir: Directory containing .bin.zst model file(s).
    model_name: Optional model name filter. If None, auto-detects.

  Returns:
    Haiku params dict with nested structure matching the original checkpoint.

  Raises:
    FileNotFoundError: If no model files are found in the directory.
    RecordError: If a record is malformed.
  """
  flat_params: dict[str, dict[str, jnp.Array]] = {}
  model_files, is_compressed = select_model_files(model_dir, model_name)
  with open_for_reading(model_files, is_compressed=is_compressed) as stream:
    for scope, name, arr in read_records(stream):
      flat_params.setdefault(scope, {})[name] = jnp.array(arr)
  if not flat_params:
    raise FileNotFoundError(f'Model missing from "{model_dir}"')
  return _unflatten_params(flat_params)
