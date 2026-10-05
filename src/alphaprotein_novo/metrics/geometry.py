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

"""Geometry functions for rigid-body alignment."""

from collections.abc import Callable, Sequence

import numpy as np


def transform_ls(
    x: np.ndarray,
    b: np.ndarray,
    *,
    allow_reflection: bool = False,
) -> np.ndarray:
  """Find the least squares best fit rotation between two sets of N points.

  Solve Ax = b for A. Where A is the transform rotating x^T into b^T.

  Args:
    x: NxD numpy array of coordinates. Usually dimension D is 3.
    b: NxD numpy array of coordinates. Usually dimension D is 3.
    allow_reflection: Whether the returned transformation can reflect as well as
      rotate.

  Returns:
    Matrix A transforming x into b, i.e. s.t. Ax^T = b^T.
  """
  assert x.shape[1] >= b.shape[1]
  assert b.shape[0] == x.shape[0], '%d, %d' % (b.shape[0], x.shape[0])
  # First postmultiply by x.;
  # Axx^t = b x^t
  bxt = np.dot(b.transpose(), x) / b.shape[0]

  u, _, v = np.linalg.svd(bxt)

  r = np.dot(u, v)
  if not allow_reflection:
    flip = np.ones((v.shape[1], 1))
    flip[v.shape[1] - 1, 0] = np.sign(np.linalg.det(r))
    r = np.dot(u, v * flip)

  return r


def get_alignment_transform(
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
  """Get transform to align x to y considering only specified indices.

  Args:
    x: NxD np array of coordinates.
    y: NxD np array of coordinates.
    x_indices: An np array of indices for `x` that will be used in the
      alignment. Must be of the same length as `y_included_idxs`.
    y_indices: An np array of indices for `y` that will be used in the
      alignment. Must be of the same length as `x_included_idxs`.

  Returns:
    A function that takes a numpy array with coordinates and transforms it
    (centers, rotates, and translates) so that is it aligned to y.

  Raises:
    ValueError: If the number of included indices is not the same for both
    input arrays.
  """
  if len(x_indices) != len(y_indices):
    raise ValueError(
        'Number of included indices must be the same for both input arrays,'
        f' but got for x: {len(x_indices)}, and for y: {len(y_indices)}.'
    )

  x = x[x_indices, :]
  y = y[y_indices, :]

  x_mean = np.mean(x, axis=0)
  y_mean = np.mean(y, axis=0)

  t = transform_ls(x - x_mean, y - y_mean)
  return lambda x: np.dot(x - x_mean, t.transpose()) + y_mean


def align(
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
) -> np.ndarray:
  """Align x to y considering only included_idxs.

  Args:
    x: NxD np array of coordinates.
    y: NxD np array of coordinates.
    x_indices: An np array of indices for `x` that will be used in the
      alignment. Must be of the same length as `y_included_idxs`.
    y_indices: An np array of indices for `y` that will be used in the
      alignment. Must be of the same length as `x_included_idxs`.

  Returns:
    NxD np array of points obtained by applying a rigid transformation to x.
    These points are aligned to y and the alignment is the optimal alignment
    over the points in included_idxs.
  """
  align_transform = get_alignment_transform(
      x=x, y=y, x_indices=x_indices, y_indices=y_indices
  )
  return align_transform(x)


def centroid_rmsd(coords_list: Sequence[np.ndarray]) -> float:
  """RMSD of coordinate arrays around their centroid.

  Calculates the sample standard deviation / RMSD of coordinate vectors
  around their arithmetic mean centroid with N - 1 degrees of freedom.

  Args:
    coords_list: Sequence of coordinate arrays to compare.

  Returns:
    RMSD value, or ``np.nan`` if fewer than 2 coordinate arrays are provided.
  """
  if len(coords_list) < 2:
    return float(np.nan)
  coords = np.stack(coords_list)
  centroid = np.mean(coords, axis=0)
  total_variance = np.sum((coords - centroid) ** 2) / (len(coords) - 1)
  return float(np.sqrt(total_variance))
