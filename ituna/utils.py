from typing import Tuple

import numpy as np
import typeguard


@typeguard.typechecked
def sparse_to_dense(indices: np.ndarray, values: np.ndarray, shape: Tuple[int, ...], fill_value: float = np.nan, symmetric: bool = False) -> np.ndarray:
    """
    Convert sparse representation to a dense matrix.

    Parameters
    ----------
    indices : array-like of shape (n_pairs, 2)
        The indices of the non-zero elements.
    values : array-like of shape (n_pairs,) or (n_pairs, ...)
        The values of the non-zero elements.
    shape : tuple of int
        The shape of the dense matrix.
    fill_value : float, default=np.nan
        The value to fill the dense matrix with for unknown elements.
    symmetric : bool, default=False
        If True, the dense matrix is assumed to be symmetric, and the values
        are filled for both (i, j) and (j, i).

    Returns
    -------
    numpy.ndarray
        The dense matrix.
    """
    if len(indices) == 0:
        if len(values) != 0:
            raise ValueError(f"Values must be empty if indices are empty, got {len(values)} value(s)")
        return np.full(shape, fill_value, dtype=values.dtype)

    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"Indices must be a 2D array with shape (n_pairs, 2), got shape {indices.shape}")
    if len(indices) != len(values):
        raise ValueError(f"Length of indices and values must be the same, got {len(indices)} and {len(values)}")

    if symmetric:
        # Check for duplicate symmetric indices
        off_diagonal_indices = indices[indices[:, 0] != indices[:, 1]]
        swapped_indices = off_diagonal_indices[:, ::-1]

        # Create a set of tuples for efficient lookup
        indices_set = set(map(tuple, off_diagonal_indices))

        for i, j in swapped_indices:
            if (i, j) in indices_set:
                raise ValueError(f"Symmetric sparse array cannot contain both ({j}, {i}) and ({i}, {j})")

    dense_array = np.full(shape, fill_value, dtype=values.dtype if np.issubdtype(values.dtype, np.number) else object)
    for (i, j), value in zip(indices, values):
        dense_array[i, j] = value
        if symmetric:
            dense_array[j, i] = value
    return dense_array


@typeguard.typechecked
def dense_to_sparse(dense_array: np.ndarray, symmetric: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a dense matrix to a sparse representation.
    Parameters
    ----------
    dense_array : numpy.ndarray
        The dense matrix.
    symmetric : bool, default=False
        If True, the dense matrix is assumed to be symmetric, and only the
        upper triangular part is converted to sparse.
    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices and values of the non-nan elements.
    """
    if dense_array.ndim < 2:
        raise ValueError(f"Dense array must be at least 2D, got {dense_array.ndim} dimension(s) with shape {dense_array.shape}")
    if symmetric and not np.allclose(dense_array, dense_array.T, equal_nan=True):
        raise ValueError("Dense array must be symmetric when symmetric=True")

    nan_mask = ~np.isnan(dense_array)
    if symmetric:
        nan_mask &= np.triu(np.ones_like(dense_array, dtype=bool))

    indices = np.argwhere(nan_mask)

    values = dense_array[indices[:, 0], indices[:, 1]]
    return indices, values
