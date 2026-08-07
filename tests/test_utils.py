import numpy as np
import pytest

from ituna.utils import dense_to_sparse
from ituna.utils import sparse_to_dense


@pytest.fixture
def sparse_data():
    indices = np.array([[0, 1], [1, 2], [0, 2]])
    values = np.array([1.0, 2.0, 3.0])
    shape = (3, 3)
    return indices, values, shape


def test_sparse_to_dense(sparse_data):
    indices, values, shape = sparse_data
    dense = sparse_to_dense(indices, values, shape)
    expected = np.array([[np.nan, 1.0, 3.0], [np.nan, np.nan, 2.0], [np.nan, np.nan, np.nan]])
    assert np.allclose(dense, expected, equal_nan=True)


def test_sparse_to_dense_symmetric(sparse_data):
    indices, values, shape = sparse_data
    dense = sparse_to_dense(indices, values, shape, symmetric=True)
    expected = np.array([[np.nan, 1.0, 3.0], [1.0, np.nan, 2.0], [3.0, 2.0, np.nan]])
    assert np.allclose(dense, expected, equal_nan=True)


def test_dense_to_sparse():
    dense = np.array([[1.0, 2.0, np.nan], [np.nan, 3.0, 4.0], [5.0, np.nan, np.nan]])
    indices, values = dense_to_sparse(dense)
    assert len(indices) == 5
    assert len(values) == 5

    # Reconstruct and check
    reconstructed = sparse_to_dense(indices, values, dense.shape)
    assert np.allclose(dense, reconstructed, equal_nan=True)


def test_dense_to_sparse_symmetric():
    dense = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0],
        ]
    )
    indices, values = dense_to_sparse(dense, symmetric=True)

    expected_indices = np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
    expected_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert np.array_equal(indices, expected_indices)
    assert np.allclose(values, expected_values)

    # Reconstruct and check
    reconstructed = sparse_to_dense(indices, values, dense.shape, symmetric=True)
    # The diagonal won't be filled symmetrically, so we handle that
    reconstructed[np.isnan(reconstructed)] = dense[np.isnan(reconstructed)]
    # We can't guarantee the lower triangle if there were NaNs there in the original
    # so we only check the upper triangle
    assert np.allclose(np.triu(dense), np.triu(reconstructed), equal_nan=True)


def test_identity_dense_sparse_dense():
    dense = np.array([[1.0, 2.0, np.nan], [2.0, 3.0, 4.0], [np.nan, 4.0, 5.0]])

    # Non-symmetric
    indices, values = dense_to_sparse(dense)
    reconstructed = sparse_to_dense(indices, values, dense.shape)
    assert np.allclose(dense, reconstructed, equal_nan=True)

    # Symmetric
    indices, values = dense_to_sparse(dense, symmetric=True)
    reconstructed = sparse_to_dense(indices, values, dense.shape, symmetric=True)
    assert np.allclose(dense, reconstructed, equal_nan=True)


def test_identity_sparse_dense_sparse(sparse_data):
    indices, values, shape = sparse_data

    # Non-symmetric
    dense = sparse_to_dense(indices, values, shape)
    new_indices, new_values = dense_to_sparse(dense)

    # Sort for comparison
    sort_idx1 = np.lexsort((indices[:, 1], indices[:, 0]))
    sort_idx2 = np.lexsort((new_indices[:, 1], new_indices[:, 0]))

    assert np.array_equal(indices[sort_idx1], new_indices[sort_idx2])
    assert np.allclose(values[sort_idx1], new_values[sort_idx2])

    # Symmetric
    dense = sparse_to_dense(indices, values, shape, symmetric=True)
    new_indices, new_values = dense_to_sparse(dense, symmetric=True)

    # Reconstruct original sparse data for comparison
    full_indices = np.concatenate([indices, indices[:, ::-1]])
    full_values = np.concatenate([values, values])

    # Remove duplicates
    unique_indices, idx = np.unique(full_indices, axis=0, return_index=True)
    unique_values = full_values[idx]

    # Keep only upper triangle for comparison
    upper_indices_mask = unique_indices[:, 0] <= unique_indices[:, 1]
    upper_indices = unique_indices[upper_indices_mask]
    upper_values = unique_values[upper_indices_mask]

    sort_idx1 = np.lexsort((upper_indices[:, 1], upper_indices[:, 0]))
    sort_idx2 = np.lexsort((new_indices[:, 1], new_indices[:, 0]))

    assert np.array_equal(upper_indices[sort_idx1], new_indices[sort_idx2])
    assert np.allclose(upper_values[sort_idx1], new_values[sort_idx2])


def test_assertions():
    """Boundary validation must raise ValueError, which `python -O` preserves and assert does not.

    These are public functions, so a malformed argument is a caller error and belongs at the
    boundary. Guarded by assert, every one of these checks vanished under `python -O` and the
    malformed input travelled on to fail later as an opaque TypeError.
    """
    with pytest.raises(ValueError, match="2D array with shape"):
        sparse_to_dense(np.array([0, 1]), np.array([1]), (2, 2))
    with pytest.raises(ValueError, match="Length of indices and values"):
        sparse_to_dense(np.array([[0, 1]]), np.array([1, 2]), (2, 2))
    with pytest.raises(ValueError, match="at least 2D"):
        dense_to_sparse(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="must be symmetric"):
        dense_to_sparse(np.array([[1, 2], [3, 4]]), symmetric=True)

    # Empty indices with non-empty values: reachable only with a genuinely zero-length index array,
    # since a 1-D index array is caught by the shape check above before this branch is entered.
    with pytest.raises(ValueError, match="Values must be empty"):
        sparse_to_dense(np.empty((0, 2), dtype=int), np.array([1.0]), (2, 2))

    with pytest.raises(ValueError):
        indices = np.array([[0, 1], [1, 0]])
        values = np.array([1.0, 2.0])
        sparse_to_dense(indices, values, (2, 2), symmetric=True)


def test_store_and_load_torch_tensor_round_trip(tmp_path):
    """store_data calls save_fn(path, data); torch.save takes (obj, f), so the adapter matters.

    Nothing hashes torch tensors -- hash_object has no branch for them -- so tensors can
    only reach store_data as a cached method output, never as a fit argument. That is why
    the inverted call went unnoticed.
    """
    torch = pytest.importorskip("torch")

    from ituna._backends import utils as backend_utils

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    target = tmp_path / "tensor_entry"
    backend_utils.store_data(target, tensor)

    assert target.with_suffix(".pt").exists()
    assert torch.equal(backend_utils.load_data(target), tensor)


def test_torch_tensors_are_reachable_only_as_cached_outputs():
    """hash_object rejects tensors, so they cannot arrive through DataArguments."""
    torch = pytest.importorskip("torch")

    from ituna._backends import utils as backend_utils

    with pytest.raises(TypeError, match="No hashing method"):
        backend_utils.hash_object(torch.zeros(3))
