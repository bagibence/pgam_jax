import jax.numpy as jnp
import numpy as np
import pytest

from pgam_jax._utils import singular_value_keep_mask


def test_singular_value_keep_mask_detects_dependent_columns():
    matrix = jnp.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    keep = singular_value_keep_mask(singular_values, matrix.shape)

    np.testing.assert_array_equal(keep, [True, False])
