"""Compatibility shims for differences across supported nemos versions."""

import nemos
from packaging.version import Version

_NEMOS_HAS_N_INPUTS = Version(nemos.__version__) >= Version("0.2.7")


def get_n_inputs(basis) -> int:
    """Return the number of inputs a basis component consumes.

    Wraps the attribute rename from ``_n_input_dimensionality`` (nemos < 0.2.7)
    to ``_n_inputs`` (nemos >= 0.2.7).
    """
    if _NEMOS_HAS_N_INPUTS:
        return basis._n_inputs
    return basis._n_input_dimensionality
