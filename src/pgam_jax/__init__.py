import jax
from jax._src.config import bool_env

# enable x64 by default
if bool_env("JAX_ENABLE_X64", True):
    jax.config.update("jax_enable_x64", True)

# monkey-patch nemos.basis.BSplineEval and nemos.basis.BSPlineConv to have a derivative() method
from ._patch_nemos import *  # isort: skip

from .gam import GAM

__all__ = ["GAM"]
