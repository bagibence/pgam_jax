from typing import TYPE_CHECKING

from jaxtyping import Array, Float
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from pynapple import Tsd, TsdFrame, TsdTensor

type JaxFloatScalar = Float[Array, ""]
type JaxFloatMatrix = Float[Array, "n n"]
# One model input. A pynapple time series also carries its time axis and time support.
type InputArray = ArrayLike | Tsd | TsdFrame | TsdTensor
# The response, shape (n_samples,). A pynapple Tsd also carries its time axis and time support.
type ResponseArray = ArrayLike | Tsd
