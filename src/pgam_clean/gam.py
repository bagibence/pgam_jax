import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from nemos.observation_models import Observations, PoissonObservations
from nemos.basis._basis import Basis
from .basis._basis_utils import to_gam_basis

from .iterative_optim import pql_outer_iteration
from .gcv_compute import gcv_compute_factory
from .penalty_utils import tree_compute_sqrt_penalty, compute_energy_penalty_tensor


# TODO: Should any other observation model be supported?
def _make_variance_function(observation_model: Observations):
    if isinstance(observation_model, PoissonObservations):
        return lambda mu: mu
    else:
        raise NotImplementedError("Currently only Poisson observations are supported.")


class GAM:
    def __init__(
        self,
        basis: Basis,
        observation_model: Observations = PoissonObservations(),
        maxiter: int = 100,
        tol_update: float = 1e-6,
        tol_optim: float = 1e-10,
    ):
        self.basis = to_gam_basis(basis, identifiability=True)
        self.observation_model = observation_model
        self.variance_function = _make_variance_function(self.observation_model)
        self.positive_mon_func_for_lambda = jnp.exp
        self.maxiter = maxiter
        self.tol_update = tol_update
        self.tol_optim = tol_optim
        self.n_simpson_sample = int(1e4)

        # TODO: positive_mon_func_for_lambda should have a setter or _inner_func a property
        self._inner_func = gcv_compute_factory(
            self.positive_mon_func_for_lambda,
            lambda x: x[..., :-1],
            lambda x: x[..., :-1, :-1],
            1.5,
        )

    def init_params(self, X, y):
        in_ndim = X.shape[1]
        out_ndim = 1 if y.ndim == 1 else y.shape[1]

        coef = jnp.zeros(in_ndim)
        intercept = jnp.zeros(out_ndim)

        return (coef, intercept)

    # TODO: Move these into a WiggleRegularizer class or something?
    def _compute_sqrt_penalty(self, *args):
        return tree_compute_sqrt_penalty(
            *args,
            shift_by=0,
            positive_mon_func=self.positive_mon_func_for_lambda,
            apply_identifiability=lambda x: x[..., :-1],
        )

    def get_penalty_tree(self):
        # This needs identifiability=False
        orig_identifiability = self.basis.identifiability
        self.basis.identifiability = False

        energy_penalty = compute_energy_penalty_tensor(
            self.basis, self.n_simpson_sample, penalize_null_space=True
        )

        # set it back
        self.basis.identifiability = orig_identifiability

        return energy_penalty

    def get_design_matrix(self, inputs):
        # requires identifiability setter I added to some bases
        # identifiability=True is needed for compute_features to give the right dimensions
        orig_identifiability = self.basis.identifiability
        self.basis.identifiability = True

        X = self.basis.compute_features(*inputs)
        X = np.array(X)

        # original PGAM zeros out nans
        X[np.isnan(X)] = 0.0
        # center columns
        X = X - X.mean(axis=0)

        # set it back
        self.basis.identifiability = orig_identifiability

        return X

    def _init_regularizer_strength(self, penalty_tree):
        """Initialize log-space regularizer strengths to 0 (i.e., λ=1) for each penalty component."""
        # in the example there was: regularizer_strength = [jnp.log(jnp.array([1.0, 2.0]))] * 2
        return [jnp.zeros(pen.shape[0]) for pen in penalty_tree]

    # TODO: Do we need to take X in fit? or is this fine?
    # TODO: Accept the initial regularizer_strength here or in the constructor?
    # In the constructor it's tricky because it's derived from the penalty_tree, which would have to be calculated there
    # (Although it could just be stashed in self for later.)
    # it's optimized, so in that sense it's similar to params, and those are given here
    def fit(self, xi, y, init_params=None, init_regularizer_strength=None):
        # TODO: Handle different types if accepting xi instead of X
        if not isinstance(xi, tuple):
            raise TypeError("Inputs xi have to be wrapped in a tuple.")

        X = self.get_design_matrix(xi)
        penalty_tree = self.get_penalty_tree()

        # TODO: Pull out the GLM initialization here?
        if init_params is None:
            init_params = self.init_params(X, y)

        if init_regularizer_strength is None:
            init_regularizer_strength = self._init_regularizer_strength(penalty_tree)

        opt_coef, opt_pen, n_iter = pql_outer_iteration(
            init_regularizer_strength,  # initial log(λ) for each smooth
            init_params,  # initial (coefficients, intercept)
            X,
            y,
            penalty_tree,
            self.observation_model,
            self.variance_function,
            self._inner_func,
            self._compute_sqrt_penalty,
            fisher_scoring=False,  # use observed information, not expected
            max_iter=self.maxiter,
            tol_update=self.tol_update,  # convergence tolerance for coefficient updates
            tol_optim=self.tol_optim,  # tolerance for inner GCV optimization
        )

        self.coef_, self.intercept_ = opt_coef
        self.regularizer_strength = opt_pen
        self.n_iter = n_iter

        return self

    def _predict(self, params, xi):
        w, b = params
        return self.observation_model.default_inverse_link_function(
            self.get_design_matrix(xi) @ w + b
        )

    def predict(self, xi):
        params = (self.coef_, self.intercept_)
        return self._predict(params, xi)

    def score(self, xi, y, aggregate_sample_scores=jnp.mean):
        params = (self.coef_, self.intercept_)
        return self.observation_model.log_likelihood(
            y,
            self._predict(params, xi),
            # scale = 1.0,
            aggregate_sample_scores=aggregate_sample_scores,
        )
