from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from nemos.basis import AdditiveBasis, BSplineEval, MultiplicativeBasis
from nemos.observation_models import Observations, PoissonObservations
from numpy.typing import ArrayLike

from ._identifiable_features import compute_features_identifiable
from .gcv_compute import gcv_compute_factory
from .iterative_optim import pql_outer_iteration
from .penalty_utils import compute_energy_penalty_tensor, tree_compute_sqrt_penalty


# TODO: Should any other observation model be supported?
def _make_variance_function(
    observation_model: Observations,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Return the variance function V(mu) corresponding to the given observation model.

    Parameters
    ----------
    observation_model :
        A nemos observation model instance.

    Returns
    -------
    :
        A function mapping the mean mu to the variance V(mu).

    Raises
    ------
    NotImplementedError
        If the observation model is not Poisson.
    """
    if isinstance(observation_model, PoissonObservations):
        return lambda mu: mu
    else:
        raise NotImplementedError("Currently only Poisson observations are supported.")


class GAM:
    """
    Generalized Additive Model.

    Wraps supported nemos ``Basis`` objects and an ``ObservationModel`` to fit a GAM using
    IRLS for coefficients, GCV for smoothing-parameter selection, and identifiability
    constraints on the basis functions.

    Parameters
    ----------
    basis :
        A nemos ``Basis`` describing the smooth terms.
        Must be a ``BSpline`` or an additive or multiplicative composite of ``BSpline`` bases.
    observation_model :
        A nemos observation model. Default is ``PoissonObservations()``.
    maxiter :
        Maximum number of outer PQL iterations. Default is 100.
    tol_update :
        Convergence tolerance on coefficient updates. Default is 1e-6.
    tol_optim :
        Tolerance for the inner GCV optimization (L-BFGS-B). Default is 1e-10.

    Attributes
    ----------
    coef_ :
        Fitted coefficients (available after ``fit``).
    intercept_ :
        Fitted intercept (available after ``fit``).
    regularizer_strength_ :
        Log-space regularization strengths for each smooth term (available after ``fit``).
    n_iter_ :
        Number of outer PQL iterations performed (available after ``fit``).
    """

    def __init__(
        self,
        basis: BSplineEval | AdditiveBasis | MultiplicativeBasis,
        observation_model: Observations = PoissonObservations(),
        maxiter: int = 100,
        tol_update: float = 1e-6,
        tol_optim: float = 1e-10,
    ) -> None:
        self.basis = basis
        self.observation_model = observation_model
        self.variance_function = _make_variance_function(self.observation_model)
        self.maxiter = maxiter
        self.tol_update = tol_update
        self.tol_optim = tol_optim
        self.n_simpson_sample = int(1e4)

        self._positive_mon_func_for_lambda = jnp.exp
        self._inner_func = gcv_compute_factory(
            self._positive_mon_func_for_lambda,
            lambda x: x[..., :-1],
            lambda x: x[..., :-1, :-1],
            1.5,
        )

    def initialize_params(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize model parameters to zeros.

        Parameters
        ----------
        X :
            Design matrix, shape ``(n_samples, n_features)``.
        y :
            Response variable, shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.

        Returns
        -------
        :
            Zero-initialized (coefficients, intercept).
        """
        in_ndim = X.shape[1]
        out_ndim = 1 if y.ndim == 1 else y.shape[1]

        coef = jnp.zeros(in_ndim)
        intercept = jnp.zeros(out_ndim)

        return (coef, intercept)

    # TODO: Move these into a WiggleRegularizer class or something?
    def _compute_sqrt_penalty(self, *args):
        """
        Compute the square-root of the penalty matrix.

        Passed to ``pql_outer_iteration``.
        Delegates to ``tree_compute_sqrt_penalty`` with identifiability constraints
        (drops last column/row per smooth) and the exp parameterization for lambda.
        """
        return tree_compute_sqrt_penalty(
            *args,
            shift_by=0,
            positive_mon_func=self._positive_mon_func_for_lambda,
            apply_identifiability=lambda x: x[..., :-1],
        )

    def _get_penalty_tree(self) -> list[jnp.ndarray]:
        """
        Compute the penalty tensor tree for all smooth terms.

        Delegatest to ``compute_energy_penalty_tensor``.
        """
        return compute_energy_penalty_tensor(
            self.basis, self.n_simpson_sample, penalize_null_space=True
        )

    def get_design_matrix(self, inputs: tuple[ArrayLike, ...]) -> jnp.ndarray:
        """
        Build the centered, identifiability-constrained design matrix with NaNs zeroed out.

        Parameters
        ----------
        inputs :
            Input arrays, one per input dimension of the basis.

        Returns
        -------
        :
            Design matrix with NaNs zeroed out and columns mean-centered,
            shape ``(n_samples, n_features)``.
        """
        X = compute_features_identifiable(self.basis, *inputs)
        X = jnp.array(X)

        # TODO: Drop rows with NaNs instead?
        # original PGAM zeros out nans, but nemos drops rows with NaNs
        X = jnp.where(jnp.isnan(X), 0.0, X)
        # center columns
        X = X - X.mean(axis=0)

        return X

    def _init_regularizer_strength(
        self, penalty_tree: list[jnp.ndarray]
    ) -> list[jnp.ndarray]:
        """Initialize log-space regularizer strengths to zero (i.e. lambda=1) for each penalty component."""
        # in the example there was: regularizer_strength = [jnp.log(jnp.array([1.0, 2.0]))] * 2
        return [jnp.zeros(pen.shape[0]) for pen in penalty_tree]

    # TODO: Do we need to take X in fit? or is this fine?
    # TODO: Accept the initial regularizer_strength here or in the constructor?
    # In the constructor it's tricky because it's derived from the penalty_tree, which would have to be calculated there
    # (Although it could just be stashed in self for later.)
    # it's optimized, so in that sense it's similar to params, and those are given here
    def fit(
        self,
        xi: tuple[ArrayLike, ...],
        y: ArrayLike,
        init_params: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        init_regularizer_strength: list[jnp.ndarray] | None = None,
    ) -> GAM:
        """
        Fit the GAM to data.

        Delegates to ``pql_outer_iteration``, which alternates between IRLS
        (updating coefficients) and GCV-based optimizationo f the smoothing
        parameters until convergence.

        Parameters
        ----------
        xi :
            Input arrays, one per input dimension of the basis.
        y :
            Response variable, shape ``(n_samples,)``.
        init_params :
            Initial (coefficients, intercept). If None, initialized to zeros.
        init_regularizer_strength :
            Initial log-space regularization strengths. If None, initialized to zeros
            (lambda=1 for every penalty component).

        Returns
        -------
        :
            The fitted model, with ``coef_``, ``intercept_``, ``regularizer_strength_``,
            and ``n_iter_`` set.
        """
        # TODO: Handle different types if accepting xi instead of X
        if not isinstance(xi, tuple):
            raise TypeError("Inputs xi have to be wrapped in a tuple.")

        X = self.get_design_matrix(xi)
        penalty_tree = self._get_penalty_tree()
        y = jnp.asarray(y)

        # TODO: Pull out the GLM initialization here?
        if init_params is None:
            init_params = self.initialize_params(X, y)

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
        self.regularizer_strength_ = opt_pen
        self.n_iter_ = n_iter

        return self

    def _predict(
        self,
        params: tuple[jnp.ndarray, jnp.ndarray],
        xi: tuple[ArrayLike, ...],
    ) -> jnp.ndarray:
        """
        Compute predicted mean response for the given parameters and inputs.

        Parameters
        ----------
        params :
            Model (coefficients, intercept).
        xi :
            Input arrays, one per input dimension of the basis.

        Returns
        -------
        :
            Predicted mean response (after applying the inverse link function),
            shape ``(n_samples,)``.
        """
        w, b = params
        return self.observation_model.default_inverse_link_function(
            self.get_design_matrix(xi) @ w + b
        )

    def predict(self, xi: tuple[ArrayLike, ...]) -> jnp.ndarray:
        """
        Predict the mean response for new inputs.

        Parameters
        ----------
        xi :
            Input arrays, one per input dimension of the basis.

        Returns
        -------
        :
            Predicted mean response, shape ``(n_samples,)``.
        """
        params = (self.coef_, self.intercept_)
        return self._predict(params, xi)

    def score(
        self,
        xi: tuple[ArrayLike, ...],
        y: ArrayLike,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        """
        Compute the log-likelihood score of the model on the given data.

        Parameters
        ----------
        xi :
            Input arrays, one per input dimension of the basis.
        y :
            Observed response variable, shape ``(n_samples,)``.
        aggregate_sample_scores :
            Function to aggregate per-sample log-likelihoods. Default is ``jnp.mean``.

        Returns
        -------
        score :
            Aggregated log-likelihood score.
        """
        params = (self.coef_, self.intercept_)
        return self.observation_model.log_likelihood(
            y,
            self._predict(params, xi),
            # scale = 1.0,
            aggregate_sample_scores=aggregate_sample_scores,
        )
