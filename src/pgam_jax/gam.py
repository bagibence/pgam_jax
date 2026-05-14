from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from nemos.basis import AdditiveBasis, BSplineEval, MultiplicativeBasis
from nemos.glm.initialize_parameters import INVERSE_FUNCS
from nemos.observation_models import Observations, PoissonObservations
from numpy.typing import ArrayLike
from scipy import stats as sts

from ._identifiable_features import (
    BasisComponentInfo,
    _compute_features_identifiable,
    _get_basis_component_infos,
    _should_drop_basis_col,
    compute_features_identifiable,
)

from ._laplace_reml_fit import laplace_reml_outer_iteration, make_inner_solver
from ._penalty_handler import PenaltyHandler, _drop_last_col
from ._pql_gcv import gcv_compute_factory
from ._pql_reml import reml_compute_factory
from .iterative_optim import (
    VALID_CONVERGENCE_CRITERIA,
    model_constructors_for_weights_and_pseudo_data,
    pql_outer_iteration,
)
from .penalty_utils import compute_energy_penalty_tensor, compute_penalty_blocks
from nemos.inverse_link_function_utils import identity as _id_no_drop


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


def _validate_eval_bases_have_bounds(basis) -> None:
    """Raise if any eval-mode leaf has ``bounds=None``.

    Without explicit bounds, ``nemos`` rescales each input array to ``[0, 1]``
    using its own min and max, so the same physical x maps to different
    normalized coordinates across fit/predict batches.
    """
    missing = [
        leaf
        for leaf in basis._iterate_over_components()
        if isinstance(leaf, BSplineEval) and leaf.bounds is None
    ]
    if not missing:
        return
    types = ", ".join(type(b).__name__ for b in missing)
    raise ValueError(
        f"{len(missing)} eval-mode basis component(s) ({types}) were "
        "constructed without explicit bounds. pgam_jax requires every "
        "eval-mode basis to be built with bounds=(lo, hi) covering the "
        "covariate range, so that fit and predict use the same normalized "
        "coordinates."
    )


def _make_identifiability_dropper(
    basis_component,
    square: bool,
    drop_conv_basis_col: bool,
):
    """
    Per-leaf identifiability function matching ``compute_features_identifiable``.

    Convolutional bases follow ``drop_conv_basis_col``, other bases drop the last column.
    ``square=True`` returns a function that drops both the last row and column for use on penalty matrices.
    """
    if not _should_drop_basis_col(basis_component, drop_conv_basis_col):
        return lambda x: x
    if square:
        return lambda x: x[..., :-1, :-1]
    return lambda x: x[..., :-1]


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
        Outer-loop convergence tolerance. Its meaning depends on
        ``convergence_criterion``. Default is 1e-5.
    tol_optim :
        Tolerance for the inner GCV optimization (L-BFGS-B). Default is 1e-10.
    use_scipy :
        If True, use scipy's L-BFGS-B for both the inner GCV minimization
        and the initial GLM fit instead of jaxopt's. Often faster on CPU.
        Default is False.
    convergence_criterion :
        Outer-loop convergence monitor passed to ``pql_outer_iteration``.
        ``"gcv"`` matches legacy PGAM, while ``"coef"`` and ``"coef_and_reg"``
        are fixed-point style monitors.
        Default is ``"gcv"``.
    drop_conv_basis_col :
        If True, convolutional basis leaves drop their last column for
        identifiability. If False, convolutional basis leaves keep all columns.
        Default is False.
        Convolution doesn't create linearly dependent columns, so in theory there is no need to drop,
        but the option is added for matching the original implementation if required.
    method :
        Smoothing-parameter selection algorithm. Default ``"pql_gcv"``.

        - ``"pql_gcv"``: Generalized Cross-Validation on the PQL-linearized
          working model.
        - ``"pql_reml"``: Restricted Maximum Likelihood on the PQL-linearized
          working model.
        - ``"laplace_reml"``: Laplace-approximated REML on the true GLM
          likelihood at the MAP. Currently Poisson-only (φ fixed at 1).

        The ``pql_*`` methods share the IRLS outer loop (``pql_outer_iteration``);
        ``laplace_reml`` instead optimizes ρ directly over the Laplace REML
        objective, re-fitting β̂ to the MAP at each ρ evaluation.
    method_kwargs :
        Optional per-method tuning dict. For ``"laplace_reml"`` the recognized
        keys are ``inner_solver`` / ``inner_solver_kwargs`` (MAP β̂ solve) and
        ``outer_solver`` / ``outer_solver_kwargs`` (ρ optimization), each a
        ``nemos.solvers`` registry name and its constructor kwargs.

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
    cov_beta_ :
        Posterior covariance of the full coefficient vector ``[intercept, coef_]``
        treating smoothing parameters as fixed (available after ``fit``).
        Includes the dispersion factor: ``φ̂ · (X'WX + S_λ)⁻¹``.
    scale_ :
        Estimated dispersion parameter φ̂ (available after ``fit``).
        Poisson/Bernoulli models always return 1.0; Gaussian/Gamma models
        estimate φ̂ from Pearson residuals divided by residual degrees of freedom.
    edf_ :
        Effective degrees of freedom — Wood's ``edf1 = 2·tr(F) − tr(F²)``
        (available after ``fit``).
    dof_resid_ :
        Residual degrees of freedom ``n_obs − edf_`` (available after ``fit``).
    """

    def __init__(
        self,
        basis: BSplineEval | AdditiveBasis | MultiplicativeBasis,
        observation_model: Observations = PoissonObservations(),
        maxiter: int = 100,
        tol_update: float = 1e-5,
        tol_optim: float = 1e-10,
        use_scipy: bool = False,
        convergence_criterion: str = "gcv",
        drop_conv_basis_col: bool = False,
        method: Literal["pql_gcv", "pql_reml", "laplace_reml"] = "pql_gcv",
        method_kwargs: dict | None = None,
    ) -> None:
        # TODO: Make basis immutable
        if convergence_criterion not in VALID_CONVERGENCE_CRITERIA:
            raise ValueError(
                f"convergence_criterion must be one of {VALID_CONVERGENCE_CRITERIA}, "
                f"got {convergence_criterion!r}."
            )
        valid_methods = ["pql_gcv", "pql_reml", "laplace_reml"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        _validate_eval_bases_have_bounds(basis)
        self.basis = basis
        self.method = method
        self.method_kwargs = method_kwargs or {}
        self.observation_model = observation_model

        # Laplace-REML currently supports only Poisson: phi is fixed at 1 there.
        # The empirical phi-scaling test (_script/check_phi_scaling.py) shows
        # rho-hat does NOT factor from phi for the Laplace objective, so non-
        # Poisson families need a joint rho/phi outer iteration not yet built.
        if method == "laplace_reml" and not isinstance(
            observation_model, PoissonObservations
        ):
            raise NotImplementedError(
                'method="laplace_reml" currently supports only '
                "PoissonObservations (phi fixed at 1). Gaussian/Gamma require a "
                "joint rho/phi outer iteration, which is not yet implemented."
            )

        self.variance_function = _make_variance_function(self.observation_model)
        self.maxiter = maxiter
        self.tol_update = tol_update
        self.tol_optim = tol_optim
        self.use_scipy = use_scipy
        self.convergence_criterion = convergence_criterion
        self.drop_conv_basis_col = drop_conv_basis_col
        self.n_simpson_sample = int(1e4)

        self._positive_mon_func_for_lambda = jnp.exp
        # Identifiability is applied per basis component to match how the design matrix is built:
        # BSplineConv leaves follow ``drop_conv_basis_col``; other leaves drop the last column.
        self._apply_identifiability_column = tuple(
            _make_identifiability_dropper(
                b,
                square=False,
                drop_conv_basis_col=self.drop_conv_basis_col,
            )
            for b in self.basis
        )
        self._apply_identifiability_square = tuple(
            _make_identifiability_dropper(
                b,
                square=True,
                drop_conv_basis_col=self.drop_conv_basis_col,
            )
            for b in self.basis
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

    def _build_penalty_handler(self, penalty_tree: list) -> PenaltyHandler:
        """Construct a PenaltyHandler from the penalty tensor list."""
        ph = PenaltyHandler(non_linearity=self._positive_mon_func_for_lambda)
        id_fns = [
            _drop_last_col if _should_drop_basis_col(b, self.drop_conv_basis_col) else _id_no_drop
            for b in self.basis
        ]
        for S_tensor, id_fn in zip(penalty_tree, id_fns):
            # null-space penalty already incorporated into S_tensor by compute_energy_penalty_tensor
            ph.add(S_tensor, penalize_null_space=False, identifiability_fn=id_fn)
        return ph

    def _make_inner_func(self, penalty_tree, compute_sqrt, compute_log_det_and_grad):
        """Build the PQL inner-loop smoothing-parameter objective for ``method``."""
        if self.method == "pql_gcv":
            return gcv_compute_factory(
                compute_sqrt,
                self._positive_mon_func_for_lambda,
                self._apply_identifiability_column,
                self._apply_identifiability_square,
                1.5,
            )
        if self.method == "pql_reml":
            return reml_compute_factory(
                compute_sqrt,
                compute_log_det_and_grad,
                self._positive_mon_func_for_lambda,
                self._apply_identifiability_column,
                self._apply_identifiability_square,
            )

        raise ValueError(
            f"_make_inner_func handles only the PQL methods; got method={self.method!r}"
        )

    def _fit_laplace_reml(
        self,
        X,
        y,
        penalty_tree,
        compute_sqrt,
        compute_log_det_and_grad,
        init_params,
        init_regularizer_strength,
    ):
        """Direct Laplace-REML fit: optimise rho via the Laplace-approximated REML.

        Mirrors the output contract of ``pql_outer_iteration`` — returns
        ``((coef, intercept), regularizer_strength, n_iter)`` — so ``fit`` can
        treat both paths uniformly.

        Unlike the PQL path (which alternates IRLS coefficient updates with a
        linearised smoothing-parameter step), here the outer optimiser drives
        rho directly over the Laplace-approximated REML, re-fitting beta_hat to
        the MAP at each rho evaluation.
        """
        n_obs = X.shape[0]
        # laplace_reml works on the full design [intercept | smooths] and the
        # full coefficient vector; the PQL path keeps the intercept separate.
        X_full = jnp.column_stack((jnp.ones(n_obs), X))

        # S_all: (M, P, P) penalty stack in the full coef space, intercept
        # row/col zero — same construction _pql_reml uses for its gradient.
        penalty_blocks = compute_penalty_blocks(
            penalty_tree,
            apply_identifiability=self._apply_identifiability_square,
            shift_by=1,
        )
        S_all = jnp.concatenate(jax.tree_util.tree_leaves(penalty_blocks), axis=0)

        coef0, intercept0 = init_params
        init_beta = jnp.concatenate([jnp.atleast_1d(intercept0), coef0])

        # M_null: _get_penalty_tree penalises every smooth's null space
        # (penalize_null_space=True), so the only unpenalised direction is the
        # intercept — M_null is 1, analytically, with no rank estimation.
        M_null = 1

        # phi = 1: Laplace-REML is Poisson-only here (enforced at construction).
        inner_solve = make_inner_solver(
            self.method_kwargs.get("inner_solver", "LBFGS"),
            self.method_kwargs.get("inner_solver_kwargs"),
        )
        rhos_opt, beta_opt, n_iter = laplace_reml_outer_iteration(
            init_regularizer_strength,
            init_beta,
            X_full,
            y,
            self.observation_model,
            self.observation_model.default_inverse_link_function,
            S_all,
            1.0,  # phi
            M_null,
            compute_sqrt,
            compute_log_det_and_grad,
            inner_solve=inner_solve,
            outer_solver_name=self.method_kwargs.get("outer_solver", "LBFGS"),
            outer_solver_kwargs=self.method_kwargs.get("outer_solver_kwargs"),
        )
        return (beta_opt[1:], beta_opt[:1]), rhos_opt, n_iter

    def _get_penalty_tree(self) -> list[jnp.ndarray]:
        """
        Compute the penalty tensor tree for all smooth terms.

        Delegatest to ``compute_energy_penalty_tensor``.
        """
        return compute_energy_penalty_tensor(
            self.basis, self.n_simpson_sample, penalize_null_space=True
        )

    def _compute_uncentered_design_matrix(
        self,
        inputs: tuple[ArrayLike, ...],
        setup_basis: bool,
    ) -> jnp.ndarray:
        """Build the identifiability-constrained design matrix before centering."""
        if setup_basis:
            X = compute_features_identifiable(
                self.basis,
                *inputs,
                drop_conv_basis_col=self.drop_conv_basis_col,
            )
        else:
            X = _compute_features_identifiable(
                self.basis,
                *inputs,
                drop_conv_basis_col=self.drop_conv_basis_col,
            )
        X = jnp.array(X)

        # TODO: Drop rows with NaNs instead?
        # original PGAM zeros out nans, but nemos drops rows with NaNs
        return jnp.where(jnp.isnan(X), 0.0, X)

    def _fit_design_matrix(self, inputs: tuple[ArrayLike, ...]) -> jnp.ndarray:
        """
        Build the training design matrix and cache its column means.

        This is the only design-matrix path that calls ``basis.setup_basis``.
        Prediction must reuse this fitted basis state and centering.
        """
        X = self._compute_uncentered_design_matrix(inputs, setup_basis=True)
        self.feature_mean_ = X.mean(axis=0)
        return X - self.feature_mean_

    def _transform_design_matrix(self, inputs: tuple[ArrayLike, ...]) -> jnp.ndarray:
        """
        Build a prediction/evaluation design matrix using fitted centering.

        This intentionally does not call ``basis.setup_basis``: predict should
        transform new inputs with the basis state learned during fit.
        """
        if not hasattr(self, "feature_mean_"):
            raise AttributeError(
                "GAM instance is not fitted yet. Call fit before predict."
            )
        X = self._compute_uncentered_design_matrix(inputs, setup_basis=False)
        if X.shape[1] != self.feature_mean_.shape[0]:
            raise ValueError(
                "Prediction design matrix has "
                f"{X.shape[1]} columns, but the fitted model expects "
                f"{self.feature_mean_.shape[0]} columns."
            )
        return X - self.feature_mean_

    def _init_regularizer_strength(
        self, penalty_tree: list[jnp.ndarray]
    ) -> list[jnp.ndarray]:
        """Initialize log-space regularizer strengths to zero (i.e. lambda=1) for each penalty component."""
        # in the example there was: regularizer_strength = [jnp.log(jnp.array([1.0, 2.0]))] * 2
        return [jnp.zeros(pen.shape[0]) for pen in penalty_tree]

    # TODO: Test against the original implementation
    def _compute_cov_beta_from_fit_state(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: tuple[jnp.ndarray, jnp.ndarray],
        regularizer_strength: list[jnp.ndarray],
        compute_sqrt,
        rtol: float = 1e-8,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute posterior covariance, EDF, and dispersion scale after fitting.

        This mirrors legacy PGAM's final refresh: recompute final IRLS weights
        from the returned coefficients, form QR(sqrt(W) X), then invert
        ``X.T W X + S_lambda`` via the SVD of ``[R; sqrt(S_lambda)]``.

        Returns
        -------
        cov_beta :
            Posterior covariance ``φ · (X'WX + S_λ)⁻¹``, shape ``(p+1, p+1)``.
        edf :
            Effective degrees of freedom — Wood's ``edf1 = 2·tr(F) − tr(F²)``.
        scale :
            Estimated dispersion φ̂ from ``observation_model.estimate_scale``.
        """
        coef, intercept = params
        n_obs = X.shape[0]
        eta = X @ coef + intercept
        mu = self.observation_model.default_inverse_link_function(eta)

        inverse_link = self.observation_model.default_inverse_link_function
        link_func = INVERSE_FUNCS[inverse_link]
        get_pseudo_data_and_weight = model_constructors_for_weights_and_pseudo_data(
            self.variance_function,
            link_func,
            fisher_scoring=False,
        )
        _, weights = get_pseudo_data_and_weight(y, mu)
        weights = jnp.reshape(weights, (-1,))

        if bool(jnp.any(weights < -1e-12)):
            raise ValueError("Final IRLS weights contain negative values.")
        weights = jnp.clip(weights, 0.0, jnp.inf)

        # Xw^T Xw = X^T W X
        X_full = jnp.column_stack((jnp.ones(n_obs), X))
        Xw = X_full * jnp.sqrt(weights)[:, None]
        R = jnp.linalg.qr(Xw, mode="r")

        sqrt_penalty = compute_sqrt(regularizer_strength)
        sqrt_penalty = jnp.column_stack(
            (jnp.zeros(sqrt_penalty.shape[0]), sqrt_penalty)
        )

        # SVD of A = [R; B],  A^T A = X^T W X + S_lambda
        # U1 = U[:k] (first k=R.shape[0] rows) encodes the hat matrix via A = Q_xw U1 U1' Q_xw'
        U_svd, singular_values, Vt = jnp.linalg.svd(
            jnp.vstack((R, sqrt_penalty)),
            full_matrices=False,
        )
        U1 = U_svd[: R.shape[0], :]  # (k, k) where k = p + 1

        # EDF: edf1 = 2·tr(F) − tr(F²) where F = (X'WX + S_λ)⁻¹ X'WX
        # Expressed via U1: tr(F) = ‖U1‖²_F,  tr(F²) = ‖U1'U1‖²_F  (Wood 2017 eq. 6.13)
        edf = jnp.sum(U1 ** 2)
        edf1 = 2.0 * edf - jnp.sum((U1.T @ U1) ** 2)

        # dispersion: Poisson → 1.0; Gaussian/Gamma → Pearson χ²/dof
        scale = self.observation_model.estimate_scale(y, mu, dof_resid=n_obs - edf1)

        # tiny singular values would blow up 1 / s^2, so discard them
        keep = singular_values >= rtol * singular_values.max()
        singular_values_inv = jnp.where(keep, 1.0 / singular_values, 0.0)
        cov_beta = scale * (Vt.T * singular_values_inv**2) @ Vt
        return cov_beta, edf1, scale

    def _resolve_basis_component(
        self,
        component: int | str,
    ) -> BasisComponentInfo:
        """Resolve a component index or basis label to component metadata."""
        infos = _get_basis_component_infos(
            self.basis,
            drop_conv_basis_col=self.drop_conv_basis_col,
        )
        if isinstance(component, str):
            for info in infos:
                if info.basis.label == component:
                    return info

            available_labels = ", ".join(repr(info.basis.label) for info in infos)
            raise ValueError(
                f"No smooth component with label {component!r} was found. "
                f"Available labels: {available_labels}."
            )

        # if component is an index
        return infos[component]

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

        X = self._fit_design_matrix(xi)
        penalty_tree = self._get_penalty_tree()
        y = jnp.asarray(y)

        ph = self._build_penalty_handler(penalty_tree)
        compute_sqrt, compute_log_det_and_grad = ph.build()

        # TODO: Pull out the GLM initialization here?
        if init_params is None:
            init_params = self.initialize_params(X, y)

        if init_regularizer_strength is None:
            init_regularizer_strength = self._init_regularizer_strength(penalty_tree)

        if self.method == "laplace_reml":
            opt_coef, opt_pen, n_iter = self._fit_laplace_reml(
                X, y, penalty_tree, compute_sqrt, compute_log_det_and_grad,
                init_params, init_regularizer_strength,
            )
        else:
            opt_coef, opt_pen, n_iter = pql_outer_iteration(
                init_regularizer_strength,  # initial log(λ) for each smooth
                init_params,  # initial (coefficients, intercept)
                X,
                y,
                penalty_tree,
                self.observation_model,
                self.variance_function,
                self._make_inner_func(
                    penalty_tree, compute_sqrt, compute_log_det_and_grad
                ),
                compute_sqrt,
                fisher_scoring=False,  # use observed information, not expected
                max_iter=self.maxiter,
                tol_update=self.tol_update,  # convergence tol for coefficient updates
                tol_optim=self.tol_optim,  # tolerance for inner GCV optimization
                use_scipy=self.use_scipy,
                convergence_criterion=self.convergence_criterion,
            )

        self.coef_, self.intercept_ = opt_coef
        self.regularizer_strength_ = opt_pen
        self.n_iter_ = n_iter
        self.cov_beta_, self.edf_, self.scale_ = self._compute_cov_beta_from_fit_state(
            X,
            y,
            opt_coef,
            opt_pen,
            compute_sqrt,
        )
        self.dof_resid_ = X.shape[0] - self.edf_

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
            self._transform_design_matrix(xi) @ w + b
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

    # TODO: Test against original implementation
    def smooth_compute(
        self,
        xi: tuple[ArrayLike, ...],
        component_index: int | str,
        perc: float = 0.95,
        se_with_mean: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evaluate one grid-centered smooth and confidence band.

        This mirrors legacy PGAM's ``smooth_compute`` display convention: the
        selected component is evaluated with the fitted basis state, centered
        on the supplied grid, and combined with the matching coefficient block.
        It is intentionally separate from ``predict``, which uses training
        centering.
        """
        if not hasattr(self, "cov_beta_"):
            raise AttributeError("GAM instance is not fitted yet. Call fit first.")
        if not isinstance(xi, tuple):
            xi = (xi,)

        info = self._resolve_basis_component(component_index)

        if len(xi) != info.input_slice.stop - info.input_slice.start:
            raise ValueError(
                f"component_index {component_index} expects "
                f"{info.input_slice.stop - info.input_slice.start} input array(s), "
                f"got {len(xi)}."
            )
        fX = _compute_features_identifiable(
            info.basis,
            *xi,
            drop_conv_basis_col=self.drop_conv_basis_col,
        )
        fX = jnp.asarray(fX)

        nan_filter = jnp.asarray(
            jnp.sum(jnp.isnan(jnp.asarray(xi)), axis=0),
            dtype=bool,
        )
        nan_filter = nan_filter | jnp.any(jnp.isnan(fX), axis=1)
        if not bool(jnp.any(~nan_filter)):
            raise ValueError(
                "No valid rows remain after evaluating the selected smooth. "
                "For convolutional smooths, pass an input longer than the "
                "basis window so at least one row is not NaN-padded."
            )

        center = jnp.mean(fX[~nan_filter], axis=0)

        fX = jnp.where(jnp.isnan(fX), 0.0, fX)
        fX = fX - center
        fX = fX.at[nan_filter, :].set(0.0)

        mean_y = fX @ self.coef_[info.identifiable_feature_slice]

        coef_idx = (
            jnp.arange(
                info.identifiable_feature_slice.start,
                info.identifiable_feature_slice.stop,
            )
            + 1
        )
        if se_with_mean:
            X_se = jnp.column_stack((jnp.ones(fX.shape[0]), fX))
            cov_idx = jnp.concatenate((jnp.array([0]), coef_idx))
        else:
            X_se = fX
            cov_idx = coef_idx

        cov = self.cov_beta_[cov_idx[:, None], cov_idx[None, :]]
        sigma2 = jnp.sum((X_se @ cov) * X_se, axis=1)
        se_y = jnp.sqrt(jnp.clip(sigma2, 0.0, jnp.inf))
        delta = se_y * sts.norm().ppf(1 - (1 - perc) * 0.5)
        return mean_y, mean_y - delta, mean_y + delta

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
            scale=self.scale_,
            aggregate_sample_scores=aggregate_sample_scores,
        )
