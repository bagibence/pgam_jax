import jax
import numpy as np
from nemos.observation_models import PoissonObservations


import pgam_clean.penalty_utils as pen_utils
from pgam_clean.basis import GAMBSplineEval
from pgam_clean.gcv_compute import gcv_compute_factory
from pgam_clean.iterative_optim import pql_outer_iteration

from PGAM.GAM_library import *
import statsmodels.api as sm

import jax.tree_util as jtu
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)
bas = GAMBSplineEval(12, identifiability=False)
add = bas + bas
penalty_tree = pen_utils.compute_energy_penalty_tensor(
    add, 10**4, penalize_null_space=True)

np.random.seed(123)
# define a model matrix without intercept
x1, x2 = np.random.randn(3000), np.random.randn(3000)

sm_handler = smooths_handler()
knots = np.linspace(0,1, 10)
sm_handler.add_smooth("linspace", [x1], knots=[knots], ord=4, is_temporal_kernel=False,
                          trial_idx=None, is_cyclic=[False], penalty_type="der", der=2, lam=[1,2],
                          knots_num=10, kernel_length=None, kernel_direction=None,
                          time_bin=0.006,knots_percentiles=(0,100))

sm_handler.add_smooth("linspace2", [x2], knots=[knots], ord=4, is_temporal_kernel=False,
                          trial_idx=None, is_cyclic=[False], penalty_type="der", der=2, lam=[1,2],
                          knots_num=10, kernel_length=None, kernel_direction=None,
                          time_bin=0.006,knots_percentiles=(0,100))



X =  jnp.asarray(sm_handler.get_exog_mat_fast(sm_handler.smooths_var)[0])[:,1:]
w = np.hstack([np.random.randn(11), np.zeros(11)])
inerc = np.array([0.])
y = np.random.poisson(np.exp(X.dot(w)))
inv_link_func = jax.numpy.exp
variance_func = lambda x: x
compute_sqrt_penalty = lambda *args: pen_utils.tree_compute_sqrt_penalty(
    *args, shift_by=0, positive_mon_func=jax.numpy.exp, apply_identifiability=lambda x: x[...,:-1]
)

inner_func = gcv_compute_factory(
    jax.numpy.exp, lambda x: x[...,:-1], lambda x: x[...,:-1, :-1], 1.5
)



link = sm.genmod.families.links.log()
poissFam = sm.genmod.families.family.Poisson(link=link)

# create the pgam model
pgam = general_additive_model(sm_handler,
                              sm_handler.smooths_var, # list of covariate we want to include in the model
                              np.asarray(y), # vector of spike counts
                              poissFam # poisson family with exponential link from statsmodels.api
                             )

res = pgam.optim_gam(sm_handler.smooths_var,
                                          max_iter=10 ** 2, # max number of iteration
                                          use_dgcv=True,
                                          method="L-BFGS-B",
                                          fit_initial_beta=True,
                                          filter_trials=np.ones(len(y), dtype=bool))

opt_coef, opt_pen, niter = pql_outer_iteration(
        [jax.numpy.log(jax.numpy.array([1., 2.]))]*2,
        jtu.tree_map(jnp.zeros_like, (w, inerc)),
        X,
        y,
        penalty_tree,
        PoissonObservations(),
        variance_func,
        inner_func,
        compute_sqrt_penalty,
        fisher_scoring=False,
        max_iter=100,
        tol_update=10**-6,tol_optim=10**-10
)

from nemos.glm import GLM
model = GLM().fit(X, y)
print(niter)
import matplotlib.pyplot as plt
plt.close('all')
f, axs = plt.subplots(1, 4, figsize=(10,3), sharey=True, sharex=True)
axs[0].set_xlabel("orig gam")
axs[0].set_ylabel("jax gam")
axs[0].scatter(res.beta, np.hstack((opt_coef[1], opt_coef[0])))
axs[0].plot([-3,3],[-3,3], "k")
axs[0].set_aspect("equal")

axs[1].scatter(np.hstack((inerc, w)), res.beta)
axs[1].plot([-3,3],[-3,3], "k")
axs[1].set_xlabel("true")
axs[1].set_ylabel("orig gam")

axs[2].scatter(np.hstack((inerc, w)), np.hstack((opt_coef[1], opt_coef[0])))
axs[2].plot([-3,3],[-3,3], "k")
axs[2].set_xlabel("true")
axs[2].set_ylabel("jax gam")

axs[3].scatter(np.hstack((inerc, w)), np.hstack((model.intercept_, model.coef_)), color="orange")
axs[3].plot([-3,3],[-3,3], "k")
axs[3].set_xlabel("true")
axs[3].set_ylabel("GLM")
plt.tight_layout()
plt.show()

plt.figure()

plt.plot(res.beta[1:], label="orig-gam")
plt.plot(opt_coef[0], ls="--", label="jax-gam")
plt.plot(model.coef_, label="glm")
plt.scatter(np.arange(len(model.coef_)), w, label="true")
plt.legend()
plt.show()