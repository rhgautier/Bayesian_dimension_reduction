from copy import deepcopy
import datetime
from pathlib import Path
import time

import h5py
import jax
from jax import jit, numpy as jnp
from jax.ops import index, index_update
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
import numpy as np
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from pymanopt import tools
from pymanopt.solvers.linesearch import LineSearchAdaptive
from pymanopt.solvers.solver import Solver
import pandas as pd
from scipy.linalg import svd, subspace_angles

#################################
# Householder Reparametrization #
#################################


def householder(v, n):
    v = v[:, None]
    k = v.shape[0]
    sgn = jnp.sign(v[0])
    u = v + sgn * jnp.linalg.norm(v) * jnp.eye(k, 1)
    u = u / jnp.linalg.norm(u)
    h_k = -sgn * (jnp.eye(k) - 2 * u @ u.T)
    h_k_hat = jnp.eye(n)
    start = n - k
    h_k_hat = index_update(h_k_hat, index[start:, start:], h_k)
    return h_k_hat


def householder_reparameterization(
    projection_parameters, num_original_inputs, num_active_dims
):
    m, n = num_active_dims, num_original_inputs
    h = jnp.eye(n)
    i_end = 0
    for i in range(m):
        i_start = i_end
        i_end = i_start + n - i
        h_k_hat = householder(projection_parameters[i_start:i_end], n)
        h = h_k_hat @ h
    return h[:, :m]


#################
# Error Metrics #
#################


@jit
def rmse_nrmse_r_squared_and_absolute_error(y_predicted, y_actual):
    var_y_actual = jnp.var(y_actual)
    absolute_error = y_predicted.reshape(-1) - y_actual.reshape(-1)
    mse = jnp.mean(jnp.square(absolute_error))
    r_squared = 1 - mse / var_y_actual
    rmse = jnp.sqrt(mse)
    nrmse = rmse / jnp.sqrt(var_y_actual)
    return rmse, nrmse, r_squared, absolute_error


@jit
def normalized_posterior_log_likelihood(y_means, y_variances, y_actual):
    pointwise_pll = logsumexp(
        dist.Normal(loc=y_means, scale=jnp.sqrt(y_variances)).log_prob(y_actual),
        axis=1,
    ) - jnp.log(y_means.shape[1])
    npll = jnp.mean(pointwise_pll, axis=0)
    return npll, pointwise_pll


####################
# Subspace Metrics #
####################


def compute_subspace_metrics(
    training_data,
    validation_data,
    training_parameters,
    training_artifacts,
    get_w_draws,
    dy_dx,
):
    # Retrieve samples of W from the method
    w_draws = get_w_draws(
        training_data, validation_data, training_parameters, training_artifacts
    )

    # Problem dimensions
    num_posterior_draws = w_draws.shape[0]
    dim_feature_space = w_draws.shape[2]

    # Compute the true AS
    true_w, _, _ = svd(np.transpose(dy_dx), full_matrices=False)
    true_w = true_w[:, :dim_feature_space]

    # Compute the subspace angles
    all_subspace_angles = np.empty((num_posterior_draws, dim_feature_space))
    for i in range(num_posterior_draws):
        all_subspace_angles[i] = subspace_angles(true_w, w_draws[i],)
    first_subspace_angle = all_subspace_angles[:, 0]
    mean_fsa = np.mean(first_subspace_angle)
    std_fsa = np.std(first_subspace_angle)

    return {
        "subspace_angles": all_subspace_angles,
        "first_subspace_angle": first_subspace_angle,
        "mean_fsa": mean_fsa,
        "std_fsa": std_fsa,
    }


######################
# Validation Metrics #
######################


@jit
def compute_validation_metrics(
    predicted_samples,
    predicted_means,
    predicted_variances,
    actual,
    validation_parameters,
):
    quantile_values = jnp.array(validation_parameters["quantile_values"])

    quantiles = jnp.quantile(predicted_samples, quantile_values, axis=1)

    (
        rmse,
        nrmse,
        r_squared,
        absolute_pointwise_error,
    ) = rmse_nrmse_r_squared_and_absolute_error(quantiles[1], actual)

    npll, pointwise_pll = normalized_posterior_log_likelihood(
        predicted_means, predicted_variances, actual
    )

    return (
        {"rmse": rmse, "nrmse": nrmse, "r_squared": r_squared, "npll": npll},
        {
            "quantiles": quantiles,
            "absolute_pointwise_error": absolute_pointwise_error,
            "pointwise_pll": pointwise_pll,
        },
    )


#################
# Normalization #
#################


def compute_normalization(x):
    return np.mean(x), np.std(x)


def normalize(x, offset, scaling):
    return (x - offset) / scaling


def denormalize(x, offset, scaling):
    return offset + x * scaling


########
# MCMC #
########


def mcmc(
    numpyro_model,
    model_arguments,
    target_acceptance_probability=0.8,
    num_chains=1,
    chain_method="parallel",
    num_warmup_draws=500,
    num_posterior_draws=1000,
    random_seed=0,
    progress_bar=True,
    display_summary=True,
    return_mcmc_sampler=False,
):
    mcmc_kernel = NUTS(numpyro_model, target_accept_prob=target_acceptance_probability)
    mcmc_sampler = MCMC(
        mcmc_kernel,
        num_warmup_draws,
        num_posterior_draws,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc_sampler.run(PRNGKey(random_seed), *model_arguments)

    if display_summary:
        mcmc_sampler.print_summary(exclude_deterministic=False)

    if not return_mcmc_sampler:
        return mcmc_sampler.get_samples(group_by_chain=False)
    else:
        return mcmc_sampler.get_samples(group_by_chain=False), mcmc_sampler


##################################
# Post-Processing of MCMC chains #
##################################


def process_chains(
    posterior_draws, grouped_by_chain, ungroup=False, num_thinned_draws=None
):
    # Alternate dictionary holding the newly processed chains
    processed_draws = {}

    for site_name, site_draws in posterior_draws.items():
        # Whether the current site draws is grouped by chain
        site_draws_grouped_by_chain = grouped_by_chain

        # Flatten if necessary
        if site_draws_grouped_by_chain and ungroup:
            processed_draws[site_name] = jnp.reshape(
                site_draws, (-1,) + site_draws.shape[2:]
            )
            site_draws_grouped_by_chain = False
        else:
            processed_draws[site_name] = site_draws

        # Thin if necessary
        if num_thinned_draws is not None:
            chain_length = (
                processed_draws[site_name].shape[1]
                if site_draws_grouped_by_chain
                else processed_draws[site_name].shape[0]
            )
            if num_thinned_draws < chain_length:
                thinning_factor = chain_length // num_thinned_draws
                last_index = thinning_factor * num_thinned_draws

                if site_draws_grouped_by_chain:
                    processed_draws[site_name] = processed_draws[site_name][
                        :, :last_index:thinning_factor
                    ]
                else:
                    processed_draws[site_name] = processed_draws[site_name][
                        :last_index:thinning_factor
                    ]

    return processed_draws


#########################################
# Custom CG implementation for PyManOpt #
#########################################

# TODO: Use Python's enum module.
BetaTypes = tools.make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)


class CustomConjugateGradient(Solver):
    """
    Module containing conjugate gradient algorithm based on
    conjugategradient.m from the manopt MATLAB package.
    """

    def __init__(
        self,
        beta_type=BetaTypes.HestenesStiefel,
        orth_value=np.inf,
        linesearch=None,
        use_cost_improvement_criterion=True,
        cost_improvement_threshold=1e-3,
        no_cost_improvement_streak=10,
        *args,
        **kwargs,
    ):
        """
        Instantiate gradient solver class.
        Variable attributes (defaults in brackets):
            - beta_type (BetaTypes.HestenesStiefel)
                Conjugate gradient beta rule used to construct the new search
                direction
            - orth_value (numpy.inf)
                Parameter for Powell's restart strategy. An infinite
                value disables this strategy. See in code formula for
                the specific criterion used.
            - linesearch (LineSearchAdaptive)
                The linesearch method to used.
        """
        super().__init__(*args, **kwargs)

        self._beta_type = beta_type
        self._orth_value = orth_value

        self.use_cost_improvement_criterion = use_cost_improvement_criterion
        self.cost_improvement_threshold = cost_improvement_threshold
        self.no_cost_improvement_streak = no_cost_improvement_streak
        self.old_cost = None
        self.cost_improvement_counter = 0

        if linesearch is None:
            self._linesearch = LineSearchAdaptive()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    def check_cost_improvement(self, cost, time0):
        if not self.use_cost_improvement_criterion:
            return None

        if self.old_cost is None:
            cost_improvement = float("inf")
        else:
            cost_improvement = (self.old_cost - cost) / self.old_cost
        self.old_cost = cost
        if cost_improvement < self.cost_improvement_threshold:
            self.cost_improvement_counter += 1
        else:
            self.cost_improvement_counter = 0

        if self.cost_improvement_counter >= self.no_cost_improvement_streak:
            self.cost_improvement_counter = 0
            return (
                f"Terminated - no cost improvement greater than "
                "{self.cost_improvement_threshold}% for more than "
                "{self.no_cost_improvement_streak} consecutive iterations."
                f"{(time.time() - time0):.2f} seconds."
            )
        return None

    def solve(self, problem, x=None, reuselinesearch=False):
        """
        Perform optimization using nonlinear conjugate gradient method with
        linesearch.
        This method first computes the gradient of obj w.r.t. arg, and then
        optimizes by moving in a direction that is conjugate to all previous
        search directions.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - reuselinesearch=False
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        stepsize = np.nan
        time0 = time.time()

        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")

        # Calculate initial cost-related quantities
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
        Pgrad = problem.precon(x, grad)
        gradPgrad = man.inner(x, grad, Pgrad)

        # Initial descent direction is the negative gradient
        desc_dir = -Pgrad

        self._start_optlog(
            extraiterfields=["gradnorm"],
            solverparams={
                "beta_type": self._beta_type,
                "orth_value": self._orth_value,
                "linesearcher": linesearch,
            },
        )

        while True:
            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            stop_reason = self._check_stopping_criterion(
                time0, gradnorm=gradnorm, iter=iter + 1, stepsize=stepsize
            )
            stop_reason = stop_reason
            # stop_reason = stop_reason or self.check_cost_improvement(cost, time0)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print("")
                break

            # The line search algorithms require the directional derivative of
            # the cost at the current point x along the search direction.
            df0 = man.inner(x, grad, desc_dir)

            # If we didn't get a descent direction: restart, i.e., switch to
            # the negative gradient. Equivalent to resetting the CG direction
            # to a steepest descent step, which discards the past information.
            if df0 >= 0:
                # Or we switch to the negative gradient direction.
                if verbosity >= 3:
                    print(
                        "Conjugate gradient info: got an ascent direction "
                        "(df0 = %.2f), reset to the (preconditioned) "
                        "steepest descent direction." % df0
                    )
                # Reset to negative gradient: this discards the CG memory.
                desc_dir = -Pgrad
                df0 = -gradPgrad

            # Execute line search
            stepsize, newx = linesearch.search(objective, man, x, desc_dir, cost, df0)

            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradnorm = man.norm(newx, newgrad)
            Pnewgrad = problem.precon(newx, newgrad)
            newgradPnewgrad = man.inner(newx, newgrad, Pnewgrad)

            # Apply the CG scheme to compute the next search direction
            oldgrad = man.transp(x, newx, grad)
            orth_grads = man.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad

            # Powell's restart strategy (see page 12 of Hager and Zhang's
            # survey on conjugate gradient methods, for example)
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                desc_dir = -Pnewgrad
            else:
                desc_dir = man.transp(x, newx, desc_dir)

                if self._beta_type == BetaTypes.FletcherReeves:
                    beta = newgradPnewgrad / gradPgrad
                elif self._beta_type == BetaTypes.PolakRibiere:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    beta = max(0, ip_diff / gradPgrad)
                elif self._beta_type == BetaTypes.HestenesStiefel:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    try:
                        beta = max(0, ip_diff / man.inner(newx, diff, desc_dir))
                    # if ip_diff = man.inner(newx, diff, desc_dir) = 0
                    except ZeroDivisionError:
                        beta = 1
                elif self._beta_type == BetaTypes.HagerZhang:
                    diff = newgrad - oldgrad
                    Poldgrad = man.transp(x, newx, Pgrad)
                    Pdiff = Pnewgrad - Poldgrad
                    deno = man.inner(newx, diff, desc_dir)
                    numo = man.inner(newx, diff, Pnewgrad)
                    numo -= (
                        2
                        * man.inner(newx, diff, Pdiff)
                        * man.inner(newx, desc_dir, newgrad)
                        / deno
                    )
                    beta = numo / deno
                    # Robustness (see Hager-Zhang paper mentioned above)
                    desc_dir_norm = man.norm(newx, desc_dir)
                    eta_HZ = -1 / (desc_dir_norm * min(0.01, gradnorm))
                    beta = max(beta, eta_HZ)
                else:
                    types = ", ".join(["BetaTypes.%s" % t for t in BetaTypes._fields])
                    raise ValueError(
                        "Unknown beta_type %s. Should be one of %s."
                        % (self._beta_type, types)
                    )

                desc_dir = -Pnewgrad + beta * desc_dir

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradnorm = newgradnorm
            gradPgrad = newgradPnewgrad

            iter += 1

        if self._logverbosity <= 0:
            return x, cost
        else:
            self._stop_optlog(
                x,
                cost,
                stop_reason,
                time0,
                stepsize=stepsize,
                gradnorm=gradnorm,
                iter=iter,
            )
            return x, self._optlog


##################################
# Persistence of the Run Results #
##################################


def compact_timestamp():
    return "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


def dict_to_group(_dict: dict, _group: h5py.Group):
    for key, value in _dict.items():
        if isinstance(value, dict):
            subgroup = _group.create_group(key)
            dict_to_group(value, subgroup)
        else:
            if isinstance(value, jax.interpreters.xla.DeviceArray):
                value = np.array(value)

            if isinstance(value, np.ndarray):
                if value.size == 1:
                    _group.attrs[key] = value.flatten()[0]
                else:
                    _group.create_dataset(key, data=value)
            else:
                _group.attrs[key] = value


def save_case_results(case_paramaters, training_artifacts, validation_artifacts):
    """ Persist all case inputs, along with training and validation artifacts. """

    # Directory in which outputs are stored
    out_dir = Path("results") / case_paramaters["dataset"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Append timestamp if the case already exists
    out_path = out_dir / f"{case_paramaters['name']}.h5"
    if out_path.is_file():
        out_path = out_dir / f"{case_paramaters['name']}_{compact_timestamp()}.h5"

    # Open and write the file
    with h5py.File(out_path, "w-") as case_results_file:
        dict_to_group(
            {
                "case_inputs": case_paramaters,
                "training_artifacts": training_artifacts,
                "validation_artifacts": validation_artifacts,
            },
            case_results_file,
        )


###############################
# Generic Validation Routines #
###############################


def validate_one_subset(
    data_to_validate,
    training_data,
    training_artifacts,
    prediction_routine,
    prediction_parameters,
    validation_parameters,
):
    samples, means, variances = prediction_routine(
        data_to_validate["x"], training_data, training_artifacts, prediction_parameters,
    )
    metrics, pointwise_quantities = compute_validation_metrics(
        samples, means, variances, data_to_validate["y"], validation_parameters
    )
    return {
        "samples": samples,
        "means": means,
        "variances": variances,
        "metrics": metrics,
        "pointwise_quantities": pointwise_quantities,
    }


def generic_validation_routine(
    training_data,
    training_parameters,
    training_artifacts,
    prediction_routine,
    prediction_parameters,
    validation_data,
    validation_parameters,
    get_w_draws,
    dy_dx,
):
    return {
        "training": validate_one_subset(
            training_data,
            training_data,
            training_artifacts,
            prediction_routine,
            prediction_parameters,
            validation_parameters,
        ),
        "validation": validate_one_subset(
            validation_data,
            training_data,
            training_artifacts,
            prediction_routine,
            prediction_parameters,
            validation_parameters,
        ),
        "subspace_metrics": compute_subspace_metrics(
            training_data,
            validation_data,
            training_parameters,
            training_artifacts,
            get_w_draws,
            dy_dx,
        ),
    }


###################
# Dataset Loading #
###################


def load_dataset(dataset_info):
    dataframe = pd.read_csv(f"data/{dataset_info['name']}.csv", header=0, index_col=0)
    input_names = dataframe.columns[: dataset_info["num_inputs"]]
    output_name = dataset_info["output_name"]
    x = dataframe[input_names].to_numpy()
    y = dataframe[output_name].to_numpy().reshape((-1, 1))
    dy_dx = dataframe[
        [f"d({output_name})_d({input_name})" for input_name in input_names]
    ].to_numpy()
    return x, y, dy_dx
