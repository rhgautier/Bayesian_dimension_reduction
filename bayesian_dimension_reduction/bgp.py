from functools import partial
import time

from jax import jit, vmap, numpy as jnp
from jax.lax import dynamic_slice_in_dim, map as jmap
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular, eigh
from jax.random import PRNGKey
from numpyro import enable_x64, sample, distributions as dist

from .utils import compute_normalization, normalize, denormalize, mcmc, process_chains

enable_x64()

###################################
# Numpyro model used for training #
###################################


def squared_distance(x1, x2):
    return (
        jnp.sum(x1 ** 2, axis=1, keepdims=True)
        - 2.0 * x1 @ x2.T
        + jnp.sum(x2 ** 2, axis=1, keepdims=True).T
    )


def se_kernel(x1, x2, signal_variance, length_scales):
    return signal_variance * jnp.exp(
        -0.5 * squared_distance(x1 / length_scales, x2 / length_scales)
    )


def gp_kernel(x_1, x_2, model_parameters):
    """ Kernel for the GP. """
    return se_kernel(
        x_1,
        x_2,
        model_parameters["signal_variance"],
        model_parameters["length_scales"],
    )


def bgp_model(training_data, jitter=1e-8):
    # Extract training data
    obs_x = training_data["x"]
    obs_y = training_data["y"]

    # Extract problem dimensions
    num_samples = obs_x.shape[0]
    dim_inputs = obs_x.shape[1]

    # Priors
    model_parameters = {
        "signal_variance": sample("signal_variance", dist.LogNormal(0, 1)),
        "length_scales": sample(
            "length_scales",
            dist.LogNormal(jnp.zeros((dim_inputs,)), jnp.ones((dim_inputs,))),
        ),
        "noise_variance": sample("noise_variance", dist.LogNormal(1)),
    }

    # GP
    k_x_x = (
        gp_kernel(obs_x, obs_x, model_parameters)
        + model_parameters["noise_variance"] * jnp.eye(num_samples)
        + jitter * jnp.eye(num_samples)
    )
    lower_chol_k_x_x, _ = cho_factor(k_x_x, lower=True)
    sample(
        "obs_y",
        dist.MultivariateNormal(jnp.zeros((num_samples,)), scale_tril=lower_chol_k_x_x),
        obs=obs_y.flatten(),
    )


#######################
# Prediction Routines #
#######################


@partial(vmap, in_axes=(0, None, None, None, None), out_axes=(0, 0))
def bgp_predict_inner2(pred_x, posterior_draw, training_data, c_and_lower, alpha):
    """ Compute mean and variance of the posterior LF GP given a *list* of prediction
    sites `pred_x` and a *single* posterior draw `posterior_draw'. """
    pred_x = pred_x[None, :]
    k_star = gp_kernel(training_data["x"], pred_x, posterior_draw)
    v = solve_triangular(c_and_lower[0], k_star, lower=c_and_lower[1])
    mean = (k_star.T @ alpha)[0, 0]
    variance = (gp_kernel(pred_x, pred_x, posterior_draw) - v.T @ v)[0, 0]
    return mean, variance


def bgp_predict_inner1(pred_x, posterior_draw, training_data, jitter=1e-8):
    """ Compute mean and variance of the posterior GP given a *list* of prediction
    sites `pred_x` and a single posterior draw `posterior_draw'. """
    k_x_x = (
        gp_kernel(training_data["x"], training_data["x"], posterior_draw)
        + posterior_draw["noise_variance"] * jnp.eye(training_data["x"].shape[0])
        + jitter * jnp.eye(training_data["x"].shape[0])
    )
    c_and_lower = cho_factor(k_x_x, lower=True)
    alpha = cho_solve(c_and_lower, training_data["y"])
    return bgp_predict_inner2(pred_x, posterior_draw, training_data, c_and_lower, alpha)


bgp_predict = jit(vmap(bgp_predict_inner1, in_axes=(None, 0, None), out_axes=(1, 1)))

######################################################
# High-Level Training/Prediction/Validation Routines #
######################################################


def train_bgp_model(training_data, training_parameters):
    # Extract parameters
    mcmc_params = training_parameters["mcmc_params"]

    # Normalize outputs and store normalization constants
    y_offset, y_scaling = compute_normalization(training_data["y"])

    normalization_constants = {"y_offset": y_offset, "y_scaling": y_scaling}

    # Normalize training data
    normalized_training_data = {
        "x": training_data["x"],
        "y": normalize(training_data["y"], y_offset, y_scaling),
    }

    # Measure wall-clock time
    start_time = time.time()

    posterior_draws = mcmc(bgp_model, (normalized_training_data,), **mcmc_params,)

    training_duration = time.time() - start_time

    return {
        "posterior_draws": posterior_draws,
        "normalization_constants": normalization_constants,
        "training_duration": training_duration,
    }


def predict_bgp_model(
    pred_x, training_data, training_artifacts, prediction_parameters,
):
    # Extract parameters
    posterior_draws = training_artifacts["posterior_draws"]
    normalization_constants = training_artifacts["normalization_constants"]
    num_posterior_draws = prediction_parameters["num_posterior_draws"]
    num_samples = prediction_parameters["num_samples"]
    random_seed = prediction_parameters["random_seed"]

    # Preliminaries ################################################################

    # Extract normalization constants
    y_offset = normalization_constants["y_offset"]
    y_scaling = normalization_constants["y_scaling"]

    # Normalization of the training data
    norm_training_data = {
        "x": training_data["x"],
        "y": normalize(training_data["y"], y_offset, y_scaling,),
    }

    # Problem dimensions
    num_predictions = pred_x.shape[0]

    # Pseudo-random number generator
    prng_key = PRNGKey(random_seed)

    # Thin chain
    processed_chains = process_chains(
        posterior_draws, False, ungroup=False, num_thinned_draws=num_posterior_draws
    )

    # Prediction ###################################################################

    norm_means, norm_variances = bgp_predict(
        pred_x, processed_chains, norm_training_data
    )
    norm_samples = (
        dist.Normal(loc=norm_means, scale=jnp.sqrt(norm_variances))
        .sample(prng_key, sample_shape=(num_samples,))
        .transpose((1, 2, 0))
    )

    # Denormalization and flattening of all stochastic dimensions ##################

    samples = denormalize(norm_samples, y_offset, y_scaling).reshape(
        (num_predictions, num_posterior_draws * num_samples)
    )

    means = denormalize(norm_means, y_offset, y_scaling).reshape(
        (num_predictions, num_posterior_draws)
    )

    variances = (norm_variances * y_scaling ** 2).reshape(
        (num_predictions, num_posterior_draws)
    )

    return samples, means, variances


#######################################################
# Semi-Analytical Active Subspace Computation for BGP #
#######################################################


@partial(vmap, in_axes=(0, None, None), out_axes=0)
def expected_value_of_squared_gradient_at_x_star(x_star, training_data, posterior_draw):
    # Extract parameters
    x_training = training_data["x"]
    y_training = training_data["y"]
    length_scales = posterior_draw["length_scales"]

    # Matrix used throughout the computations
    inv_lambda = jnp.diag(1.0 / length_scales ** 2)  # (D, D)

    # Inputs
    x_star = x_star.flatten()[None, :]  # enforcing (1, D))
    X = x_training  # (N, D)
    X_tilde_star = x_star - X  # (N, D)

    # Kernel computations
    k_x_star_x_star = gp_kernel(x_star, x_star, posterior_draw) + posterior_draw[
        "noise_variance"
    ] * jnp.eye(
        x_star.shape[0]
    )  # (1, 1)
    k_X_x_star = gp_kernel(X, x_star, posterior_draw)  # (N, 1)
    k_X_X = gp_kernel(X, X, posterior_draw) + posterior_draw[
        "noise_variance"
    ] * jnp.eye(
        X.shape[0]
    )  # (N, N)
    chol_k_X_X, lower = cho_factor(k_X_X, lower=True)

    # Outputs
    y_training = y_training.flatten()[:, None]  # enforcing (N, 1)
    alpha = cho_solve((chol_k_X_X, lower), y_training)

    # First term
    # Andrew McHutchon - Differentiating Gaussian Processes - Equation 37
    # when x1_star = x2_star = x_star
    d2k_x1_x2_dx1_dx2_at_x_star_x_star = (
        -inv_lambda * k_x_star_x_star
    )  # (D, D) * (1,1) = (D, D)
    first_term = d2k_x1_x2_dx1_dx2_at_x_star_x_star  # (D, D)

    # Second term
    # Andrew McHutchon - Differentiating Gaussian Processes - Equation 36
    dk_X_x2_dx2_at_x_star = k_X_x_star * (
        X_tilde_star @ inv_lambda
    )  # (N, 1) * ((N, D) @ (D, D)) = (N, D)
    beta = solve_triangular(chol_k_X_X, dk_X_x2_dx2_at_x_star, lower=lower)
    second_term = -beta.T @ beta

    # Third term
    # Andrew McHutchon - Differentiating Gaussian Processes - Equation 3
    df_bar_dx_at_x_star = (
        -inv_lambda @ X_tilde_star.T @ (k_X_x_star * alpha)
    )  # (D, D) @ (D, N) @ ((N, 1) * (N, 1)) = (D, N)
    third_term = df_bar_dx_at_x_star @ df_bar_dx_at_x_star.T  # (D, N) @ (N, D) = (D, D)

    return first_term - second_term + third_term


def estimate_c(list_x_star, training_data, posterior_draw):
    return jnp.mean(
        expected_value_of_squared_gradient_at_x_star(
            list_x_star, training_data, posterior_draw
        ),
        axis=0,
    )


# @partial(map, in_axes=(None, None, None, 0), out_axes=(0, 0))
@partial(jit, static_argnums=(1,))
def draw_w_and_eigenvalues(
    list_x_star, dim_active_subspace, training_data, posterior_draw
):
    c = estimate_c(list_x_star, training_data, posterior_draw)
    eigenvalues, w = eigh(c)
    sorted_indices = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    w = w[:, sorted_indices]
    w = dynamic_slice_in_dim(w, 0, dim_active_subspace, axis=1)
    return w, eigenvalues


def get_w_draws_bgp(
    training_data, validation_data, training_parameters, training_artifacts
):
    list_x_star = jnp.concatenate((training_data["x"], validation_data["x"]), axis=0)
    dim_active_subspace = training_parameters["dim_feature_space"]
    posterior_draws = training_artifacts["posterior_draws"]
    w_draws, _ = jmap(
        partial(
            draw_w_and_eigenvalues, list_x_star, dim_active_subspace, training_data
        ),
        posterior_draws,
    )

    return w_draws
