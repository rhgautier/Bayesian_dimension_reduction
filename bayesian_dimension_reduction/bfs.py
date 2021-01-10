from functools import partial
import time

from jax import jit, vmap, numpy as jnp
from jax.random import PRNGKey
import numpy as np
from numpyro import enable_x64, sample, distributions as dist

from .bgp import bgp_model, bgp_predict_inner1
from .utils import (
    mcmc,
    process_chains,
    compute_normalization,
    normalize,
    denormalize,
    householder_reparameterization,
)

enable_x64()


###################################
# Numpyro model used for training #
###################################


def bfs_model(training_data, num_active_dims):
    # Problem dimensions
    m = num_active_dims
    n = training_data["x"].shape[1]
    num_projection_parameters = m * n - m * (m - 1) // 2

    # Priors
    projection_parameters = sample(
        "projection_parameters",
        dist.Normal(
            jnp.zeros((num_projection_parameters,)),
            jnp.ones((num_projection_parameters,)),
        ),
    )

    # Project inputs
    w = householder_reparameterization(projection_parameters, n, m)
    projected_training_data = {"x": training_data["x"] @ w, "y": training_data["y"]}
    return bgp_model(projected_training_data)


######################
# Prediction routine #
######################


@partial(jit, static_argnums=3)
@partial(vmap, in_axes=(None, 0, None, None), out_axes=(1, 1))
def bfs_predict(pred_x, posterior_draw, training_data, num_active_dims):
    # Problem dimensions
    m = num_active_dims
    n = training_data["x"].shape[1]

    # Retrieve projection matrix
    projection_parameters = posterior_draw["projection_parameters"]
    w = householder_reparameterization(projection_parameters, n, m)

    # Project predictions sites
    projected_pred_x = pred_x @ w

    projected_training_data = {"x": training_data["x"] @ w, "y": training_data["y"]}

    return bgp_predict_inner1(projected_pred_x, posterior_draw, projected_training_data)


######################################################
# High-Level Training/Prediction/Validation Routines #
######################################################


def train_bfs_model(training_data, training_parameters):
    # Extract training parameters
    dim_feature_space = training_parameters["dim_feature_space"]
    mcmc_params = training_parameters["mcmc_params"]
    progress_bar = mcmc_params["progress_bar"]
    display_summary = mcmc_params["display_summary"]

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

    posterior_draws = mcmc(
        bfs_model, (normalized_training_data, dim_feature_space), **mcmc_params,
    )

    training_duration = time.time() - start_time

    return {
        "posterior_draws": posterior_draws,
        "normalization_constants": normalization_constants,
        "training_duration": training_duration,
    }


def predict_bfs_model(
    pred_x, training_data, training_artifacts, prediction_parameters,
):
    # Retrieve quantities of interest
    posterior_draws = training_artifacts["posterior_draws"]
    normalization_constants = training_artifacts["normalization_constants"]
    dim_feature_space = prediction_parameters["dim_feature_space"]
    num_posterior_draws = prediction_parameters["num_posterior_draws"]
    num_samples = prediction_parameters["num_samples"]
    random_seed = prediction_parameters["random_seed"]

    """ Sample from the deep MF GP model. """
    # Preliminaries ################################################################

    # Extract normalization constants
    y_offset = normalization_constants["y_offset"]
    y_scaling = normalization_constants["y_scaling"]

    # Normalization of the training data
    norm_training_data = {
        "x": training_data["x"],
        "y": normalize(training_data["y"], y_offset, y_scaling),
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

    norm_means, norm_variances = bfs_predict(
        pred_x, processed_chains, norm_training_data, dim_feature_space
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


def get_w_draws_bfs(
    training_data, validation_data, training_parameters, training_artifacts
):
    # Problem dimensions
    dim_input_space = training_data["x"].shape[1]
    dim_feature_space = training_parameters["dim_feature_space"]

    # Projection parameters posterior draws
    projection_parameters_draws = training_artifacts["posterior_draws"][
        "projection_parameters"
    ]
    num_posterior_draws = projection_parameters_draws.shape[0]

    # We transform all of these into a projection matrix
    w_draws = np.zeros((num_posterior_draws, dim_input_space, dim_feature_space))
    for i in range(num_posterior_draws):
        w_draws[i] = householder_reparameterization(
            projection_parameters_draws[i], dim_input_space, dim_feature_space
        )

    return w_draws
