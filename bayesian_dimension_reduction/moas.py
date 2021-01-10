from time import time
from functools import partial

from jax.scipy.linalg import cho_solve, cho_factor
from jax import numpy as jnp
from jax.random import PRNGKey
import numpy as np
from numpyro import enable_x64, distributions as dist
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Stiefel, Product
from tqdm import tqdm

from .bgp import gp_kernel, bgp_predict_inner2
from .utils import (
    CustomConjugateGradient,
    compute_normalization,
    normalize,
    denormalize,
)

enable_x64()

#############################################
# Multi-Fidelity GP Negative Log-Likelihood #
#############################################


def extract_model_parameters(optim_params):
    return {
        "w": optim_params[0],
        "noise_variance": jnp.exp(optim_params[1][0]),
        "signal_variance": jnp.exp(optim_params[1][1]),
        "length_scales": jnp.exp(optim_params[1][2:]),
    }


def moas_negative_log_likelihood(training_data, w, log_model_parameters):
    # Unpacking the optim parameters
    model_parameters = extract_model_parameters((w, log_model_parameters))

    # Unpacking training data and problem dimensions
    x, y = training_data["x"], training_data["y"]
    num_samples = x.shape[0]

    # Project training data
    z = training_data["x"] @ model_parameters["w"]

    # Covariance matrix
    k = gp_kernel(z, z, model_parameters) + model_parameters[
        "noise_variance"
    ] * jnp.eye(num_samples)

    # Log likelihood
    c, lower = cho_factor(k, lower=True)
    alpha = cho_solve((c, lower), y)
    return (
        0.5 * y.T @ alpha
        + jnp.sum(jnp.log(jnp.diagonal(c)))
        + 0.5 * num_samples * jnp.log(2 * jnp.pi)
    )[0, 0]


########################
# Optimization Routine #
########################


def optimize_moas_model(
    problem, pymanopt_solver_params, num_restarts, get_x_0, random_seed_for_restarts=0,
):
    # Fixing seed for reproducibility
    np.random.seed(random_seed_for_restarts)

    # Training with restarts
    solver = CustomConjugateGradient(**pymanopt_solver_params)
    best_x, best_cost = None, np.inf
    with tqdm(range(num_restarts)) as progress_bar:
        for _ in progress_bar:
            x_0 = get_x_0()
            (w, log_model_parameters), cost = solver.solve(problem, x=x_0)
            progress_bar.set_postfix_str(
                f"cost = {cost:.2f}, best_cost = {best_cost:.2f}"
            )
            best_x, best_cost = (
                ((w, log_model_parameters), cost)
                if cost < best_cost
                else (best_x, best_cost)
            )
    # print(best_x)
    # Tighten it up
    print("Tightening it up...")
    solver = CustomConjugateGradient(use_cost_improvement_criterion=False)
    x, _ = solver.solve(problem, x=best_x)
    return x


######################################################
# High-Level Training/Prediction/Validation Routines #
######################################################


def train_moas_model(training_data, training_parameters):
    # Extract training parameters
    dim_feature_space = training_parameters["dim_feature_space"]
    num_restarts = training_parameters["num_restarts"]
    random_seed_for_restarts = training_parameters["random_seed_for_restarts"]
    pymanopt_solver_params = training_parameters["pymanopt_solver_params"]

    # Problem dimensions
    num_inputs = training_data["x"].shape[1]

    # Start timer that measures training time
    start_time = time()

    # Normalize outputs and store normalization constants
    y_offset, y_scaling = compute_normalization(training_data["y"])
    normalization_constants = {"y_offset": y_offset, "y_scaling": y_scaling}

    # Normalize training data
    normalized_training_data = {
        "x": training_data["x"],
        "y": normalize(training_data["y"], y_offset, y_scaling),
    }

    # Optimization
    projection_matrix_manifold = Stiefel(num_inputs, dim_feature_space)
    hyperparameters_manifold = Euclidean(2 + num_inputs)
    product_manifold = Product((projection_matrix_manifold, hyperparameters_manifold))
    cost_function = pymanopt.function.Jax(
        partial(moas_negative_log_likelihood, normalized_training_data)
    )
    problem = Problem(product_manifold, cost_function, verbosity=-1)

    def get_x_0():
        return (
            projection_matrix_manifold.rand(),
            np.concatenate(
                [
                    np.random.uniform(-10, 1, size=(1,)),  # noise variance
                    np.random.uniform(-1, 5, size=(1,)),  # signal variance
                    np.random.uniform(-2, 5, size=(dim_feature_space,)),  # len. scales
                ]
            ),
        )

    optim_params = optimize_moas_model(
        problem,
        pymanopt_solver_params,
        num_restarts,
        get_x_0,
        random_seed_for_restarts=random_seed_for_restarts,
    )
    model_parameters = extract_model_parameters(optim_params)

    # Record time elapsed for training
    training_duration = time() - start_time

    # Return training artifacts
    return {
        "model_parameters": model_parameters,
        "normalization_constants": normalization_constants,
        "training_duration": training_duration,
    }


def predict_moas_model(
    pred_x, training_data, training_artifacts, prediction_parameters, jitter=1e-8,
):
    # Extract parameters
    model_parameters = training_artifacts["model_parameters"]
    normalization_constants = training_artifacts["normalization_constants"]
    num_samples = prediction_parameters["num_samples"]
    random_seed = prediction_parameters["random_seed"]
    num_predictions = pred_x.shape[0]

    # Preliminaries ####################################################################

    # Extract normalization constants
    y_offset = normalization_constants["y_offset"]
    y_scaling = normalization_constants["y_scaling"]

    # Project inputs onto the feature space
    proj_train_x = training_data["x"] @ model_parameters["w"]
    proj_pred_x = pred_x @ model_parameters["w"]

    # Repackaging projected training inputs and normalized training outputs
    norm_proj_training_data = {
        "x": proj_train_x,
        "y": normalize(training_data["y"], y_offset, y_scaling),
    }

    # Prediction #######################################################################
    k = (
        gp_kernel(proj_train_x, proj_train_x, model_parameters)
        + model_parameters["noise_variance"] * jnp.eye(proj_train_x.shape[0])
        + jitter * jnp.eye(proj_train_x.shape[0])
    )
    c_and_lower = cho_factor(k, lower=True)
    alpha = cho_solve(c_and_lower, norm_proj_training_data["y"])
    norm_means, norm_variances = bgp_predict_inner2(
        proj_pred_x, model_parameters, norm_proj_training_data, c_and_lower, alpha
    )

    # Pseudo-random number generator
    prng_key = PRNGKey(random_seed)

    norm_samples = (
        dist.Normal(loc=norm_means, scale=jnp.sqrt(norm_variances))
        .sample(prng_key, sample_shape=(num_samples,))
        .transpose((1, 0))
    )

    # Denormalization and flattening of all stochastic dimensions ##################

    samples = denormalize(norm_samples, y_offset, y_scaling)
    means = denormalize(norm_means, y_offset, y_scaling).reshape((num_predictions, 1))
    variances = (norm_variances * y_scaling ** 2).reshape((num_predictions, 1))

    return samples, means, variances


def get_w_draws_moas(
    training_data, validation_data, training_parameters, training_artifacts
):
    return training_artifacts["model_parameters"]["w"][None, ...]
