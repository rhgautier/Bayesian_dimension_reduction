import numpy as np

from bayesian_dimension_reduction.bfs import (
    train_bfs_model,
    predict_bfs_model,
    get_w_draws_bfs,
)
from bayesian_dimension_reduction.moas import (
    train_moas_model,
    predict_moas_model,
    get_w_draws_moas,
)
from bayesian_dimension_reduction.bgp import (
    train_bgp_model,
    predict_bgp_model,
    get_w_draws_bgp,
)
from bayesian_dimension_reduction.utils import (
    generic_validation_routine,
    load_dataset,
    save_case_results,
)

#####################
# Script Parameters #
#####################

CASE_PARAMETERS = {
    "name": "moas_naca0012_lift_60_training_samples",
    "model_name": "moas",  # one of ['bfs', 'moas', 'bgp']
    "dataset": {
        # name of the csv file containing the observed data
        # (assumed to be in the `data` directory)
        "name": "naca0012",
        "num_inputs": 18,  # the number of inputs is used to parse the csv file
        "num_outputs": 2,  # the number of outputs is used to parse the csv file
        "output_name": "lift",  # the csv column name corresponding to the response
    },
    "num_training_samples": 60,
}

MCMC_PARAMETERS = {
    "target_acceptance_probability": 0.8,
    "num_chains": 1,
    "chain_method": "parallel",
    "num_warmup_draws": 500,
    "num_posterior_draws": 1000,
    "random_seed": 0,
    "progress_bar": True,
    "display_summary": True,
}

PYMANOPT_SOLVER_PARAMS = {
    "maxiter": 5000,
    "minstepsize": 1e-5,
    "mingradnorm": 2e-1,
    "cost_improvement_threshold": 1e-3,
    "no_cost_improvement_streak": 50,
}

PROCESS_PARAMETERS = {
    "training_parameters": {
        "bfs": {"dim_feature_space": 1, "mcmc_params": MCMC_PARAMETERS},
        "moas": {
            "dim_feature_space": 1,
            "num_restarts": 100,
            "random_seed_for_restarts": 0,
            "pymanopt_solver_params": PYMANOPT_SOLVER_PARAMS,
        },
        "bgp": {"dim_feature_space": 1, "mcmc_params": MCMC_PARAMETERS},
    },
    "prediction_parameters": {
        "bfs": {
            "dim_feature_space": 1,
            "num_posterior_draws": 100,
            "num_samples": 10,
            "random_seed": 0,
        },
        "moas": {"num_samples": 10, "random_seed": 0},
        "bgp": {"num_posterior_draws": 100, "num_samples": 10, "random_seed": 0},
    },
    "validation_parameters": {"quantile_values": [0.025, 0.5, 0.975]},
}

#####################
# Main Run Function #
#####################


MODEL_ROUTINES = {
    "bfs": (train_bfs_model, predict_bfs_model, get_w_draws_bfs),
    "moas": (train_moas_model, predict_moas_model, get_w_draws_moas),
    "bgp": (train_bgp_model, predict_bgp_model, get_w_draws_bgp),
}


def run_one_case(process_parameters, case_parameters):
    # Retrieve relevant data
    model_name = case_parameters["model_name"]
    (training_routine, prediction_routine, get_w_draws) = MODEL_ROUTINES[model_name]
    training_parameters = process_parameters["training_parameters"][model_name]
    prediction_parameters = process_parameters["prediction_parameters"][model_name]
    validation_parameters = process_parameters["validation_parameters"]

    # Load dataset and
    x, y, dy_dx = load_dataset(case_parameters["dataset"])

    # Randomly split dataset as training and validation sets
    # based on the desired number of training points
    all_indices = np.arange(x.shape[0])
    training_indices = np.sort(
        np.random.choice(
            all_indices, case_parameters["num_training_samples"], replace=False
        )
    )
    validation_indices = np.setdiff1d(all_indices, training_indices)

    # Prepare training and validation sets
    training_data = {"x": x[training_indices], "y": y[training_indices]}
    validation_data = {"x": x[validation_indices], "y": y[validation_indices]}

    # Train model
    training_artifacts = training_routine(training_data, training_parameters)

    # Validate model
    validation_artifacts = generic_validation_routine(
        training_data,
        training_parameters,
        training_artifacts,
        prediction_routine,
        prediction_parameters,
        validation_data,
        validation_parameters,
        get_w_draws,
        dy_dx,
    )

    # Save results
    save_case_results(case_parameters, training_artifacts, validation_artifacts)


if __name__ == "__main__":
    run_one_case(PROCESS_PARAMETERS, CASE_PARAMETERS)
