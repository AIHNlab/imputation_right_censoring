import numpy as np
import pandas as pd
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct
from scipy.optimize import minimize

import GPy
from GPy.models import GPRegression


# Define a custom optimizer to control max_iter
def custom_optimizer(obj_func, initial_theta, bounds):
    opt_res = minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 2e05, "gtol": 1e-06},
    )
    # _check_optimize_result("lbfgs", opt_res)
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min


def train_gpr_sklearn(
    timeseries: np.ndarray, kernel=None, var=1
) -> GaussianProcessRegressor:
    """
    Train a GPR model on a given timeseries data array.

    @param timeseries:np.ndarray The CBG data for one patient. Example: train_data_dict['540']['cbg']
    @param output_model_filename:str Under this name the model params will be saved. Example: 'gpr_model_540'. If set to 'None' the model is not saved.
    @param kernel:sklearn.gaussian_process.kernels.Sum A kernel as defined  https://scikit-learn.org/1.5/modules/gaussian_process.html here. Default Kernel is in place.

    @return 'None'
    """

    # If no kernel is provided this is a default one. (We must test out, see: https://scikit-learn.org/1.5/modules/gaussian_process.html)
    if kernel is None:
        maternParams = {
            "length_scale": 10,
            "nu": 1.5,
        }
    kernel = DotProduct(sigma_0=1) * Matern(length_scale=0.1, nu=0.5) + Matern(
        length_scale=0.5, nu=0.5
    )

    # Ensure timeseries is a 1D array
    if timeseries.ndim == 2 and timeseries.shape[1] == 1:
        timeseries = timeseries.flatten()
    elif timeseries.ndim != 1:
        raise ValueError(
            "Expected timeseries to be a 1D array or a 2D array with shape (n_samples, 1)."
        )

    train_X = np.where(~np.isnan(timeseries))[0]  # Indices where values are not NaN
    train_y = timeseries[train_X]  # Actual values at those indices
    train_X = train_X.reshape((-1, 1))

    # Create a GPR newly. alpha is the noise level. Normalize true.
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=55,
        normalize_y=True,
        random_state=0,
        alpha=1e-3,
        optimizer=custom_optimizer,
    )

    # Actual training
    gpr.fit(train_X, train_y)

    return gpr


def inference_gpr_sklearn(
    predict_X: np.ndarray, input_model: str | GaussianProcessRegressor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function loads the defined model and uses it to impute the values at the given indices.

    @param predict_X:np.ndarray The CBG data for one patient. Example: train_data_dict['540']['cbg']
    @param input_model:str Pass here the trained model or from where to load the model params. Example pass a string to load params: 'gpr_model_540'

    @ return missing value indices, predicted y values, standard deviation.
    """
    # Load the model from file
    if isinstance(input_model, str):
        input: str = "gpr_models/" + input_model + ".joblib"
        gpr = joblib.load(input)
    elif isinstance(input_model, GaussianProcessRegressor):
        gpr = input_model
    else:
        print("Invalid model passed to infer. Returning.")
        return

    # Prediction
    # predict_X = np.where(np.isnan(timeseries))[0]
    predict_X = predict_X.reshape((-1, 1))
    predict_y, std_y = gpr.predict(predict_X, return_std=True)

    return (predict_X, predict_y, std_y)


def train_gpr(timeseries: np.ndarray, kernel="matern32", var=1) -> GPRegression:
    if kernel == "matern32":
        kernel = GPy.kern.sde_Matern32(input_dim=1, variance=var, lengthscale=1.0)
    elif kernel == "matern52":
        kernel = GPy.kern.sde_Matern52(input_dim=1, variance=var, lengthscale=1.0)
    elif kernel == "squared_exponential":
        kernel = GPy.kern.RBF(input_dim=1, variance=var, lengthscale=1.0)

    # Ensure timeseries is a 1D array
    if timeseries.ndim == 2 and timeseries.shape[1] == 1:
        timeseries = timeseries.flatten()
    elif timeseries.ndim != 1:
        raise ValueError(
            "Expected timeseries to be a 1D array or a 2D array with shape (n_samples, 1)."
        )

    train_X = np.where(~np.isnan(timeseries))[0]  # Indices where values are not NaN
    train_y = timeseries[train_X]  # Actual values at those indices
    train_X = train_X.reshape((-1, 1)).astype(np.float64)
    train_y = train_y.reshape(-1, 1).astype(np.float64)

    # Create a GPR newly. alpha is the noise level. Normalize true.
    gpr = GPy.models.GPRegression(
        train_X, train_y, kernel, normalizer=False, noise_var=1e-5
    )

    # Optimize the model hyperparameters
    gpr.optimize(max_iters=50)

    return gpr


def inference_gpr(
    predict_X: np.ndarray, input_model: str | GPRegression
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Load the model from file
    if isinstance(input_model, str):
        input: str = "gpr_models/" + input_model + ".joblib"
        gpr = joblib.load(input)
    elif isinstance(input_model, GPRegression):
        gpr = input_model
    else:
        print("Invalid model passed to infer. Returning.")
        return

    # Prediciotn
    if predict_X.size == 0:
        raise ValueError("No missing values found in the timeseries for inference.")
    predict_X = predict_X.reshape((-1, 1))
    predict_y, var_y = gpr.predict(predict_X)

    return predict_X, predict_y, np.sqrt(var_y)
