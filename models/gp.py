import GPy
import joblib
import numpy as np
from GPy.models import GPRegression


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

    gpr = GPy.models.GPRegression(
        train_X, train_y, kernel, normalizer=False, noise_var=1e-5
    )

    gpr.optimize(max_iters=50)

    return gpr


def inference_gpr(
    predict_X: np.ndarray, input_model: str | GPRegression
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
