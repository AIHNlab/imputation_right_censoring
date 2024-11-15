import numpy as np
import pandas as pd
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    RBF,
    WhiteKernel,
    ExpSineSquared,
    Matern,
)


def train_gpr(
    timeseries: np.ndarray, output_model_filename: str, kernel=None
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
        # kernel = RBF() + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)) # <class 'sklearn.gaussian_process.kernels.Sum'>
        # kernel = RBF(length_scale=10) + ExpSineSquared(length_scale=5, periodicity=24) + WhiteKernel(noise_level=1e-2)
        kernel = Matern(length_scale=1, nu=1.5)

    # Ensure timeseries is a 1D array
    if timeseries.ndim == 2 and timeseries.shape[1] == 1:
        timeseries = timeseries.flatten()
    elif timeseries.ndim != 1:
        raise ValueError(
            "Expected timeseries to be a 1D array or a 2D array with shape (n_samples, 1)."
        )

    # The training data is composed of the indices of valid values and the values themselves. (More features would be easily implemented -> computation time?)
    train_X = np.where(~np.isnan(timeseries))[0]  # Indices where values are not NaN
    train_y = timeseries[train_X]  # Actual values at those indices
    train_X = train_X.reshape((-1, 1))
    # print("train_X = ", train_X)

    # Create a GPR newly. alpha is the noise level. Normalize true.
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.2, normalize_y=True)

    # Actual training
    gpr.fit(train_X, train_y)

    # # Saving the model file so we do not lose the training.
    # if output_model_filename is not None:
    #     out:str = 'gpr_models/'+output_model_filename+'.joblib'
    #     joblib.dump(gpr, out, compress=True) # joblib files are mich smaller than pickle
    #     print(f"Training finished and joblib file exported to {out}")

    return gpr


def inference_gpr(
    timeseries: np.ndarray, input_model: str | GaussianProcessRegressor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function loads the defined model and uses it to impute the values at the given indices.

    @param timeseries:np.ndarray The CBG data for one patient. Example: train_data_dict['540']['cbg']
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

    # Prediciotn
    predict_X = np.where(np.isnan(timeseries))[0]
    predict_X = predict_X.reshape((-1, 1))
    predict_y, std_y = gpr.predict(predict_X, return_std=True)

    return (predict_X, predict_y, std_y)
