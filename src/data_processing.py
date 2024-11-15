import pandas as pd
import numpy as np
import os
from copy import deepcopy


def load_data(patient_id, include_test=False):
    """Loads training data from the 2018 and 2020 datasets.
    Optionally includes test data, but only for final evaluation.
    Returns both raw (unmodified) and processed (with elapsed time columns) data."""

    base_paths = ["data/Ohio2018_processed", "data/Ohio2020_processed"]
    train_paths = [
        f"{base}/train/{patient_id}-ws-training_processed.csv" for base in base_paths
    ]
    test_paths = [
        f"{base}/test/{patient_id}-ws-testing_processed.csv" for base in base_paths
    ]

    raw_train_data = pd.concat(
        [pd.read_csv(path) for path in train_paths if os.path.exists(path)],
        ignore_index=True,
    )
    # train_data = raw_train_data.copy()
    train_data = deepcopy(raw_train_data)

    train_data["minutes_elapsed"] = np.arange(0, len(train_data) * 5, 5)
    train_data["days_elapsed"] = train_data["minutes_elapsed"] / 1440

    if include_test:
        test_data = pd.concat(
            [pd.read_csv(path) for path in test_paths if os.path.exists(path)],
            ignore_index=True,
        )
        test_data["minutes_elapsed"] = np.arange(0, len(test_data) * 5, 5)
        test_data["days_elapsed"] = test_data["minutes_elapsed"] / 1440
    else:
        test_data = None

    nan_row = pd.DataFrame([[np.nan] * train_data.shape[1]], columns=train_data.columns)
    all_data = pd.concat([train_data, nan_row, test_data], axis=0, ignore_index=True)

    return raw_train_data, train_data, test_data, all_data


def apply_quantile_cut(
    patient_rawdata_dict: dict, nan_above_quantile: float = 0.8
) -> dict:
    """This function evaluates the value for the given quantile from the raw data per patient.
    For each of the segments of the patient values above are set to nan."""
    quantile_cut_data = {}
    thresh_data = {}
    for patient_id, data in patient_rawdata_dict.items():
        thresh = data["cbg"].quantile(nan_above_quantile)
        thresh_data[patient_id] = thresh

        quantile_cut_data[patient_id] = data.copy()
        quantile_cut_data[patient_id]["cbg"] = np.where(
            data["cbg"] > thresh, np.nan, data["cbg"]
        )

    return quantile_cut_data, thresh_data


def get_continuous_segments_loc(data_dict: dict, col: str) -> dict:
    """Returns a dictionary of dicts where each data series / patient is indexed individually.
    For each index there is a field 'start_indices', 'end_indices' and 'segment_lengths'.
    """

    segments = {}
    for index in data_dict.keys():
        non_nan_indices = np.asarray(
            data_dict[index][col][data_dict[index][col].notna()].index
        )  # Get indices for non nan entries in cbg
        deriv = np.diff(non_nan_indices)

        end_indices = np.append(
            non_nan_indices[np.where(deriv != 1)[0]], non_nan_indices[-1]
        )
        start_indices = np.insert(
            non_nan_indices[np.where(deriv != 1)[0] + 1], 0, non_nan_indices[0]
        )

        lengths = (end_indices - start_indices) + 1
        segments[index] = {
            "start_indices": start_indices,
            "end_indices": end_indices,
            "segment_lengths": lengths,
        }

    return segments


# Function to remove leading and trailing NaN values, while keeping the corresponding values in original_segments
def clean_segment_and_align(censored_segment, original_segment):
    # Remove leading NaNs
    start_idx = 0
    while start_idx < len(censored_segment) and np.isnan(censored_segment[start_idx]):
        start_idx += 1

    # Remove trailing NaNs
    end_idx = len(censored_segment) - 1
    while end_idx >= 0 and np.isnan(censored_segment[end_idx]):
        end_idx -= 1

    # Slice the segments to match the cleaned portion
    cleaned_censored_segment = censored_segment[start_idx : end_idx + 1]
    cleaned_original_segment = original_segment[start_idx : end_idx + 1]

    # Return the cleaned segments from both sensored and original data
    return cleaned_censored_segment, cleaned_original_segment
