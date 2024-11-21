import pandas as pd
import numpy as np
import os
from copy import deepcopy


def load_data_ohio(patient_id, include_test=False):
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


def load_data_iso(patient_id):
    """Loads training data from the 2018 and 2020 datasets.
    Optionally includes test data, but only for final evaluation.
    Returns both raw (unmodified) and processed (with elapsed time columns) data."""

    base_paths = "data/CGM-data imputationmodel CGM-ISO study"
    path = f"{base_paths}/ID_{patient_id}completx.csv"

    data = pd.read_csv(path, skiprows=4)
    # print(patient_id, " df = ", data)
    # print("data heads = ", data.head())  # See the first few rows
    data = data[["mmol/L", "Tid"]]
    data["minutes_elapsed"] = np.arange(0, len(data) * 5, 5)
    data["days_elapsed"] = data["minutes_elapsed"] / 1440
    data.rename(columns={"mmol/L": "cbg"}, inplace=True)

    return data


def load_data_cap(patient_id):
    """Loads training data from the 2018 and 2020 datasets.
    Optionally includes test data, but only for final evaluation.
    Returns both raw (unmodified) and processed (with elapsed time columns) data."""

    base_paths = "data/CGM-data imputationmodel CGM-CAP study"
    path = f"{base_paths}/{patient_id}.csv"

    data = pd.read_csv(path, skiprows=11)
    # print(patient_id, " df = ", data)
    # print("data heads = ", data.head())  # See the first few rows
    data = data[["Sensorglukose (mmol/l)", "Dato"]]
    data["minutes_elapsed"] = np.arange(0, len(data) * 5, 5)
    data["days_elapsed"] = data["minutes_elapsed"] / 1440
    data.rename(columns={"Sensorglukose (mmol/l)": "cbg"}, inplace=True)

    return data


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


def get_nan_segments_loc(data_dict: dict, col: str) -> dict:
    """Returns a dictionary of dicts where each data series / patient is indexed individually.
    For each index there is a field 'start_indices', 'end_indices' and 'segment_lengths'.
    """

    segments = {}
    for index in data_dict.keys():
        nan_indices = np.asarray(
            data_dict[index][col][data_dict[index][col].isna()].index
        )  # Get indices for nan entries in cbg
        if len(nan_indices) == 0:
            start_indices = 0
            end_indices = len(data_dict)
        else:
            deriv = np.diff(nan_indices)
            end_indices = np.append(
                nan_indices[np.where(deriv != 1)[0]], nan_indices[-1]
            )
            start_indices = np.insert(
                nan_indices[np.where(deriv != 1)[0] + 1], 0, nan_indices[0]
            )

        lengths = (end_indices - start_indices) + 1
        segments[index] = {
            "start_indices": start_indices,
            "end_indices": end_indices,
            "segment_lengths": lengths,
        }

    return segments


def remove_small_data_segments(
    data_segments, nan_segments_dict, percentile=70, min_non_nan=100
):
    """This removes from the continous data segments that are shorter than 70th percentile of the longest nan segment."""
    data_summary = []

    for key, segments_info in data_segments.items():
        nan_lengths = nan_segments_dict[key]["segment_lengths"]
        max_nan = np.percentile(nan_lengths, percentile)  # Use 70th percentile
        min_length = (
            max_nan + min_non_nan
        )  # So it is by a given amount of samples larger at least.

        keep_indices = [
            i
            for i, length in enumerate(segments_info["segment_lengths"])
            if length >= min_length
        ]
        data_segments[key]["start_indices"] = data_segments[key]["start_indices"][
            keep_indices
        ]
        data_segments[key]["end_indices"] = data_segments[key]["end_indices"][
            keep_indices
        ]
        data_segments[key]["segment_lengths"] = [
            data_segments[key]["segment_lengths"][i] for i in keep_indices
        ]

        data_summary.append(
            {
                "Patient ID": key,
                f"{percentile}th Percentile NaN Length": round(max_nan, 2),
                "Minimum Length to Keep": round(min_length, 2),
                "Num Segments Kept": len(keep_indices),
                "Segment Lengths Kept": data_segments[key]["segment_lengths"],
            }
        )

    summary_df = pd.DataFrame(data_summary)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)

    return data_segments, summary_df


def resize_segments_uniform(data_segments, raw_data_dict, col="cbg"):
    """This function resizes the segments sizes for each patient to the min that is larger than the largest nan segment.
    If a segment is at least twice the size, it splits it and appends to avoid wasting data.
    Because for training all must be same) NOTE: THis is per patient not overall"""

    resized_segments = {}

    for patient_id, segments in data_segments.items():
        if not segments["segment_lengths"]:
            print(f"Warning: No segments for patient {patient_id}. Omitting.")
            continue

        shortest_segment_len = min(
            segments["segment_lengths"]
        )  # find the shortest segment per key

        resized_segments[patient_id] = []
        for idx, length in enumerate(segments["segment_lengths"]):
            start = segments["start_indices"][idx]
            end = segments["end_indices"][idx]
            segment_data = raw_data_dict[patient_id][col].iloc[start : end + 1].values

            # check if it is at least twice the size, then split it and append, to not waste data
            if length >= 2 * shortest_segment_len:
                num_splits = length // shortest_segment_len
                for split_idx in range(num_splits):
                    split_segment = segment_data[
                        split_idx
                        * shortest_segment_len : (split_idx + 1)
                        * shortest_segment_len
                    ]
                    resized_segments[patient_id].append(split_segment)
                remaining_length = len(segment_data) % shortest_segment_len
                if remaining_length >= shortest_segment_len:
                    resized_segments[patient_id].append(
                        segment_data[-shortest_segment_len:]
                    )
            # check if it is equal to or slightly larger than the shortest segment, keep it as is
            elif shortest_segment_len <= length < 2 * shortest_segment_len:
                resized_segments[patient_id].append(segment_data[:shortest_segment_len])
            else:
                print(
                    f"Segment {idx} for patient {patient_id} is smaller than the shortest segment length, skipping."
                )

    return resized_segments


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
