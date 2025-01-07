import pandas as pd
import numpy as np
import os
from copy import deepcopy
from src.data_processing import find_intervals_more_than_15_and_fill


def load_data_ohio(patient_id, include_test=False):
    """Loads training data from the 2018 and 2020 datasets.
    Optionally includes test data, but only for final evaluation.
    Returns both raw (unmodified) and processed (with elapsed time columns) data."""

    base_paths = [
        "C:/PhD/codeAIHN/imputation_OOR/data/Ohio2018_processed",
        "C:/PhD/codeAIHN/imputation_OOR/data/Ohio2020_processed",
    ]
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
    train_data["timestamp"] = pd.to_datetime(
        train_data["5minute_intervals_timestamp"], unit="s"
    )
    train_data["timestamp"] = train_data["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
    train_data["minutes_elapsed"] = np.arange(0, len(train_data) * 5, 5)
    train_data["days_elapsed"] = train_data["minutes_elapsed"] / 1440
    # convert mg/dL to mmol/L
    train_data["cbg"] = train_data["cbg"] * 0.0555
    # Replace values >= 22.2 in 'cbg' with NaN
    train_data.loc[train_data["cbg"] >= 22.2, "cbg"] = np.nan

    if include_test:
        test_data = pd.concat(
            [pd.read_csv(path) for path in test_paths if os.path.exists(path)],
            ignore_index=True,
        )
        test_data["timestamp"] = pd.to_datetime(
            test_data["5minute_intervals_timestamp"], unit="s"
        )
        test_data["timestamp"] = test_data["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
        test_data["minutes_elapsed"] = np.arange(0, len(test_data) * 5, 5)
        test_data["days_elapsed"] = test_data["minutes_elapsed"] / 1440
        test_data["cbg"] = test_data["cbg"] * 0.0555

        # Replace values >= 22.2 in 'cbg' with NaN
        test_data.loc[test_data["cbg"] >= 22.2, "cbg"] = np.nan

    else:
        test_data = None

    all_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    all_data = all_data.drop(columns=["minutes_elapsed", "days_elapsed"])
    all_data["minutes_elapsed"] = np.arange(0, len(all_data) * 5, 5)
    all_data["days_elapsed"] = all_data["minutes_elapsed"] / 1440

    return raw_train_data, train_data, test_data, all_data


def load_data_iso(patient_id):
    base_paths = (
        "C:/PhD/codeAIHN/imputation_OOR/data/CGM-data imputationmodel CGM-ISO study"
    )
    path = f"{base_paths}/ID_{patient_id}completx.csv"

    data = pd.read_csv(path, skiprows=4)

    data = data[["mmol/L", "Tid"]]
    data = find_intervals_more_than_15_and_fill(data)

    data["minutes_elapsed"] = np.arange(0, len(data) * 5, 5)
    data["days_elapsed"] = data["minutes_elapsed"] / 1440
    data.rename(columns={"mmol/L": "cbg"}, inplace=True)
    # print(patient_id, ":BG > 22 = ", data[data["cbg"] > 22])
    data.loc[data["cbg"] >= 22.2, "cbg"] = np.nan

    return data


def load_data_cap(patient_id):
    base_paths = (
        "C:/PhD/codeAIHN/imputation_OOR/data/CGM-data imputationmodel CGM-CAP study"
    )
    path = f"{base_paths}/{patient_id}.csv"

    data = pd.read_csv(path, skiprows=11)

    data = data[["Sensorglukose (mmol/l)", "Dato", "Klokkeslæt"]]
    # Combine 'Dato' and 'Klokkeslæt' into a single datetime column
    data["Tid"] = pd.to_datetime(
        data["Dato"] + " " + data["Klokkeslæt"], format="%m/%d/%Y %H:%M:%S"
    )
    data["Tid"] = data["Tid"].dt.strftime("%d/%m/%Y %H:%M")

    # Drop the original 'Dato' and 'Klokkeslæt' columns
    data.drop(columns=["Dato", "Klokkeslæt"], inplace=True)
    data.rename(columns={"Sensorglukose (mmol/l)": "cbg"}, inplace=True)
    data = find_intervals_more_than_15_and_fill(data)
    data["minutes_elapsed"] = np.arange(0, len(data) * 5, 5)
    data["days_elapsed"] = data["minutes_elapsed"] / 1440
    # print(patient_id, ":BG > 22 = ", data[data["cbg"] > 22])
    data.loc[data["cbg"] >= 22.2, "cbg"] = np.nan

    return data


def load_data_cap1(patient_id):
    base_paths = (
        "C:/PhD/codeAIHN/imputation_OOR/data/CGM-data imputationmodel CGM-CAP study"
    )
    path = f"{base_paths}/{patient_id}.csv"

    data = pd.read_csv(path, skiprows=11)

    data = data[["Sensorglukose (mmol/l)", "Dato", "Klokkeslæt"]]
    # Combine 'Dato' and 'Klokkeslæt' into a single datetime column
    data["Tid"] = pd.to_datetime(
        data["Dato"] + " " + data["Klokkeslæt"], format="%m/%d/%Y %H:%M:%S"
    )
    data["Tid"] = data["Tid"].dt.strftime("%d/%m/%Y %H:%M")

    # Drop the original 'Dato' and 'Klokkeslæt' columns
    data.drop(columns=["Dato", "Klokkeslæt"], inplace=True)
    data.rename(columns={"Sensorglukose (mmol/l)": "cbg"}, inplace=True)
    data = find_intervals_more_than_15_and_fill(data)
    data["minutes_elapsed"] = np.arange(0, len(data) * 5, 5)
    data["days_elapsed"] = data["minutes_elapsed"] / 1440
    # print(patient_id, ":BG > 22 = ", data[data["cbg"] > 22])
    data.loc[data["cbg"] >= 22.2, "cbg"] = np.nan

    return data


def load_data_glucobench(dataset_name):
    """Loads training data from the 2018 and 2020 datasets.
    Optionally includes test data, but only for final evaluation.
    Returns both raw (unmodified) and processed (with elapsed time columns) data."""

    path = "C:/PhD/codeAIHN/imputation_OOR/data/" + str(dataset_name) + ".csv"

    data = pd.read_csv(path)
    data.rename(columns={"time": "Tid", "gl": "cbg"}, inplace=True)
    data["cbg"] = data["cbg"] * 0.0555
    data["id"] = data["id"].astype(str)

    if dataset_name == "colas":
        # Keep only rows where the T2DM column is True
        data = data[data["T2DM"] == True].copy()
    elif dataset_name == "hall":
        data = data[data["diagnosis"] == 2].copy()

    # Initialize an empty dictionary to store DataFrames for each patient
    all_data_dict = {}

    # Iterate over the unique patient IDs in the DataFrame
    for patient_id in data["id"].unique():
        # Filter the DataFrame for the current patient_id
        patient_data = data[data["id"] == patient_id].copy()

        # Ensure "Tid" is in datetime format for proper processing
        patient_data["Tid"] = pd.to_datetime(patient_data["Tid"])

        # Sort the DataFrame by "Tid"
        patient_data.sort_values(by="Tid", inplace=True)
        patient_data.reset_index(drop=True, inplace=True)  # Reset the index

        # Format "Tid" back to the desired string format if needed
        patient_data["Tid"] = patient_data["Tid"].dt.strftime("%d/%m/%Y %H:%M")

        patient_data = find_intervals_more_than_15_and_fill(patient_data)
        patient_data["minutes_elapsed"] = np.arange(0, len(patient_data) * 5, 5)
        patient_data["days_elapsed"] = patient_data["minutes_elapsed"] / 1440

        patient_data.loc[patient_data["cbg"] > 22.1, "cbg"] = np.nan

        if len(patient_data) >= 288:
            # Store the patient-specific DataFrame in the dictionary
            all_data_dict[patient_id] = patient_data

    return all_data_dict, all_data_dict.keys()
