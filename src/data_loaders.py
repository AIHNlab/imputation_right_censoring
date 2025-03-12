import pandas as pd
import numpy as np
import os
from copy import deepcopy
from src.data_processing import find_intervals_more_than_15_and_fill


def calculate_percentage_above_threshold(
    dataset, threshold=22.1, output_file="percentage_above_threshold.csv"
):
    """
    Calculate the percentage of data values exceeding the given threshold (default is 22.1).

    Args:
    - dataset: The DataFrame containing the blood glucose data.
    - threshold: The glucose level threshold (default is 22.1 mmol/L).
    - output_file: The name of the CSV file to save the result (default is 'percentage_above_threshold.csv').

    Returns:
    - percentage_df: A DataFrame with the percentage of values above the threshold.
    """
    # Calculate the percentage of values >= threshold
    percentage_above_threshold = (dataset["cbg"] > threshold).mean() * 100

    # Ensure the directory exists before saving
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a DataFrame to store the result
    percentage_df = pd.DataFrame(
        {"Percentage_Above_Threshold": [percentage_above_threshold]}
    )

    # Save the result to the specified CSV file
    percentage_df.to_csv(output_file, index=False)

    return percentage_df


def load_data_ohio(patient_ids, include_test=False):
    """Loads training data from the 2018 and 2020 datasets.
    Optionally includes test data, but only for final evaluation.
    Returns both raw (unmodified) and processed (with elapsed time columns) data."""
    percentage_per_patient = []
    train_data_dict = {}
    test_data_dict = {}
    all_data_dict = {}

    for patient_id in patient_ids:
        base_paths = [
            "C:/PhD/codeAIHN/imputation_OOR/data/Ohio2018_processed",
            "C:/PhD/codeAIHN/imputation_OOR/data/Ohio2020_processed",
        ]
        train_paths = [
            f"{base}/train/{patient_id}-ws-training_processed.csv"
            for base in base_paths
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
        percentage_above_threshold_patient = calculate_percentage_above_threshold(
            train_data,
            threshold=22.1,
            output_file=f"results/ohio/percentage_above_threshold/train_percentage_above_threshold_{patient_id}.csv",
        )
        percentage_per_patient.append(
            (percentage_above_threshold_patient, len(train_data["cbg"]))
        )
        train_data.loc[train_data["cbg"] >= 22.2, "cbg"] = np.nan

        if include_test:
            test_data = pd.concat(
                [pd.read_csv(path) for path in test_paths if os.path.exists(path)],
                ignore_index=True,
            )
            test_data["timestamp"] = pd.to_datetime(
                test_data["5minute_intervals_timestamp"], unit="s"
            )
            test_data["timestamp"] = test_data["timestamp"].dt.strftime(
                "%d/%m/%Y %H:%M"
            )
            test_data["minutes_elapsed"] = np.arange(0, len(test_data) * 5, 5)
            test_data["days_elapsed"] = test_data["minutes_elapsed"] / 1440
            test_data["cbg"] = test_data["cbg"] * 0.0555

            percentage_above_threshold_patient = calculate_percentage_above_threshold(
                test_data,
                threshold=22.1,
                output_file=f"results/ohio/percentage_above_threshold/test_percentage_above_threshold_{patient_id}.csv",
            )
            percentage_per_patient.append(
                (percentage_above_threshold_patient, len(test_data["cbg"]))
            )
            # Replace values >= 22.2 in 'cbg' with NaN
            test_data.loc[test_data["cbg"] >= 22.2, "cbg"] = np.nan

        else:
            test_data = None

        all_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        all_data = all_data.drop(columns=["minutes_elapsed", "days_elapsed"])
        all_data["minutes_elapsed"] = np.arange(0, len(all_data) * 5, 5)
        all_data["days_elapsed"] = all_data["minutes_elapsed"] / 1440

        train_data_dict[patient_id] = train_data
        test_data_dict[patient_id] = test_data
        all_data_dict[patient_id] = all_data

    percentages = np.array([x[0] for x in percentage_per_patient])
    average_percentage = np.mean(percentages)
    std_dev = np.std(percentages)

    percentage_df = pd.DataFrame(
        {
            "Mean_Percentage_Above_Threshold": [average_percentage],
            "Standard_Deviation": [std_dev],
        }
    )

    percentage_df.to_csv(
        f"results/ohio/percentage_above_threshold/percentage_above_threshold_ohio.csv",
        index=False,
    )

    return (
        raw_train_data,
        train_data_dict,
        test_data_dict,
        all_data_dict,
        percentage_df,
        percentage_per_patient,
    )


def load_data_iso(patient_ids):
    base_paths = (
        "C:/PhD/codeAIHN/imputation_OOR/data/CGM-data imputationmodel CGM-ISO study"
    )

    all_data_dict = {}
    percentage_per_patient = []
    for patient_id in patient_ids:
        path = f"{base_paths}/ID_{patient_id}completx.csv"

        data = pd.read_csv(path, skiprows=4)

        data = data[["mmol/L", "Tid"]]
        data = find_intervals_more_than_15_and_fill(data)

        data["minutes_elapsed"] = np.arange(0, len(data) * 5, 5)
        data["days_elapsed"] = data["minutes_elapsed"] / 1440
        data.rename(columns={"mmol/L": "cbg"}, inplace=True)

        percentage_above_threshold_patient = calculate_percentage_above_threshold(
            data,
            threshold=22.1,
            output_file=f"results/iso/percentage_above_threshold/percentage_above_threshold_{patient_id}.csv",
        )
        percentage_per_patient.append((percentage_above_threshold_patient, len(data)))
        data.loc[data["cbg"] >= 22.2, "cbg"] = np.nan

        all_data_dict[patient_id] = data

    percentages = np.array([x[0] for x in percentage_per_patient])
    average_percentage = np.mean(percentages)
    std_dev = np.std(percentages)

    percentage_df = pd.DataFrame(
        {
            "Mean_Percentage_Above_Threshold": [average_percentage],
            "Standard_Deviation": [std_dev],
        }
    )

    percentage_df.to_csv(
        f"results/iso/percentage_above_threshold/percentage_above_threshold_iso.csv",
        index=False,
    )

    return all_data_dict, percentage_df, percentage_per_patient


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
    percentage_per_patient = []
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

        percentage_above_threshold_patient = calculate_percentage_above_threshold(
            patient_data,
            threshold=22.1,
            output_file=f"results/{dataset_name}/percentage_above_threshold/{patient_id}_percentage_above_threshold_{dataset_name}.csv",
        )
        percentage_per_patient.append(
            (percentage_above_threshold_patient, len(patient_data))
        )

        patient_data.loc[patient_data["cbg"] > 22.1, "cbg"] = np.nan

        if len(patient_data) >= 288:
            # Store the patient-specific DataFrame in the dictionary
            all_data_dict[patient_id] = patient_data

    # After processing all patients, calculate the weighted average percentage of values exceeding 22.1 mmol/L for the entire dataset
    percentages = np.array([x[0] for x in percentage_per_patient])
    average_percentage = np.mean(percentages)
    std_dev = np.std(percentages)

    percentage_df = pd.DataFrame(
        {
            "Mean_Percentage_Above_Threshold": [average_percentage],
            "Standard_Deviation": [std_dev],
        }
    )

    percentage_df.to_csv(
        f"results/{dataset_name}/percentage_above_threshold/percentage_above_threshold_{dataset_name}.csv",
        index=False,
    )

    return all_data_dict, all_data_dict.keys(), percentage_df, percentage_per_patient
