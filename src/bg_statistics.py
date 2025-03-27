import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def calculate_segment_statistics(
    segments_dict, nan_segments_dict=None, is_resized=False
):
    """Calculate statistics for given segment dictionaries."""
    statistics = {}

    for patient_id in segments_dict.keys():
        if is_resized:
            cont_lengths = [len(seg) for seg in segments_dict[patient_id]]
        else:
            cont_lengths = segments_dict[patient_id]["segment_lengths"]

        nan_lengths = []
        if nan_segments_dict:
            nan_lengths = nan_segments_dict[patient_id]["segment_lengths"]

        statistics[patient_id] = {
            "num_cont_seg": len(cont_lengths),
            "min_cont_length": np.min(cont_lengths) if len(cont_lengths) > 0 else 0,
            "max_cont_length": np.max(cont_lengths) if len(cont_lengths) > 0 else 0,
            "70_percent_cont_length": (
                np.percentile(cont_lengths, 70) if len(cont_lengths) > 0 else 0
            ),
        }

        if nan_segments_dict:
            statistics[patient_id].update(
                {
                    "num_nan_seg": len(nan_lengths),
                    "min_nan_length": (
                        np.min(nan_lengths) if len(nan_lengths) > 0 else 0
                    ),
                    "max_nan_length": (
                        np.max(nan_lengths) if len(nan_lengths) > 0 else 0
                    ),
                    "70_percent_nan_length": (
                        np.percentile(nan_lengths, 70) if len(nan_lengths) > 0 else 0
                    ),
                }
            )

    return pd.DataFrame(statistics).T


def calculate_statistics(data_segments):
    """
    Calculate mean, standard deviation, and coefficient of variation for each patient's data segments.
    """
    stats = {}
    for patient_id, segments in data_segments.items():
        all_data = np.concatenate(segments)
        mean_glucose = np.nanmean(all_data)
        std_glucose = np.nanstd(all_data)
        cv_glucose = std_glucose / mean_glucose
        stats[patient_id] = {
            "Mean": mean_glucose,
            "SD": std_glucose,
            "CV": cv_glucose,
        }
    return pd.DataFrame(stats).T


def compare_statistics(bg_original, bg_before_imputation, bg_after_imputation):
    """
    Compare statistics for bias and MSE between original and imputed data.

    Args:
        bg_original: Dictionary of original glucose data (per patient).
        bg_before_imputation: Dictionary of glucose data before imputation (per patient).
        bg_after_imputation: Dictionary of glucose data after imputation (per patient).
        dataset: Name of the dataset for labeling purposes.

    Returns:
        DataFrame containing Bias and MSE before and after imputation for each metric.
    """
    # Calculate statistics
    statistic_original = calculate_statistics(bg_original)
    statistic_before_imputation = calculate_statistics(bg_before_imputation)
    statistic_after_imputation = calculate_statistics(bg_after_imputation)

    # Prepare results dictionary
    results = {
        "Metric": [],
        "Bias Before Imputation": [],
        "Bias After Imputation": [],
        "MSE Before Imputation": [],
        "MSE After Imputation": [],
        "R-squared Before Imputation": [],
        "R-squared After Imputation": [],
        
    }

    # Metrics to compare
    metrics = ["Mean", "SD", "CV"]

    for metric in metrics:
        # Bias calculations
        bias_before = (
            statistic_before_imputation[metric] - statistic_original[metric]
        ).mean()
        bias_after = (
            statistic_after_imputation[metric] - statistic_original[metric]
        ).mean()

        # MSE calculations
        mse_before = (
            (statistic_before_imputation[metric] - statistic_original[metric]) ** 2
        ).mean()
        mse_after = (
            (statistic_after_imputation[metric] - statistic_original[metric]) ** 2
        ).mean()
        r2_before = r2_score(statistic_before_imputation[metric], statistic_original[metric])
        r2_after = r2_score(statistic_after_imputation[metric], statistic_original[metric])

        # Append to results
        if metric == "CV":
            results["Metric"].append(metric)
            results["Bias Before Imputation"].append(bias_before*100)
            results["Bias After Imputation"].append(bias_after*100)
            results["MSE Before Imputation"].append(mse_before*100)
            results["MSE After Imputation"].append(mse_after*100)
            results["R-squared Before Imputation"].append(r2_before)
            results["R-squared After Imputation"].append(r2_after)
        else:
            results["Metric"].append(metric)
            results["Bias Before Imputation"].append(bias_before)
            results["Bias After Imputation"].append(bias_after)
            results["MSE Before Imputation"].append(mse_before)
            results["MSE After Imputation"].append(mse_after)
            results["R-squared Before Imputation"].append(r2_before)
            results["R-squared After Imputation"].append(r2_after)

    # Return results as DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def compute_errors_flatten(
    args, bg_original, bg_before_imputation, bg_after_imputation
):
    """
    Compute MSE and R² for imputed values and a partial bfill version of before_values.

    Args:
        bg_original (dict): Original glucose data per patient.
        bg_before_imputation (dict): Glucose data before imputation per patient.
        bg_after_imputation (dict): Glucose data after imputation per patient.

    Returns:
        pd.DataFrame: DataFrame containing MSE and R² for imputed values and selective bfill.
    """

    # Function to concatenate all arrays from dictionary values
    def concatenate_values(data_dict):
        return np.concatenate([np.array(v).flatten() for v in data_dict.values()])

    # Convert dictionaries to NumPy arrays
    original_values = concatenate_values(bg_original)
    before_values = concatenate_values(bg_before_imputation)
    after_values = concatenate_values(bg_after_imputation)

    # ---- 1️ Find the indices where imputation happened ----
    imputed_mask = np.isnan(before_values) & ~np.isnan(after_values)

    # Extract corresponding values for imputed indices
    original_imputed = original_values[imputed_mask]
    after_imputed = after_values[imputed_mask]

    # Compute MSE and R² for imputed values
    if original_imputed.size == 0 or after_imputed.size == 0:
        mse_after, r2_after = np.nan, np.nan
    else:
        mse_after = mean_squared_error(original_imputed, after_imputed)
        r2_after = r2_score(original_imputed, after_imputed)

    # ---- 2️ Apply Backward Fill (bfill) only on imputed indices ----
    before_series = pd.Series(before_values)  # Convert to pandas Series
    before_series[imputed_mask] = before_series.ffill()[imputed_mask]
    before_series[imputed_mask] = before_series.bfill()[imputed_mask]
    before_bfilled = before_series.to_numpy()  # Convert back to NumPy array

    # Extract values at imputed indices after selective bfill
    before_bfill_imputed = before_bfilled[imputed_mask]

    # Compute MSE and R² for bfill values
    if original_imputed.size == 0 or before_bfill_imputed.size == 0:
        mse_bfill, r2_bfill = np.nan, np.nan
    else:
        mse_bfill = mean_squared_error(original_imputed, before_bfill_imputed)
        r2_bfill = r2_score(original_imputed, before_bfill_imputed)

    # Store results in a DataFrame
    results = {
        "MSE between true and imputed values": [mse_after],
        "R² between true and imputed values": [r2_after],
        "MSE between true and ffill values": [mse_bfill],
        "R² between true and ffill values": [r2_bfill],
    }

    file_name = f"results/{args.dataset}/imputationMetrics/{args.method}_{args.dataset}_{args.kernel}_{args.percentile}_imputationMetricsFlatten.csv"
    pd.DataFrame(results).to_csv(file_name, index=False)

    return pd.DataFrame(results)


def compute_errors(bg_original, bg_before_imputation, bg_after_imputation):
    """
    Compute MSE and R² for imputed values and a forward-fill version of before_values, per patient.

    Args:
        bg_original (dict): Original glucose data per patient.
        bg_before_imputation (dict): Glucose data before imputation per patient.
        bg_after_imputation (dict): Glucose data after imputation per patient.

    Returns:
        pd.DataFrame: DataFrame containing mean and SD of MSE and R² per patient.
    """

    # Initialize lists to store per-patient results
    mse_after_list, r2_after_list = [], []
    mse_ffill_list, r2_ffill_list = [], []

    # ---- Loop through each patient ----
    for patient_id in bg_original.keys():
        original_values = np.array(bg_original[patient_id]).flatten()
        before_values = np.array(bg_before_imputation[patient_id]).flatten()
        after_values = np.array(bg_after_imputation[patient_id]).flatten()

        # Find indices where imputation happened
        imputed_mask = np.isnan(before_values) & ~np.isnan(after_values)

        # Extract only imputed values
        original_imputed = original_values[imputed_mask]
        after_imputed = after_values[imputed_mask]

        # Compute MSE & R² for imputed values
        mse_after = mean_squared_error(original_imputed, after_imputed)
        r2_after = r2_score(original_imputed, after_imputed)

        mse_after_list.append(mse_after)
        r2_after_list.append(r2_after)

        # ---- Apply Forward Fill (ffill) only on imputed indices ----
        before_series = pd.Series(before_values)
        before_series[imputed_mask] = before_series.ffill()[imputed_mask]
        before_series[imputed_mask] = before_series.bfill()[imputed_mask]
        before_ffilled = before_series.to_numpy()

        # Extract ffill-imputed values
        before_ffill_imputed = before_ffilled[imputed_mask]

        # Compute MSE & R² for ffill values
        mse_ffill = mean_squared_error(original_imputed, before_ffill_imputed)
        r2_ffill = r2_score(original_imputed, before_ffill_imputed)

        mse_ffill_list.append(mse_ffill)
        r2_ffill_list.append(r2_ffill)

    # Compute mean and standard deviation across patients
    results = {
        "MSE_GP_Mean": [np.mean(mse_after_list)],
        "MSE_GP_Median": [np.median(mse_after_list)],
        "MSE_GP_SD": [np.std(mse_after_list)],
        "MSE_GP_Q25": [np.quantile(mse_after_list, 0.25)],
        "MSE_GP_Q75": [np.quantile(mse_after_list, 0.75)],
        "R2_GP_Mean": [np.mean(r2_after_list)],
        "R2_GP_Median": [np.median(r2_after_list)],
        "R2_GP_SD": [np.std(r2_after_list)],
        "R2_GP_Q25": [np.quantile(r2_after_list, 0.25)],
        "R2_GP_Q75": [np.quantile(r2_after_list, 0.75)],
        "MSE_BAS_Mean": [np.mean(mse_ffill_list)],
        "MSE_BAS_Median": [np.median(mse_ffill_list)],
        "MSE_BAS_SD": [np.std(mse_ffill_list)],
        "MSE_BAS_Q25": [np.quantile(mse_ffill_list, 0.25)],
        "MSE_BAS_Q75": [np.quantile(mse_ffill_list, 0.75)],
        "R2_BAS_Mean": [np.mean(r2_ffill_list)],
        "R2_BAS_Median": [np.median(r2_ffill_list)],
        "R2_BAS_SD": [np.std(r2_ffill_list)],
        "R2_BAS_Q25": [np.quantile(r2_ffill_list, 0.25)],
        "R2_BAS_Q75": [np.quantile(r2_ffill_list, 0.75)],
    }

    return pd.DataFrame(results)
