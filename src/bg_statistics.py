import pandas as pd
import numpy as np


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
        stats[patient_id] = {"Mean": mean_glucose, "SD": std_glucose, "CV": cv_glucose}
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

        # Append to results
        results["Metric"].append(metric)
        results["Bias Before Imputation"].append(bias_before)
        results["Bias After Imputation"].append(bias_after)
        results["MSE Before Imputation"].append(mse_before)
        results["MSE After Imputation"].append(mse_after)

    # Return results as DataFrame
    results_df = pd.DataFrame(results)

    return results_df
