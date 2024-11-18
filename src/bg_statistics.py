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


def calculate_mean_sd(data_segments):
    """Calculate mean and standard deviation for each patient's data segments."""
    stats = {}
    for patient_id, segments in data_segments.items():
        all_data = np.concatenate(segments)
        mean_glucose = np.nanmean(all_data)
        std_glucose = np.nanstd(all_data)
        stats[patient_id] = {"Mean": mean_glucose, "Standard Deviation": std_glucose}
    return pd.DataFrame(stats).T


def compare_statistics(original_stats, interpolated_stats, method_name):
    """Compare original and interpolated statistics."""
    comparison = original_stats.copy()
    comparison.columns = ["Original Mean", "Original SD"]
    comparison["Interpolated Mean"] = interpolated_stats["Mean"]
    comparison["Interpolated SD"] = interpolated_stats["Standard Deviation"]
    comparison["Mean Difference"] = (
        comparison["Interpolated Mean"] - comparison["Original Mean"]
    )
    comparison["SD Difference"] = (
        comparison["Interpolated SD"] - comparison["Original SD"]
    )
    comparison
    return comparison
