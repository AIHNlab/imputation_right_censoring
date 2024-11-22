import matplotlib.pyplot as plt
import numpy as np


def plot_patient_data_with_quantile(
    original_data: dict, quantile_cut_data: dict, quantile_cutoff: float = 0.8
):
    """Visualizes the original and quantile-cut data for each patient side-by-side with shared y-axis limits, using days as the x-axis."""
    fig, axes = plt.subplots(
        nrows=len(original_data),
        ncols=2,
        figsize=(20, 3 * len(original_data)),
    )
    axes = axes.flatten()

    for i, patient_id in enumerate(original_data.keys()):
        if i * 2 + 1 >= len(axes):
            break

        original_patient_data = original_data[patient_id]
        quantile_cut_patient_data = quantile_cut_data[patient_id]
        y_min = min(
            original_patient_data["cbg"].min(), quantile_cut_patient_data["cbg"].min()
        )
        y_max = max(
            original_patient_data["cbg"].max(), quantile_cut_patient_data["cbg"].max()
        )

        # Left plot: Original data using days as x-axis
        axes[2 * i].plot(
            original_patient_data["days_elapsed"],
            original_patient_data["cbg"],
            color="blue",
        )
        axes[2 * i].set_ylim([y_min, y_max])  # Set shared y-axis limits
        axes[2 * i].set_title(f"Patient {patient_id} | Original Data")
        axes[2 * i].set_xlabel("Days Elapsed")
        axes[2 * i].set_ylabel("CBG (mg/dL)")
        axes[2 * i].grid(True)

        # Right plot: Quantile-cut data with days as x-axis and 80th percentile threshold in title
        quantile_value = original_patient_data["cbg"].quantile(quantile_cutoff)
        axes[2 * i + 1].plot(
            quantile_cut_patient_data["days_elapsed"],
            quantile_cut_patient_data["cbg"],
            color="blue",
        )
        axes[2 * i + 1].set_ylim([y_min, y_max])
        axes[2 * i + 1].set_title(
            f"Patient {patient_id} | Quantile-Cut Data (80th Percentile: {int(quantile_value)})"
        )
        axes[2 * i + 1].set_xlabel("Days Elapsed")
        axes[2 * i + 1].set_ylabel("CBG (mg/dL)")
        axes[2 * i + 1].grid(True)

    plt.tight_layout()
    plt.savefig("figures/plot_patient_data_with_quantile.png")
    # plt.show()


def visualize_original_interpolated(
    interpolated_segments,
    cleaned_censored_segments,
    cleaned_original_segments,
    thresh_data,
    method_name="Polynomian",
):
    """Visualize original and interpolated test data segments for each patient."""
    fig, axes = plt.subplots(
        nrows=len(cleaned_original_segments),
        ncols=2,
        figsize=(
            20,
            3 * len(cleaned_original_segments),
        ),
    )
    axes = axes.flatten()

    for i, patient_id in enumerate(cleaned_original_segments.keys()):
        if i * 2 + 1 >= len(axes):
            break

        if len(cleaned_original_segments[patient_id]) > 1:
            original_segment = np.concatenate(cleaned_censored_segments[patient_id])
            interpolated_segment = np.concatenate(interpolated_segments[patient_id])
            original_patient_data = np.concatenate(
                cleaned_original_segments[patient_id]
            )
        else:
            original_segment = cleaned_censored_segments[patient_id][0]
            interpolated_segment = interpolated_segments[patient_id][0]
            original_patient_data = cleaned_original_segments[patient_id][0]

        patient_thresh_data = thresh_data[patient_id]
        repeated_thresh = np.repeat(patient_thresh_data, 288)
        # original_censored_segment = original_censored_segment[patient_id][0]
        # interpolated_segment = interpolated_segment[patient_id][0]
        # original_patient_data = original_patient_data[patient_id][0]
        y_min = min(np.nanmin(interpolated_segment), np.nanmin(original_patient_data))
        y_max = max(np.nanmax(interpolated_segment), np.nanmax(original_patient_data))

        axes[2 * i + 1].plot(
            interpolated_segment, color="orange", label="Interpolated Data", alpha=0.8
        )
        axes[2 * i + 1].plot(original_segment, color="purple", label="Original Data")
        axes[2 * i + 1].set_ylim([y_min, y_max])  # Set shared y-axis limits
        axes[2 * i + 1].set_title(
            f"Patient {patient_id} | {method_name} Interpolation on Test Data"
        )
        axes[2 * i + 1].set_xlabel("Time (minutes)")
        axes[2 * i + 1].set_ylabel("CBG (mg/dL)")
        axes[2 * i + 1].grid(True)

        axes[2 * i].plot(original_patient_data, color="purple")
        axes[2 * i].plot(
            repeated_thresh,
            color="orange",
            linestyle="--",
            label="80th Percentile Threshold",
        )
        axes[2 * i].set_ylim([y_min, y_max])  # Set shared y-axis limits
        axes[2 * i].set_title(
            f"Patient {patient_id} | Quantile-Cut Data (80th Percentile)"
        )
        axes[2 * i].set_xlabel("Days Elapsed")
        axes[2 * i].set_ylabel("CBG (mg/dL)")

    plt.tight_layout()
    plt.savefig("figures/" + method_name + ".png")
    # plt.show()
