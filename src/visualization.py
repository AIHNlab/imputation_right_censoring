import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_original_interpolated(
    args,
    interpolated_segments,
    cleaned_censored_segments,
    cleaned_original_segments,
    thresh_data,
    method_name="Polynomian",
):
    """Visualize original and interpolated test data segments for each patient."""

    for i, patient_id in enumerate(cleaned_original_segments.keys()):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(
                20,
                5,
            ),
        )
        axes = axes.flatten()

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

        y_min = min(np.nanmin(interpolated_segment), np.nanmin(original_patient_data))
        y_max = max(np.nanmax(interpolated_segment), np.nanmax(original_patient_data))

        axes[1].plot(
            interpolated_segment, color="orange", label="Interpolated Data", alpha=0.8
        )
        axes[1].plot(original_segment, color="purple", label="Original Data")
        axes[1].set_ylim([y_min, y_max])
        axes[1].set_title(
            f"Patient {patient_id} | {method_name} on Right-cencored Data"
        )
        axes[1].set_xlabel("Time (minutes)")
        axes[1].set_ylabel("CGM (mmol/L)")
        axes[1].grid(
            True, which="both", linestyle="--", linewidth=0.5
        )

        axes[0].plot(original_patient_data, color="purple")
        axes[0].plot(
            repeated_thresh,
            color="orange",
            linestyle="--",
            label=str(args.percentile * 100) + "th Percentile Threshold",
        )
        axes[0].set_ylim([y_min, y_max])
        axes[0].set_title(
            f"Patient {patient_id} | Quantile-Cut Data ({args.percentile * 100}th Percentile)"
        )
        axes[0].set_xlabel("Time (minutes)")
        axes[0].set_ylabel("CGM (mmol/L)")
        axes[0].grid(
            True, which="both", linestyle="--", linewidth=0.5
        )

        plt.tight_layout()
        os.makedirs(
            f"figures/{args.method}/{args.dataset}/{args.kernel}_{args.percentile}",
            exist_ok=True,
        )
        plt.savefig(
            f"figures/{args.method}/{args.dataset}/{args.kernel}_{args.percentile}/{patient_id}.png"
        )

        plt.close()

    # plt.show()
