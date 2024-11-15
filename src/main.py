import argparse
import numpy as np
import pandas as pd

from src.data_processing import (
    load_data,
    apply_quantile_cut,
    get_continuous_segments_loc,
    clean_segment_and_align,
)
from src.visualization import (
    plot_patient_data_with_quantile,
    visualize_original_interpolated,
)
from models.baselines import naive_baseline_imputation
from models.gp import train_gpr, inference_gpr
from src.bg_statistics import (
    calculate_segment_statistics,
    calculate_mean_sd,
    compare_statistics,
)

if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Pipeline for glucose data analysis.")

    parser.add_argument(
        "--patient_ids",
        nargs="+",
        required=True,
        help="List of patient IDs to process (e.g., 540 544 552).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.8,
        help="Quantile threshold for cutoff (default: 0.8).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save processed data and plots (optional).",
    )

    args = parser.parse_args()

    # Step 1: Load data
    print("Loading data...")
    patient_ids = [
        "540",
        "544",
        "552",
        "559",
        "563",
        "567",
        "570",
        "575",
        "584",
        "588",
        "591",
        "596",
    ]
    train_data_dict = {}
    test_data_dict = {}
    all_data_dict = {}
    for patient_id in patient_ids:
        _, train_data, test_data, all_data = load_data(patient_id, include_test=True)
        train_data_dict[patient_id] = train_data
        test_data_dict[patient_id] = test_data
        all_data_dict[patient_id] = all_data

    # Step 2: Apply percentile cut
    print(f"Applying {args.percentile * 100}% percentile cutoff...")
    quantile_cut_data, thresh_data = apply_quantile_cut(
        all_data_dict, nan_above_quantile=args.percentile
    )

    # Step 3: Visualize results
    print("Generating plots...")
    plot_patient_data_with_quantile(all_data, quantile_cut_data, args.percentile)

    # 1. create continues
    cont_segments_dict = {}

    for patient_id in patient_ids:
        patient_data = all_data_dict
        cont_segments_loc = get_continuous_segments_loc(
            {patient_id: patient_data[patient_id]}, "cbg"
        )
        cont_segments_dict[patient_id] = cont_segments_loc[patient_id]

    # extract the original segments before censoring
    # Dictionary for storing the interpolated data for each patient
    original_segments = {}

    # Assuming 'original_data' is a dictionary with patient data, and each entry is a 1D array of continuous data points
    for patient_id, segments_info in cont_segments_dict.items():
        start_indices = segments_info["start_indices"]
        end_indices = segments_info["end_indices"]

        # Store interpolated segments for the current patient
        original_segments[patient_id] = []

        for start, end in zip(start_indices, end_indices):
            # Extract the segment data between the start and end indices
            segment_data = all_data_dict[patient_id]["cbg"][start : end + 1]
            original_segments[patient_id].append(np.array(segment_data))

    # 2. censored segments
    # Dictionary for storing the interpolated data for each patient
    censored_segments = {}

    # Assuming 'original_data' is a dictionary with patient data, and each entry is a 1D array of continuous data points
    for patient_id, segments_info in cont_segments_dict.items():
        start_indices = segments_info["start_indices"]
        end_indices = segments_info["end_indices"]

        # Store interpolated segments for the current patient
        censored_segments[patient_id] = []

        for start, end in zip(start_indices, end_indices):
            # Extract the segment data between the start and end indices
            segment_data = all_data_dict[patient_id]["cbg"][start : end + 1]
            has_nan = np.isnan(segment_data).any()
            # print("Contains NaN values:", has_nan)
            censored_segment = np.where(
                segment_data > thresh_data[patient_id], np.nan, segment_data
            )
            censored_segments[patient_id].append(censored_segment)

    # 3. clean segments

    # Assuming sensored_segments and original_segments are dictionaries containing segments
    # Initialize cleaned versions of both dictionaries
    cleaned_censored_segments = {}
    cleaned_original_segments = {}

    # Iterate through each patient in sensored_segments and original_segments
    for patient_id in censored_segments.keys():
        # Initialize lists to store cleaned segments
        cleaned_censored_segments[patient_id] = []
        cleaned_original_segments[patient_id] = []

        # Process each segment in sensored_segments and original_segments
        for censored_segment, original_segment in zip(
            censored_segments[patient_id], original_segments[patient_id]
        ):
            cleaned_censored_segment, cleaned_original_segment = (
                clean_segment_and_align(censored_segment, original_segment)
            )

            # Append the cleaned segments
            if (
                cleaned_censored_segment.shape[0] != 0
                and cleaned_original_segment.shape[0] != 0
            ):
                cleaned_censored_segments[patient_id].append(cleaned_censored_segment)
            if (
                cleaned_original_segment.shape[0] != 0
                and cleaned_censored_segment.shape[0] != 0
            ):
                cleaned_original_segments[patient_id].append(cleaned_original_segment)

    # Now the cleaned_sensored_segments and cleaned_original_segments dictionaries
    # contain segments without leading or trailing NaN values, and they are aligned.
    # print("Cleaned Sensored Segments:", cleaned_sensored_segments)
    # print("Cleaned Original Segments:", cleaned_original_segments)

    interpolated_segments_poly = naive_baseline_imputation(
        data=cleaned_censored_segments, method="polynomial", order=2
    )
    visualize_original_interpolated(
        interpolated_segments_poly,
        cleaned_censored_segments,
        cleaned_original_segments,
        thresh_data,
        method_name="Polynomian",
    )

    interpolated_segments_cubic = naive_baseline_imputation(
        data=cleaned_censored_segments, method="cubic"
    )
    visualize_original_interpolated(
        interpolated_segments_cubic,
        cleaned_censored_segments,
        cleaned_original_segments,
        thresh_data,
        method_name="Cubic",
    )

    original_stats = calculate_mean_sd(cleaned_original_segments)
    interpolated_stats_poly = calculate_mean_sd(interpolated_segments_poly)
    interpolated_stats_cubic = calculate_mean_sd(interpolated_segments_cubic)

    interpolated_segments_gp = {}
    for patient_id, segments in cleaned_censored_segments.items():
        interpolated_segments_gp[patient_id] = []
        segment_no = 0
        for segment in segments:
            # Convert segment to DataFrame to use interpolation functions
            segment_df = pd.DataFrame(segment, columns=["cbg"])

            # Check if there are any NaN values in the segment
            if segment_df["cbg"].isna().any():
                # Only train and infer if there are NaN values
                gpr = train_gpr(
                    np.array(segment_df),
                    "gpr_" + str(patient_id) + "_" + str(segment_no),
                    kernel=None,
                )
                indices, interpolated_segment, std = inference_gpr(
                    np.array(segment_df), gpr
                )

                # Create a copy of the original segment and replace NaNs with interpolated values
                indices = indices.flatten()  # Ensure `indices` is a flat array
                full_interpolated_segment = np.array(segment_df).flatten().copy()

                full_interpolated_segment[indices] = np.array(
                    interpolated_segment
                ).flatten()  # Replace NaNs with interpolated values

                interpolated_segments_gp[patient_id].append(full_interpolated_segment)
            else:
                # If no NaN values, add the original segment without modification
                interpolated_segments_gp[patient_id].append(
                    np.array(segment_df).flatten()
                )

            segment_no += 1

    visualize_original_interpolated(
        interpolated_segments_gp,
        cleaned_censored_segments,
        cleaned_original_segments,
        thresh_data,
        method_name="Gaussian Process",
    )

    # # Step 4: Save processed data (if specified)
    # if args.output_dir:
    #     print(f"Saving results to {args.output_dir}...")
    #     # Save data or plots (implement logic in visualization or data_processing)
    #     save_results(args.output_dir, quantile_cut_data, thresholds)
