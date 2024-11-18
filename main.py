import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing import (
    load_data_ohio,
    load_data_iso,
    apply_quantile_cut,
    get_continuous_segments_loc,
    clean_segment_and_align,
    get_nan_segments_loc,
    remove_small_data_segments,
    resize_segments_uniform,
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
        "--dataset",
        type=str,
        default="ohio",
        help="Name of Dataset (e.g., ohio, iso,..).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.8,
        help="Quantile threshold for cutoff (default: 0.8).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="polynomial",
        help="Method to test (default polynomial).",
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
    if args.dataset == "ohio":
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
        patient_ids = ["540"]
        train_data_dict = {}
        test_data_dict = {}
        all_data_dict = {}
        for patient_id in patient_ids:
            _, train_data, test_data, all_data = load_data_ohio(
                patient_id, include_test=True
            )
            train_data_dict[patient_id] = train_data
            test_data_dict[patient_id] = test_data
            all_data_dict[patient_id] = all_data
    elif args.dataset == "iso":
        patient_ids = [str(i) for i in range(102, 223)]
        to_remove = [
            "103",
            "108",
            "110",
            "113",
            "115",
            "122",
            "126",
            "127",
            "128",
            "130",
            "134",
            "139",
            "143",
            "144",
            "146",
            "149",
            "152",
            "155",
            "156",
            "164",
            "174",
            "175",
            "176",
            "177",
            "180",
            "185",
            "186",
            "190",
            "193",
            "194",
            "197",
            "203",
            "206",
            "211",
            "212",
            "220",
        ]
        patient_ids = list(set(patient_ids) - set(to_remove))
        all_data_dict = {}
        patient_ids = ["161"]
        for patient_id in patient_ids:
            all_data = load_data_iso(patient_id)
            all_data_dict[patient_id] = all_data

    # Step 2: Apply percentile cut
    print(f"Applying {args.percentile * 100}% percentile cutoff...")
    quantile_cut_data, thresh_data = apply_quantile_cut(
        all_data_dict, nan_above_quantile=args.percentile
    )

    # Step 3: Visualize results
    print("Generating plots...")
    plot_patient_data_with_quantile(
        all_data_dict, quantile_cut_data, quantile_cutoff=args.percentile
    )

    # 1. create continues
    cont_segments_dict = {}
    nan_segments_dict = {}

    for patient_id in patient_ids:
        patient_data = all_data_dict
        cont_segments_loc = get_continuous_segments_loc(
            {patient_id: patient_data[patient_id]}, "cbg"
        )
        nan_segments_loc = get_nan_segments_loc(
            {patient_id: patient_data[patient_id]}, "cbg"
        )
        cont_segments_dict[patient_id] = cont_segments_loc[patient_id]
        nan_segments_dict[patient_id] = nan_segments_loc[patient_id]

    kept_cont_segments, kept_cont_segments_df = remove_small_data_segments(
        cont_segments_dict, nan_segments_dict
    )
    original_segments = resize_segments_uniform(kept_cont_segments, all_data_dict)
    resized_cont_segments_df = calculate_segment_statistics(
        original_segments, is_resized=True
    )

    # 2. censored segments
    # Dictionary for storing the interpolated data for each patient
    censored_segments = {}

    # Assuming 'original_data' is a dictionary with patient data, and each entry is a 1D array of continuous data points
    for patient_id, segments in original_segments.items():
        # Store interpolated segments for the current patient
        censored_segments[patient_id] = []

        for segment in segments:
            # Extract the segment data between the start and end indices
            has_nan = np.isnan(segment).any()
            # print("Contains NaN values:", has_nan)
            censored_segment = np.where(
                segment > thresh_data[patient_id], np.nan, segment
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
    original_stats = calculate_mean_sd(cleaned_original_segments)
    if args.method == "polynomial":
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
        interpolated_stats_poly = calculate_mean_sd(interpolated_segments_poly)
        comparison = compare_statistics(
            original_stats, interpolated_stats_poly, "Polynomial"
        )
        print(comparison)
    elif args.method == "cubic":
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
        interpolated_stats_cubic = calculate_mean_sd(interpolated_segments_cubic)
        comparison = compare_statistics(
            original_stats, interpolated_stats_cubic, "Cubic"
        )
        print(comparison)
    elif args.method == "ffill":
        interpolated_segments_ffill = naive_baseline_imputation(
            data=cleaned_censored_segments, method="ffill"
        )
        visualize_original_interpolated(
            interpolated_segments_ffill,
            cleaned_censored_segments,
            cleaned_original_segments,
            thresh_data,
            method_name="Ffill",
        )
        interpolated_stats_ffill = calculate_mean_sd(interpolated_segments_ffill)
        comparison = compare_statistics(
            original_stats, interpolated_stats_ffill, "Ffill"
        )
        print(comparison)
    elif args.method == "gp":
        interpolated_segments_gp = {}
        for patient_id, segments in cleaned_censored_segments.items():
            interpolated_segments_gp[patient_id] = []
            segment_no = 0
            for segment in segments:
                # Convert segment to DataFrame to use interpolation functions
                segment_df = pd.DataFrame(segment, columns=["cbg"])

                # Check if there are any NaN values in the segment
                if segment_df["cbg"].isna().any():
                    # print("segment_df = ", segment_df)
                    # Only train and infer if there are NaN values
                    gpr = train_gpr(
                        np.array(segment_df),
                        "gpr_" + str(patient_id) + "_" + str(segment_no),
                        kernel=None,
                    )
                    indices, interpolated_segment, std = inference_gpr(
                        np.array(segment_df), gpr
                    )
                    # print("std = ", std)
                    # Create a copy of the original segment and replace NaNs with interpolated values
                    indices = indices.flatten()  # Ensure `indices` is a flat array
                    full_interpolated_segment = np.array(segment_df).flatten().copy()
                    # print("indices = ", indices)
                    # print("interpolated_segment = ", interpolated_segment)
                    full_interpolated_segment[indices] = np.array(
                        interpolated_segment
                    ).flatten()  # Replace NaNs with interpolated values
                    # print("full_interpolated_segment = ", full_interpolated_segment)
                    interpolated_segments_gp[patient_id].append(
                        full_interpolated_segment
                    )
                    # plt.figure()
                    # plt.plot(full_interpolated_segment)
                    # plt.plot(np.array(segment_df).flatten().copy())
                    # plt.show()

                else:
                    # If no NaN values, add the original segment without modification
                    interpolated_segments_gp[patient_id].append(
                        np.array(segment_df).flatten()
                    )

                segment_no += 1
        # break
        visualize_original_interpolated(
            interpolated_segments_gp,
            cleaned_censored_segments,
            cleaned_original_segments,
            thresh_data,
            method_name="Gaussian Process",
        )
        interpolated_stats_gp = calculate_mean_sd(interpolated_segments_gp)
        compare_statistics(original_stats, interpolated_stats_gp, "GP")

# # Step 4: Save processed data (if specified)
# if args.output_dir:
#     print(f"Saving results to {args.output_dir}...")
#     # Save data or plots (implement logic in visualization or data_processing)
#     save_results(args.output_dir, quantile_cut_data, thresholds)
