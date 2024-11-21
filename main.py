import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy

from src.data_processing import (
    load_data_ohio,
    load_data_iso,
    load_data_cap,
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
from src.bg_statistics import calculate_segment_statistics, compare_statistics

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
        "--kernel",
        type=str,
        default="matern32",
        help="Kernel for gp (default matern32), options: matern32, matern52.",
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

        for patient_id in patient_ids:
            all_data = load_data_iso(patient_id)
            all_data_dict[patient_id] = all_data

        patient_ids_cap = [
            "0620",
            "627",
            "0639",
            "0652",
            "0675",
            "CGM_007",
            "CGM_008",
            "CGM_009",
            "CGM_011",
            "CGM_013",
            "CGM_014",
            "CGM_017",
            "CGM_018",
            "CGM_020",
            "CGM_021",
            "CGM_022",
            "CGM_023",
            "CGM_024",
            "CGM_025",
        ]
        for patient_id in patient_ids_cap:
            all_data = load_data_cap(patient_id)
            all_data_dict[patient_id] = all_data

        patient_ids = patient_ids + patient_ids_cap

    print("all data = ", len(all_data_dict))
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
        df_poly = compare_statistics(
            cleaned_original_segments,
            cleaned_censored_segments,
            interpolated_segments_poly,
        )
        if args.dataset == "ohio":
            numeric_cols = df_poly.select_dtypes(include="number")
            df_poly[numeric_cols.columns] = (
                df_poly.select_dtypes(include="number") * 0.0555
            )
        print(df_poly)
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
        df_cubic = compare_statistics(
            cleaned_original_segments,
            cleaned_censored_segments,
            interpolated_segments_cubic,
        )
        if args.dataset == "ohio":
            numeric_cols = df_cubic.select_dtypes(include="number")
            df_cubic[numeric_cols.columns] = (
                df_cubic.select_dtypes(include="number") * 0.0555
            )
        print(df_cubic)
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
        df_ffill = compare_statistics(
            cleaned_original_segments,
            cleaned_censored_segments,
            interpolated_segments_ffill,
        )
        if args.dataset == "ohio":
            numeric_cols = df_ffill.select_dtypes(include="number")
            df_ffill[numeric_cols.columns] = (
                df_ffill.select_dtypes(include="number") * 0.0555
            )
        print(df_ffill)
    elif args.method == "gp":
        interpolated_segments_gp = {}
        for patient_id, segments in cleaned_censored_segments.items():
            interpolated_segments_gp[patient_id] = []
            all_segments = np.concatenate(segments)
            ind_all = np.where(~np.isnan(all_segments))[0]
            all_segments_actual = all_segments[ind_all]
            var = np.std(all_segments_actual) ** 2
            mean = np.mean(all_segments_actual)
            for segment in segments:
                # Convert segment to DataFrame to use interpolation functions
                segment_df = pd.DataFrame(segment, columns=["cbg"])
                # Check if there are any NaN values in the segment
                if segment_df["cbg"].isna().any():
                    # Only train and infer if there are NaN values
                    gpr = train_gpr(np.array(segment_df), kernel=args.kernel, var=var)
                    indices, interpolated_segment, std = inference_gpr(
                        np.array(segment_df), gpr
                    )

                    # Create a copy of the original segment and replace NaNs with interpolated values
                    indices = indices.flatten()  # Ensure `indices` is a flat array
                    full_interpolated_segment = np.array(segment_df).flatten().copy()

                    full_interpolated_segment[indices] = np.array(
                        interpolated_segment
                    ).flatten()  # Replace NaNs with interpolated values
                    interpolated_segments_gp[patient_id].append(
                        full_interpolated_segment
                    )

                    # plt.figure()
                    # plt.plot(full_interpolated_segment)
                    # plt.plot(np.array(segment_df).flatten().copy())

                else:
                    # If no NaN values, add the original segment without modification
                    interpolated_segments_gp[patient_id].append(
                        np.array(segment_df).flatten()
                    )

        # break
        visualize_original_interpolated(
            interpolated_segments_gp,
            cleaned_censored_segments,
            cleaned_original_segments,
            thresh_data,
            method_name="Gaussian Process",
        )
        df_gp = compare_statistics(
            cleaned_original_segments,
            cleaned_censored_segments,
            interpolated_segments_gp,
        )
        if args.dataset == "ohio":
            numeric_cols = df_gp.select_dtypes(include="number")
            df_gp[numeric_cols.columns] = df_gp.select_dtypes(include="number") * 0.0555
        print(df_gp)
