import argparse
import numpy as np
import pandas as pd
import os


from src.data_processing import (
    get_daily_segments_loc,
    check_and_filter_nan_segments,
)
from src.data_loaders import (
    load_data_iso,
    load_data_cap,
    load_data_cap1,
    load_data_ohio,
    load_data_glucobench,
)
from src.visualization import (
    visualize_original_interpolated,
)
from models.baselines import naive_baseline_imputation
from models.gp import train_gpr, inference_gpr, train_gpr_sklearn, inference_gpr_sklearn
from src.bg_statistics import compare_statistics

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
        patient_ids = [str(i) for i in range(102, 224)]
        to_remove = [
            "103",
            "108",
            "110",
            "113",
            "115",
            "116",  # Not sure yet
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

        patient_ids_cap = ["620"]
        patient_ids_cap1 = [
            "627",
            "639",
            "652",
            "675",
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

        for patient_id in patient_ids_cap1:
            all_data = load_data_cap1(patient_id)
            all_data_dict[patient_id] = all_data

        patient_ids = patient_ids + patient_ids_cap + patient_ids_cap1

    elif (
        args.dataset == "iglu"
        or args.dataset == "dubosson"
        or args.dataset == "weinstock"
        or args.dataset == "colas"
        or args.dataset == "hall"
        or args.dataset == "T1DEXI_adults"
    ):
        all_data_dict, patient_ids = load_data_glucobench(args.dataset)
    elif args.dataset == "t1d":
        patient_id_ohio = [
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
        # Update patient ids to be unique for Ohio dataset
        for patient_id in patient_id_ohio:
            unique_patient_id = f"ohio_{patient_id}"  # Add a prefix to make it unique
            _, train_data, test_data, all_data = load_data_ohio(
                patient_id, include_test=True
            )

            train_data_dict[unique_patient_id] = train_data
            test_data_dict[unique_patient_id] = test_data
            all_data_dict[unique_patient_id] = all_data

        # Update patient ids to be unique for Dubosson dataset
        all_data_dubosson, patient_id_dubosson = load_data_glucobench("dubosson")
        for patient_id in patient_id_dubosson:
            unique_patient_id = f"dubosson_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_dubosson[patient_id]

        # Update patient ids to be unique for Weinstock dataset
        all_data_weinstock, patient_id_weinstock = load_data_glucobench("weinstock")
        for patient_id in patient_id_weinstock:
            unique_patient_id = f"weinstock_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_weinstock[patient_id]

        # Update patient ids to be unique for T1DEXI_adults dataset
        all_data_t1dexi, patient_id_t1dexi = load_data_glucobench("T1DEXI_adults")
        for patient_id in patient_id_t1dexi:
            unique_patient_id = (
                f"T1DEXI_adults_{patient_id}"  # Add a prefix for uniqueness
            )
            all_data_dict[unique_patient_id] = all_data_t1dexi[patient_id]

        # Create a list of all unique patient IDs from all datasets
        patient_ids = (
            [f"ohio_{id}" for id in patient_id_ohio]
            + [f"weinstock_{id}" for id in patient_id_weinstock]
            + [f"dubosson_{id}" for id in patient_id_dubosson]
            + [f"T1DEXI_adults_{id}" for id in patient_id_t1dexi]
        )
    elif args.dataset == "t2d":
        all_data_dict = {}
        all_data_colas, patient_id_colas = load_data_glucobench("colas")
        print(patient_id_colas)
        for patient_id in patient_id_colas:
            unique_patient_id = f"colas_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_colas[patient_id]

        all_data_iglu, patient_id_iglu = load_data_glucobench("iglu")
        print(patient_id_iglu)
        for patient_id in patient_id_iglu:
            unique_patient_id = f"iglu_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_iglu[patient_id]

        all_data_hall, patient_id_hall = load_data_glucobench("hall")
        print(patient_id_hall)
        for patient_id in patient_id_hall:
            unique_patient_id = f"hall_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_hall[patient_id]

        # Create a list of all unique patient IDs from all datasets
        patient_ids = (
            [f"colas_{id}" for id in patient_id_colas]
            + [f"iglu_{id}" for id in patient_id_iglu]
            + [f"hall_{id}" for id in patient_id_hall]
        )

    # print("patient ids = ", patient_ids)
    # print("all data = ", all_data_dict)

    print("all data = ", len(all_data_dict))
    # 1. create continues
    cont_segments_dict = {}
    filtered_segments_dict = {}

    for patient_id in patient_ids:
        patient_data = all_data_dict
        cont_segments_loc = get_daily_segments_loc(
            {patient_id: patient_data[patient_id]}, "cbg"
        )
        cont_segments_dict[patient_id] = cont_segments_loc[patient_id]
        filtered_segments = check_and_filter_nan_segments(
            patient_data[patient_id], cont_segments_dict[patient_id], "cbg"
        )
        filtered_segments_dict[patient_id] = filtered_segments

    # Initialize a new dictionary to store the segmented data per patient
    segmented_data_dict = {}
    segment_stats_dict = {}
    # Loop over each patient and extract their segments based on the filtered indices
    for patient_id, filtered_segments in filtered_segments_dict.items():
        # Get the start indices for the patient's filtered segments
        start_indices = filtered_segments["start_indices"]
        end_indices = filtered_segments["end_indices"]

        # Extract the corresponding data for each segment
        patient_data = all_data_dict[
            patient_id
        ]  # Assuming this is the full data for the patient

        # Initialize a list to store the segmented data for the patient
        patient_segments = []
        patient_segment_stats = []

        # Loop over the start indices to slice the data for each segment
        for i in range(
            len(start_indices)
        ):  # Loop until the second-to-last index (since we're using pairs)
            start_idx = start_indices[i]
            end_idx = end_indices[i]

            # Extract the segment for this pair of indices
            segment_data = patient_data[start_idx : end_idx + 1]["cbg"]

            # Calculate the length of the segment
            segment_length = len(segment_data)

            # Count the number of NaN values in the segment
            num_nan_values = np.sum(np.isnan(segment_data))

            if segment_length == 288:
                # print(f"The segment has {num_nan_values} NaN values.")
                # Append the segment to the patient's list of segments
                patient_segments.append(np.array(segment_data))

                # Store the statistics (length and NaN count) for this segment
                patient_segment_stats.append(
                    {"segment_length": segment_length, "num_nan_values": num_nan_values}
                )
        if len(patient_segments) != 0:
            # Store the segmented data in the dictionary, using the patient_id as the key
            segmented_data_dict[patient_id] = patient_segments
            segment_stats_dict[patient_id] = patient_segment_stats

    original_segments = segmented_data_dict

    # 2. censored segments
    # Dictionary for storing the interpolated data for each patient
    censored_segments = {}
    censored_thresh = {}
    censored_segment_indices = {}
    thresh_data = {}

    # Assuming 'original_data' is a dictionary with patient data, and each entry is a 1D array of continuous data points
    for patient_id, segments in original_segments.items():
        # Store interpolated segments for the current patient
        censored_segments[patient_id] = []
        censored_thresh[patient_id] = []
        censored_segment_indices[patient_id] = []
        thresh_data[patient_id] = []

        for segment in segments:
            if len(segment) != 288:
                print("segment len = ", len(segment))
            thresh = np.quantile(segment[~np.isnan(segment)], args.percentile)
            if thresh == np.max(segment[~np.isnan(segment)]):
                print("maximum thresh")
                filtered_segment = segment[~np.isnan(segment)]  # Remove NaN values
                filtered_segment = filtered_segment[
                    filtered_segment < np.max(filtered_segment)
                ]  # Exclude the max value
                # Compute the quantile on the filtered dataset
                thresh = np.quantile(filtered_segment, args.percentile)
                print("new thresh = ", thresh)
            # pint("thresh = ", thresh)
            censored_thresh[patient_id].append(thresh)
            censored_segment = np.where(segment > thresh, np.nan, segment)
            censored_segment_index = np.where(segment > thresh)

            censored_segments[patient_id].append(censored_segment)
            censored_segment_indices[patient_id].append(censored_segment_index)
            thresh_data[patient_id].append(thresh)

    # print("All censored segments:", censored_segments)

    # Now the cleaned_sensored_segments and cleaned_original_segments dictionaries
    # contain segments without leading or trailing NaN values, and they are aligned.
    # print("Cleaned Sensored Segments:", cleaned_sensored_segments)
    # print("Cleaned Original Segments:", cleaned_original_segments)
    if args.method == "polynomial":
        interpolated_segments_poly = naive_baseline_imputation(
            data=censored_segments, method="polynomial", order=2
        )
        visualize_original_interpolated(
            args,
            interpolated_segments_poly,
            censored_segments,
            original_segments,
            thresh_data,
            method_name="Polynomian",
        )
        df_poly = compare_statistics(
            original_segments,
            censored_segments,
            interpolated_segments_poly,
        )
        print(df_poly)
    elif args.method == "cubic":
        interpolated_segments_cubic = naive_baseline_imputation(
            data=censored_segments, method="cubic"
        )
        visualize_original_interpolated(
            args,
            interpolated_segments_cubic,
            censored_segments,
            original_segments,
            thresh_data,
            method_name="Cubic",
        )
        df_cubic = compare_statistics(
            original_segments,
            censored_segments,
            interpolated_segments_cubic,
        )
        print(df_cubic)
    elif args.method == "ffill":
        interpolated_segments_ffill = naive_baseline_imputation(
            data=censored_segments, method="ffill"
        )
        visualize_original_interpolated(
            args,
            interpolated_segments_ffill,
            censored_segments,
            original_segments,
            thresh_data,
            method_name="Ffill",
        )
        df_ffill = compare_statistics(
            original_segments,
            censored_segments,
            interpolated_segments_ffill,
        )
        print(df_ffill)
    elif args.method == "gp":
        interpolated_segments_gp = {}
        for patient_id, segments in censored_segments.items():
            interpolated_segments_gp[patient_id] = []
            for i in range(len(segments)):
                # Convert segment to DataFrame to use interpolation functions
                segment_df = pd.DataFrame(segments[i], columns=["cbg"])
                # Check if there are any NaN values in the segment
                if segment_df["cbg"].isna().any():
                    ind_all = np.where(~np.isnan(np.array(segment_df)))[0]
                    var = np.std(np.array(segment_df)[ind_all]) ** 2
                    # Only train and infer if there are NaN values
                    gpr = train_gpr_sklearn(
                        np.array(segment_df), var=var, kernel=args.kernel
                    )
                    indices, interpolated_segment, std = inference_gpr_sklearn(
                        censored_segment_indices[patient_id][i][0], gpr
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

        visualize_original_interpolated(
            args,
            interpolated_segments_gp,
            censored_segments,
            original_segments,
            thresh_data,
            method_name="Gaussian Process",
        )
        df_gp = compare_statistics(
            original_segments,
            censored_segments,
            interpolated_segments_gp,
        )
        print(df_gp)
        os.makedirs(
            f"results/{args.dataset}",
            exist_ok=True,
        )
        # Construct the file name
        # file_name = f"results/{args.dataset}/{args.method}_{args.dataset}_{args.kernel}_{args.percentile}.csv"

        # # Save the DataFrame to a CSV file
        # df_gp.to_csv(file_name, index=False)
