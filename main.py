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
    load_data_ohio,
    load_data_glucobench,
)
from src.visualization import (
    visualize_original_interpolated,
)
from models.baselines import naive_baseline_imputation
from models.gp import train_gpr, inference_gpr
from src.bg_statistics import compare_statistics, compute_errors, compute_errors_flatten

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
        (
            _,
            train_data_dict,
            test_data_dict,
            all_data_dict,
            percentage_df,
            percentage_per_patient,
        ) = load_data_ohio(patient_ids, include_test=True)
    elif args.dataset == "iso":
        patient_ids = [str(i) for i in range(102, 224)]
        to_remove = [
            "103",
            "108",
            "110",
            "113",
            "115",
            "116",
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
        all_data_dict, percentage_df, percentage_per_patient = load_data_iso(
            patient_ids
        )

    elif (
        args.dataset == "iglu"
        or args.dataset == "dubosson"
        or args.dataset == "weinstock"
        or args.dataset == "colas"
        or args.dataset == "hall"
        or args.dataset == "T1DEXI_adults"
    ):
        all_data_dict, patient_ids, percentage_df, percentage_per_patient = (
            load_data_glucobench(args.dataset)
        )
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

        all_data_dict = {}
        (
            _,
            train_data_dict,
            test_data_dict,
            all_data_ohio,
            percentage_df_ohio,
            percentage_per_patient_ohio,
        ) = load_data_ohio(patient_id_ohio, include_test=True)
        for patient_id in patient_id_ohio:
            unique_patient_id = f"ohio_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_ohio[patient_id]

        # Update patient ids to be unique for Dubosson dataset
        (
            all_data_dubosson,
            patient_id_dubosson,
            percentage_df_dubosson,
            percentage_per_patient_dubosson,
        ) = load_data_glucobench("dubosson")
        for patient_id in patient_id_dubosson:
            unique_patient_id = f"dubosson_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_dubosson[patient_id]

        # Update patient ids to be unique for Weinstock dataset
        (
            all_data_weinstock,
            patient_id_weinstock,
            percentage_df_weinstock,
            percentage_per_patient_weinstock,
        ) = load_data_glucobench("weinstock")
        for patient_id in patient_id_weinstock:
            unique_patient_id = f"weinstock_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_weinstock[patient_id]

        # Update patient ids to be unique for T1DEXI_adults dataset
        (
            all_data_t1dexi,
            patient_id_t1dexi,
            percentage_df_t1dexi,
            percentage_per_patient_t1dexi,
        ) = load_data_glucobench("T1DEXI_adults")
        for patient_id in patient_id_t1dexi:
            unique_patient_id = f"T1DEXI_adults_{patient_id}"
            all_data_dict[unique_patient_id] = all_data_t1dexi[patient_id]

        # Create a list of all unique patient IDs from all datasets
        patient_ids = (
            [f"ohio_{id}" for id in patient_id_ohio]
            + [f"weinstock_{id}" for id in patient_id_weinstock]
            + [f"dubosson_{id}" for id in patient_id_dubosson]
            + [f"T1DEXI_adults_{id}" for id in patient_id_t1dexi]
        )

        percentage_per_patient = sum(
            [
                percentage_per_patient_ohio,
                percentage_per_patient_weinstock,
                percentage_per_patient_dubosson,
                percentage_per_patient_t1dexi,
            ],
            [],
        )
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
            f"results/{args.dataset}/percentage_above_threshold/percentage_above_threshold_{args.dataset}.csv",
            index=False,
        )

    elif args.dataset == "t2d":
        all_data_dict = {}
        (
            all_data_colas,
            patient_id_colas,
            percentage_df_colas,
            percentage_per_patient_colas,
        ) = load_data_glucobench("colas")

        for patient_id in patient_id_colas:
            unique_patient_id = f"colas_{patient_id}"  # Add a prefix for uniqueness
            all_data_dict[unique_patient_id] = all_data_colas[patient_id]

        (
            all_data_iglu,
            patient_id_iglu,
            percentage_df_iglu,
            percentage_per_patient_iglu,
        ) = load_data_glucobench("iglu")

        for patient_id in patient_id_iglu:
            unique_patient_id = f"iglu_{patient_id}"
            all_data_dict[unique_patient_id] = all_data_iglu[patient_id]

        (
            all_data_hall,
            patient_id_hall,
            percentage_df_hall,
            percentage_per_patient_hall,
        ) = load_data_glucobench("hall")

        for patient_id in patient_id_hall:
            unique_patient_id = f"hall_{patient_id}"
            all_data_dict[unique_patient_id] = all_data_hall[patient_id]

        # Create a list of all unique patient IDs from all datasets
        patient_ids = (
            [f"colas_{id}" for id in patient_id_colas]
            + [f"iglu_{id}" for id in patient_id_iglu]
            + [f"hall_{id}" for id in patient_id_hall]
        )
        percentage_per_patient = sum(
            [
                percentage_per_patient_colas,
                percentage_per_patient_iglu,
                percentage_per_patient_hall,
            ],
            [],
        )
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
            f"results/{args.dataset}/percentage_above_threshold/percentage_above_threshold_{args.dataset}.csv",
            index=False,
        )

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
    for patient_id, filtered_segments in filtered_segments_dict.items():
        start_indices = filtered_segments["start_indices"]
        end_indices = filtered_segments["end_indices"]

        patient_data = all_data_dict[
            patient_id
        ]  

        patient_segments = []
        patient_segment_stats = []

        for i in range(
            len(start_indices)
        ):
            start_idx = start_indices[i]
            end_idx = end_indices[i]

            segment_data = patient_data[start_idx : end_idx + 1]["cbg"]
            segment_length = len(segment_data)

            num_nan_values = np.sum(np.isnan(segment_data))

            if segment_length == 288:
                patient_segments.append(np.array(segment_data))

                patient_segment_stats.append(
                    {"segment_length": segment_length, "num_nan_values": num_nan_values}
                )
        if len(patient_segments) != 0:
            segmented_data_dict[patient_id] = patient_segments
            segment_stats_dict[patient_id] = patient_segment_stats

    original_segments = segmented_data_dict

    # 2. censored segments
    censored_segments = {}
    censored_thresh = {}
    censored_segment_indices = {}
    thresh_data = {}

    for patient_id, segments in original_segments.items():
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
                thresh = np.quantile(filtered_segment, args.percentile)
                print("new thresh = ", thresh)
            censored_thresh[patient_id].append(thresh)
            censored_segment = np.where(segment > thresh, np.nan, segment)
            censored_segment_index = np.where(segment > thresh)

            censored_segments[patient_id].append(censored_segment)
            censored_segment_indices[patient_id].append(censored_segment_index)
            thresh_data[patient_id].append(thresh)

    # Now the cleaned_sensored_segments and cleaned_original_segments dictionaries
    # contain segments without leading or trailing NaN values, and they are aligned.
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
                segment_df = pd.DataFrame(segments[i], columns=["cbg"])
                if segment_df["cbg"].isna().any():
                    ind_all = np.where(~np.isnan(np.array(segment_df)))[0]
                    var = np.std(np.array(segment_df)[ind_all]) ** 2
                    gpr = train_gpr(np.array(segment_df), var=var, kernel=args.kernel)
                    indices, interpolated_segment, std = inference_gpr(
                        censored_segment_indices[patient_id][i][0], gpr
                    )

                    indices = indices.flatten()
                    full_interpolated_segment = np.array(segment_df).flatten().copy()

                    full_interpolated_segment[indices] = np.array(
                        interpolated_segment
                    ).flatten() 
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
        file_name = f"results/{args.dataset}/{args.method}_{args.dataset}_{args.kernel}_{args.percentile}.csv"

        # Save the DataFrame to a CSV file
        df_gp.to_csv(file_name, index=False)

        _ = compute_errors_flatten(
            args,
            original_segments,
            censored_segments,
            interpolated_segments_gp,
        )
        imputation_errors = compute_errors(
            original_segments,
            censored_segments,
            interpolated_segments_gp,
        )

        print(
            imputation_errors[
                [
                    "MSE_GP_Median",
                    "MSE_GP_Q25",
                    "MSE_GP_Q75",
                    "R2_GP_Median",
                    "R2_GP_Q25",
                    "R2_GP_Q75",
                    "MSE_BAS_Median",
                    "MSE_BAS_Q25",
                    "MSE_BAS_Q75",
                    "R2_BAS_Median",
                    "R2_BAS_Q25",
                    "R2_BAS_Q75",
                ]
            ]
        )
        file_name = f"results/{args.dataset}/imputationMetrics/{args.method}_{args.dataset}_{args.kernel}_{args.percentile}_imputationMetrics.csv"

        # Save the DataFrame to a CSV file
        imputation_errors.to_csv(file_name, index=False)
