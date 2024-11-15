import pandas as pd


def naive_baseline_imputation(data, method="polynomial", order=2):
    interpolated_segments = {}
    method = "polynomial"
    # Assuming 'original_data' is a dictionary with patient data, and each
    # entry is a 1D array of continuous data points
    for patient_id, segments in data.items():
        interpolated_segments[patient_id] = []

        for segment in segments:
            # Convert segment to DataFrame to use interpolation functions
            segment_df = pd.DataFrame(segment, columns=["cbg"])
            if method == "polynomial":
                interpolated_segment = segment_df.interpolate(
                    method="polynomial", order=2
                )
            else:  # Use 'cubic' for cubic spline interpolation
                interpolated_segment = segment_df.interpolate(method="cubic")

            interpolated_segments[patient_id].append(interpolated_segment["cbg"].values)

    return interpolated_segments
