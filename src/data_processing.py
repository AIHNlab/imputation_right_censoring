import pandas as pd
import numpy as np


def fill_intervals_with_nan(
    df, time_col="Tid", interval_col="minutes_elapsed", fill_interval=5
):
    """
    Fills intervals greater than 15 minutes with rows containing NaN values for every 5-minute interval.
    Args:
    - df (pd.DataFrame): The dataframe with the 'Tid' and 'minutes_elapsed' columns.
    - time_col (str): The name of the time column.
    - interval_col (str): The name of the interval column.
    - fill_interval (int): The interval in minutes to insert rows.
    Returns:
    - pd.DataFrame: The modified dataframe with added rows containing NaN values.
    """
    # Convert Tid to datetime
    df[time_col] = pd.to_datetime(df[time_col], format="%d/%m/%Y %H:%M")

    # Calculate time differences in minutes
    df["time_diff"] = df[time_col].diff().dt.total_seconds() / 60

    # Loop through the dataframe to find gaps larger than 15 minutes
    new_rows = []
    for i in range(1, len(df)):
        gap = df.loc[i, "time_diff"]
        if gap > 5:
            # Find the start and end times of the gap
            start_time = df.loc[i - 1, time_col]
            end_time = df.loc[i, time_col]

            # Generate missing timestamps
            missing_times = pd.date_range(
                start=start_time + pd.Timedelta(minutes=fill_interval),
                end=end_time - pd.Timedelta(minutes=fill_interval),
                freq=f"{fill_interval}min",
            )

            # Create rows with NaN values for these missing timestamps
            for time in missing_times:
                new_row = {time_col: time, interval_col: None, "time_diff": None}
                new_rows.append(new_row)

    # Append the missing rows to the original dataframe
    if new_rows:
        new_df = pd.DataFrame(new_rows)

        # Filter out columns that are entirely NaN before concatenation
        new_df = new_df.dropna(axis=1, how="all")

        # Concatenate the original dataframe with the new rows
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values(by=time_col).reset_index(drop=True)
        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=[time_col], keep="first")

        # df['Tid'] = df['Tid'].dt.strftime('%d/%m/%Y %H:%M')

    # Clean up the temporary columns
    df.drop(columns=["time_diff"], inplace=True)

    return df


def find_intervals_more_than_15_and_fill(df):

    # Find rows where intervals are more than 15 minutes
    df_filled = fill_intervals_with_nan(df)

    return df_filled


def get_daily_segments_loc(data_dict: dict, col: str) -> dict:
    """
    Returns a dictionary of dicts where each data series/patient is indexed individually.
    For each index, there is a field 'start_indices', 'end_indices', and 'segment_lengths',
    ensuring segments have a fixed length of 288 time points (1-day interval).
    This version includes NaN values in the segments.
    """
    segments = {}
    max_length = 288  # Maximum length of each segment (1 day)

    for index in data_dict.keys():
        # Get indices for all entries (including NaN values) in the specified column
        all_indices = np.asarray(data_dict[index].index)

        # Initialize start and end indices for the segments
        start_indices = []
        end_indices = []

        # Create segments with a fixed length of 288 time points
        start = all_indices[0]
        for i in range(0, len(all_indices), max_length):
            end = min(
                start + max_length - 1, all_indices[-1]
            )  # Ensure the end doesn't exceed the last index
            start_indices.append(start)
            end_indices.append(end)
            start = end + 1  # Update the start for the next segment

        # Compute segment lengths
        lengths = (np.array(end_indices) - np.array(start_indices)) + 1

        segments[index] = {
            "start_indices": np.array(start_indices),
            "end_indices": np.array(end_indices),
            "segment_lengths": lengths,
        }

    return segments


def check_and_filter_nan_segments(
    data: pd.DataFrame, segments_loc: pd.DataFrame, col: str, max_nan_length: int = 18
) -> dict:
    """
    Checks the segments for continuous NaN values and filters out segments with more than the allowed number of consecutive NaNs.

    Args:
    - data_dict: Dictionary of DataFrames, where each key is a patient and each value is a DataFrame.
    - col: The column to check for NaN values.
    - max_nan_length: The maximum allowed consecutive NaN values for a segment.

    Returns:
    - Filtered segments dictionary with only segments that have fewer than max_nan_length consecutive NaN values.
    """
    filtered_segments = {}

    # Initialize start and end indices for each segment
    start_indices = []
    end_indices = []
    nan_lengths = []  # To store the length of consecutive NaNs in each segment

    for segment_start, segment_end in zip(
        segments_loc["start_indices"], segments_loc["end_indices"]
    ):
        # Slice the segment from the DataFrame
        segment_data = data.iloc[segment_start : segment_end + 1][col]

        # Find continuous NaN sequences within this segment
        # Generate a mask of NaN values (True for NaN)
        is_nan = segment_data.isna()

        # Use a counter for consecutive NaNs and identify sequences
        nan_sequences = is_nan.groupby((~is_nan).cumsum()).cumsum()

        # Find the maximum consecutive NaN sequence length in this segment
        max_nan_in_segment = nan_sequences.max() if not nan_sequences.empty else 0
        # print("max_nan_in_segment = ", max_nan_in_segment)
        # Count the number of NaN values in the segment
        num_nan_values = np.sum(np.isnan(segment_data))

        num_nan_values = np.isnan(segment_data).sum()
        nan_ratio = num_nan_values / len(segment_data)

        # Check if the segment has fewer than the max allowed consecutive NaNs
        if max_nan_in_segment < max_nan_length and nan_ratio < 0.5:
            # If valid, store the start and end indices, and the segment's length
            start_indices.append(segment_start)
            end_indices.append(segment_end)
            nan_lengths.append(max_nan_in_segment)
        # else:
        #     print("too many Nan's")

    filtered_segments = {
        "start_indices": start_indices,
        "end_indices": end_indices,
        "nan_lengths": nan_lengths,
    }

    return filtered_segments
