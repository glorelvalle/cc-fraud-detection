# -*- coding: utf-8 -*-
"""
Created on Thu 3 Aug 2023 18:00:00 CEST

@author: Gloria del Valle
"""

import pandas as pd
import numpy as np
from dateutil.parser import parse
import plots as plts
import re
import argparse

# Define the MCC categories
# List stracted from https://stripe.com/es/guides/merchant-category-codes
mcc_categories = {
    1: "Agricultural services",
    range(1500, 3000): "Contracted services",
    range(4000, 4800): "Transportation services",
    range(4800, 5000): "Utility services",
    range(5000, 5600): "Retail outlet services",
    range(5600, 5700): "Clothing stores",
    range(5700, 7300): "Miscellaneous stores",
    range(7300, 8000): "Business services",
    range(8000, 9000): "Professional services and membership organizations",
    range(9000, 10000): "Government services",
}


def is_valid_date(date_str):
    """
    Check if a date string is valid.

    Args:
        date_str (str): The date string to check.
    Returns:
        bool: True if the date string is valid, False otherwise.
    """
    try:
        parse(date_str)
        return True
    except ValueError:
        return False


def check_valid_dates(data):
    """
    Check if the dates in a dataframe are valid.

    Args:
        data (pd.DataFrame): The dataframe to check.
    """
    date_columns = data.select_dtypes(include=["datetime64[ns, UTC]"]).columns
    print(f"Checking {len(date_columns)} columns for valid dates.")

    for col in date_columns:
        is_valid = pd.to_datetime(data[col], errors="coerce").notnull()
        num_nans = data[col].isnull().sum()
        valid_count = is_valid.sum() + num_nans
        total_count = len(data[col])
        print(f"Column '{col}' contains {num_nans} NaNs.")
        print(f"Valid dates: {valid_count}/{total_count}")
        if valid_count == total_count:
            print(f"All dates in column '{col}' are valid.")
        else:
            print(
                f"Column '{col}' contains invalid dates or is not formatted correctly."
            )


def validate_merchant_zip(data):
    """
    Validates the merchantZip column in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to validate.
    Returns:
        data_copy (pd.DataFrame): The DataFrame with the invalid zip codes replaced with NaN.
    """

    # Check if merchantZip column exists in the DataFrame
    if "merchantZip" not in data.columns:
        raise ValueError("merchantZip column not found in the DataFrame")

    # Create a copy of the DataFrame to modify
    data_copy = data.copy()

    print("Validating merchantZip column...")

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        merchant_zip = row["merchantZip"]

        # Skip if merchantZip is NaN
        if pd.isna(merchant_zip):
            continue

        # Validate zip code format
        zip_pattern = r"^[A-Za-z0-9\s]+$"
        if not re.match(zip_pattern, merchant_zip):
            # Invalid zip code: {merchant_zip}, replacing with NaN
            data_copy.at[index, "merchantZip"] = np.nan

    return data_copy


def check_amounts(data):
    """
    Checks the total cash of every accountNumber.

    Args:
        data (pd.DataFrame): The data to check.
    """

    # Group by 'accountNumber' and check if 'availableCash' is constant for each group
    result = data.groupby("accountNumber")["availableCash"].apply(
        lambda x: x.nunique() == 1
    )

    # Print the result
    for account_number, is_constant in result.items():
        status = "constant" if is_constant else "varies"
        print(
            f"The value of 'availableCash' {status} for accountNumber '{account_number}'."
        )


def get_mcc_category(mcc):
    """
    Get the MCC category for a given MCC code.

    Args:
        mcc (int): The MCC code.
    Returns:
        str: The MCC category.
    """

    for mcc_range, category in mcc_categories.items():
        if isinstance(mcc_range, int):
            if mcc == mcc_range:
                return category
        elif isinstance(mcc_range, range):
            if mcc in mcc_range:
                return category
    return "Unknown"


def get_data():
    """
    Read and merge the data.

    Returns:
        data (pd.DataFrame): The final merged data.
    """

    print("Loading data...")

    # Read data
    labels = pd.read_csv(LABELS)
    transactions = pd.read_csv(TRANSACTIONS)

    # Check data
    assert labels["eventId"].duplicated().sum() == 0
    assert (
        transactions.loc[transactions["eventId"].isin(labels["eventId"])].shape[0]
        == labels.shape[0]
    )

    # Merge data
    data = pd.merge(transactions, labels, on="eventId", how="left")

    return data


def transform_data(data):
    """
    Make useful transformations to the data, so the exploratory analysis can be executed.

    Args:
        data (pd.DataFrame): The loaded data.
    Returns:
        data (pd.DataFrame): The transformed data.
    """

    print("Transforming data...")

    # Drop duplicates (there are no duplicates in the data, but just in case)
    data.drop_duplicates(inplace=True)

    # Reformat dates and check validity
    data["reportedTime"] = pd.to_datetime(data["reportedTime"])
    data["transactionTime"] = pd.to_datetime(data["transactionTime"])
    check_valid_dates(data)
    data = validate_merchant_zip(data)

    # Add label: fraud = 1, not fraud = 0
    data["label"] = data["reportedTime"].apply(lambda x: 1 if not pd.isna(x) else 0)

    # Add mcc category
    data["mccCategory"] = data["mcc"].apply(get_mcc_category)

    # Add time difference between reportedTime and transactionTime
    data["timeDiff"] = data["reportedTime"] - data["transactionTime"]
    data["timeDiff"] = data["timeDiff"].apply(lambda x: x.total_seconds())

    # Add day of the week
    data["dayOfWeek"] = data["transactionTime"].apply(lambda x: x.dayofweek)

    # Add hour of the day
    data["hourOfDay"] = data["transactionTime"].apply(lambda x: x.hour)

    # Add month of the year
    data["monthOfYear"] = data["transactionTime"].apply(lambda x: x.month)

    # Add day of the month
    data["dayOfMonth"] = data["transactionTime"].apply(lambda x: x.day)

    return data


def get_summary(data):
    """
    Get a complete summary of data.

    Args:
        data (pd.DataFrame): The data to summarize.
    """

    summary = data.describe(include="all", datetime_is_numeric=True).T
    summary["unique"] = data.nunique()
    summary["missing"] = data.isnull().sum()
    summary["missing_pct"] = summary["missing"] / len(data)
    summary["dtype"] = data.dtypes
    summary = summary[
        [
            "count",
            "unique",
            "missing",
            "missing_pct",
            "top",
            "freq",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "dtype",
        ]
    ]
    print(f"Data shape: {data.shape}")
    print(f"Number of unique events: {data['eventId'].nunique()}")
    print(f"Number of unique accounts: {data['accountNumber'].nunique()}")
    print(f"Number of unique merchants: {data['merchantId'].nunique()}")
    return summary


def get_positives(data):
    """Get just the positive samples.

    Args:
        data (pd.DataFrame): The data to get the positives from.
    Returns:
        positives (pd.DataFrame): The positive samples.
    """
    print(f"Number of positive samples: {data['label'].sum()}")
    return data.loc[data["label"] == 1]


def explore_data(data):
    """
    Explore the data with some useful plots, in order to transform data properly.

    Args:
        data (pd.DataFrame): The data to explore.
    """

    # Generate the custom color palette
    custom_palette = plts.generate_custom_palette()

    print("Creating some useful plots...")

    # Check target distribution
    plts.plot_value_counts(data, "label", custom_palette)

    # Check most frequent countries
    plts.plot_most_frequent_column(data, "merchantCountry", custom_palette)

    # Check most frequent merchant zip codes
    plts.plot_most_frequent_column(data, "merchantZip", custom_palette)

    # Check most frequent mcc categories
    plts.plot_most_frequent_column_labels(data, "mccCategory", custom_palette)

    # Check most frequent day of the week
    plts.plot_most_frequent_column_labels(data, "dayOfWeek", custom_palette)

    # Check most frequent hour of the day
    plts.plot_most_frequent_column_labels(data, "hourOfDay", custom_palette)

    # Check most frequent month of the year
    plts.plot_most_frequent_column_labels(data, "monthOfYear", custom_palette)

    # Check most frequent day of the month
    plts.plot_most_frequent_column_labels(data, "dayOfMonth", custom_palette)

    # Check most frequent entry modes
    plts.plot_most_frequent_column_labels(data, "posEntryMode", custom_palette)

    # Check transaction amount ratio for frauds
    plts.plot_transaction_amount_analysis(data, custom_palette)

    # Check merchantIds for both frauds and non-frauds
    plts.plot_most_common_merchant(data, custom_palette)

    # Check transaction amount distribution for frauds
    plts.plot_transaction_frequency(data, custom_palette)
    plts.plot_transaction_frequency_normalized(data, custom_palette)


def load_data():
    """
    Load and prepare the data.

    Returns:
        data (pd.DataFrame): The preprocessed data.
    """

    # Load data
    data = get_data()

    # Preprocess data
    data = transform_data(data)

    return data


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print more information about the data.",
    )

    # Parse arguments
    args = parser.parse_args()

    return args


def main():
    """Run the EDA."""

    # Load data
    data = load_data()

    # Get summary
    if args.verbose:
        print(get_summary(data))

    # Explore data
    explore_data(data)

    # Get positives dataframe
    if args.verbose:
        print(get_positives(data))

    print("Done!")


if __name__ == "__main__":
    # Set pandas options
    pd.set_option("display.max_columns", None)
    # Parse command line arguments
    args = parse_args()
    # Run EDA
    main()
