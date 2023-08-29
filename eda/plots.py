# -*- coding: utf-8 -*-
"""
Created on Thu 3 Aug 2023 18:42:00 CEST

@athor: Gloria del Valle
"""
import pandas as pd
import numpy as np
from dateutil.parser import parse
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os


def generate_custom_palette(save=False):
    """
    Generate a custom color palette for the Featurespace plots.

    Args:
        save (bool): Whether to save the custom palette plot.
    Returns:
        custom_palette (list): The custom color palette.
    """

    # Define the Featurespace colors in HTML format
    wine_berry = "#512142"
    plum = "#82366a"
    mulberry = "#c84683"
    strawberry_red = "#fc1056"
    torch_red = "#fc0e1a"
    vermilion = "#ff3900"
    pumpkin = "#fd7421"
    selective_yellow = "#ffb500"

    # Create the custom color palette
    custom_palette = sns.color_palette(
        [
            wine_berry,
            plum,
            mulberry,
            strawberry_red,
            torch_red,
            vermilion,
            pumpkin,
            selective_yellow,
        ]
    )

    # Plot the custom palette
    sns.palplot(custom_palette)

    # Save the custom palette
    if save:
        plt.tight_layout()
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig("plots/custom_palette.png")
        plt.close()

    return custom_palette


def plot_value_counts(data, column, custom_palette=None):
    """
    Plot the value counts of a categorical column.

    Args:
        data (pd.DataFrame): The data to plot.
        column (str): The name of the column to plot.
    """

    # Get value_counts
    value_counts = data[column].value_counts()

    # Generate a custom color palette if not provided
    if custom_palette is None:
        custom_palette = generate_custom_palette()

    # Create a list of random color indices from the custom palette
    num_unique_values = len(value_counts)
    random_color_indices = random.sample(range(len(custom_palette)), num_unique_values)

    # Create the bar plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=value_counts.index,
        y=value_counts.values,
        palette=[custom_palette[i] for i in random_color_indices],
    )
    plt.xticks(rotation=45, ha="right")

    # Add value labels to the bars
    for index, value in enumerate(value_counts.values):
        ax.text(
            index,
            value + 10,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Customize title and axes
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of {column}", fontsize=14)

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", f"{column}_distribution.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_most_frequent_column(data, column="merchantCountry", custom_palette=None):
    """
    Plot the most frequent values of a categorical column.

    Args:
        data (pd.DataFrame): The data to plot.
        column (str): The name of the column to plot.
    """

    # Filter data to include only rows with label 1
    data_positives = data[data["label"] == 1]

    # Group data by 'merchantCountry' and count occurrences
    country_counts = data_positives[column].value_counts().reset_index(name="count")
    country_counts.columns = [column, "count"]

    # Sort by count in descending order to get most frequent countries
    sorted_country_counts = country_counts.sort_values(by="count", ascending=False)

    # Choose a random color palette if not provided
    if custom_palette is None:
        num_unique_countries = len(sorted_country_counts)
        custom_palette = random.sample(sns.color_palette(), num_unique_countries)

    # Plot the bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=column,
        y="count",
        data=sorted_country_counts,
        palette=custom_palette,
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"{column}", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Most frequent {column} for label 1", fontsize=14)

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", f"most_frequent_{column}.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_most_frequent_column_labels(
    data, column="merchantCountry", custom_palette=None
):
    """
    Plot the most frequent values of a categorical column for label 1 and label 0.

    Args:
        data (pd.DataFrame): The data to plot.
        column (str): The name of the column to plot.
    """

    # Separate data for label 1 and label 0
    data_label_1 = data[data["label"] == 1]
    data_label_0 = data[data["label"] == 0]

    # Group data by 'merchantCountry' and count occurrences for label 1 and label 0
    country_counts_label_1 = (
        data_label_1[column].value_counts().reset_index(name="count")
    )
    country_counts_label_1.columns = [column, "count"]
    sorted_country_counts_label_1 = country_counts_label_1.sort_values(
        by="count", ascending=False
    )

    country_counts_label_0 = (
        data_label_0[column].value_counts().reset_index(name="count")
    )
    country_counts_label_0.columns = [column, "count"]
    sorted_country_counts_label_0 = country_counts_label_0.sort_values(
        by="count", ascending=False
    )

    # Choose a random color palette if not provided
    if custom_palette is None:
        num_unique_countries_label_1 = len(sorted_country_counts_label_1)
        custom_palette_label_1 = random.sample(
            sns.color_palette(), num_unique_countries_label_1
        )

        num_unique_countries_label_0 = len(sorted_country_counts_label_0)
        custom_palette_label_0 = random.sample(
            sns.color_palette(), num_unique_countries_label_0
        )
    else:
        custom_palette_label_1 = custom_palette
        custom_palette_label_0 = custom_palette

    # Create a two-sided plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Plot for label 1
    sns.barplot(
        x=column,
        y="count",
        data=sorted_country_counts_label_1,
        palette=custom_palette_label_1,
        ax=axes[0],
    )
    axes[0].set_title(f"Most frequent {column} for label 1", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=12)

    # Plot for label 0
    sns.barplot(
        x=column,
        y="count",
        data=sorted_country_counts_label_0,
        palette=custom_palette_label_0,
        ax=axes[1],
    )
    axes[1].set_title(f"Most frequent {column} for label 0", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_xlabel(f"{column}", fontsize=12)

    plt.xticks(rotation=45, ha="right")

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", f"most_frequent_{column}_two_sided.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_transaction_amount_analysis(data, custom_palette=None):
    """
    Plot cases where transactionAmount is close to availableCash for positives cases.

    Args:
        data (pd.DataFrame): The data to plot.
    """

    # Filter data to include only positive cases
    data_label_1 = data[data["label"] == 1]

    # Ignore rows where transactionAmount is 0 in this analysis
    data_label_1 = data_label_1[data_label_1["transactionAmount"] != 0]

    if custom_palette is None:
        custom_palette = random.sample(sns.color_palette(), 1)

    # Calculate the distance of each point from the line x=y
    data_label_1["distance"] = np.abs(
        data_label_1["transactionAmount"] - data_label_1["availableCash"]
    )

    # Define a function to map the distance to point size and alpha
    def adjust_point_size(distance):
        max_size = 2500  # Maximum size for points close to the line
        min_size = 10  # Minimum size for points far from the line
        scaling_factor = 200  # Scaling factor to control the rate of size increase
        size = min_size + (max_size - min_size) * np.exp(-distance / scaling_factor)
        alpha = 0.7 + 0.3 * (size - min_size) / (
            max_size - min_size
        )  # Adjust alpha based on size
        return size, alpha

    # Adjust the point size and alpha based on the distance
    data_label_1["point_size"], data_label_1["alpha"] = zip(
        *data_label_1["distance"].apply(adjust_point_size)
    )

    # Create the scatter plot for label 1 using random colors from the custom palette
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        x="transactionAmount",
        y="availableCash",
        data=data_label_1,
        label="available cash ratio",
        color=random.choice(custom_palette),
        s=data_label_1["point_size"],  # Use the adjusted point sizes
        alpha=data_label_1["alpha"].tolist(),  # Use the adjusted alpha values
    )

    # Add the x=y line to the plot
    x_range = np.linspace(
        min(
            data_label_1["transactionAmount"].min(), data_label_1["availableCash"].min()
        ),
        max(
            data_label_1["transactionAmount"].max(), data_label_1["availableCash"].max()
        ),
        100,
    )
    plt.plot(x_range, x_range, color="grey", linestyle="--", linewidth=1, label="x=y")

    # Adjust the axis range
    x_padding = 50  # Add a padding to the x-axis range
    plt.xlim(0, data_label_1["transactionAmount"].max() + x_padding)

    # Customize plot
    plt.xlabel("Transaction amount", fontsize=12)
    plt.ylabel("Available cash", fontsize=12)
    plt.title("Available cash ratio for fraudulent cases", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", "transaction_amount_vs_available_cash.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_transaction_frequency(data, custom_palette=None):
    """
    Plot the time evolution of transactions for the most frequent accounts with label 1.

    Args:
        data (pd.DataFrame): The data to plot.
    """

    # Filter data to include only positive cases
    data_label_1 = data[data["label"] == 1]

    if custom_palette is None:
        custom_palette = random.sample(sns.color_palette(), 1)

    # Get the first 10 most repeated account numbers with label 1
    most_repeated_accounts = (
        data_label_1["accountNumber"].value_counts().nlargest(10).index
    )

    # Create a dictionary to store the colors for each account
    account_colors = {}
    for i, account_number in enumerate(most_repeated_accounts):
        account_colors[account_number] = custom_palette[i % len(custom_palette)]

    # Plot the time evolution of transactions for each of the most repeated account numbers
    plt.figure(figsize=(12, 6))

    for account_number in most_repeated_accounts:
        account_df = data_label_1[data_label_1["accountNumber"] == account_number]
        account_df = account_df.sort_values(by="transactionTime")
        plt.plot(
            account_df["transactionTime"],
            account_df["transactionAmount"],
            marker="o",
            linestyle="-",
            label=f"Account {account_number}",
            color=account_colors[account_number],
        )

    # Customize plot
    plt.xlabel("Transaction time")
    plt.ylabel("Transaction amount")
    plt.title("Time evolution of transactions for the 10 most repeated account number")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(loc="upper right")

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", "transaction_time_evolution.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_transaction_frequency_normalized(data, custom_palette=None):
    """
    Plot the time evolution of transactions for the most frequent accounts with label 1.

    Args:
        data (pd.DataFrame): The data to plot.
    """
    data_label_1 = data[data["label"] == 1].copy()

    if custom_palette is None:
        custom_palette = random.sample(sns.color_palette(), 1)

    # Get the first 10 most repeated account numbers with label 1
    most_repeated_accounts = (
        data_label_1["accountNumber"].value_counts().nlargest(10).index
    )

    # Convert the 'transactionTime' column to datetime format
    data_label_1["transactionTime"] = pd.to_datetime(data_label_1["transactionTime"])
    # Normalize the time by hours
    data_label_1["hourOfDay"] = data_label_1["transactionTime"].dt.hour

    # Create a dictionary to store the colors for each account
    account_colors = {}
    for i, account_number in enumerate(most_repeated_accounts):
        account_colors[account_number] = custom_palette[i % len(custom_palette)]

    # Plot the time evolution of transactions for each of the most repeated account numbers
    plt.figure(figsize=(12, 6))

    for account_number in most_repeated_accounts:
        account_df = data_label_1[data_label_1["accountNumber"] == account_number]
        account_hourly_count = account_df.groupby("hourOfDay")[
            "transactionAmount"
        ].count()
        plt.plot(
            account_hourly_count.index,
            account_hourly_count.values,
            marker="o",
            linestyle="-",
            label=f"Account {account_number}",
            color=account_colors[account_number],
        )
        # Add transactionAmount as text above each data point
        for x, y, amount, date in zip(
            account_hourly_count.index,
            account_hourly_count.values,
            account_df["transactionAmount"],
            account_df["transactionTime"].dt.date,
        ):
            plt.text(
                x,
                y,
                f"{amount:.2f}\n{date}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    plt.xlabel("Hour of Day")
    plt.ylabel("Transaction Count")
    plt.title(
        "Hourly Time Evolution of transactions for the 10 most repeated account numbers (fraud cases)"
    )
    plt.xticks(range(24))
    plt.grid(True)
    plt.legend(loc="upper right")

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", "hourly_transaction_time_evolution.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_most_common_merchant(data, custom_palette=None):
    """
    Plot the 10 most common merchantId values for each label.

    Args:
        data (pd.DataFrame): The data to plot.
    """

    # Filter data for label = 1 and label = 0
    label_1_data = data[data["label"] == 1]
    label_0_data = data[data["label"] == 0]

    if custom_palette is None:
        custom_palette = random.sample(sns.color_palette(), 1)

    # Get the 10 most common merchantId values for each label
    most_common_1 = label_1_data["merchantId"].value_counts().nlargest(10).index
    most_common_0 = label_0_data["merchantId"].value_counts().nlargest(10).index

    # Randomly choose colors from the custom palette
    colors_1 = random.choices(custom_palette, k=len(most_common_1))
    colors_0 = random.choices(custom_palette, k=len(most_common_0))

    # Create separate bar plots with the randomly chosen colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    label_1_data[label_1_data["merchantId"].isin(most_common_1)][
        "merchantId"
    ].value_counts().plot(kind="bar", ax=ax1, color=colors_1)
    ax1.set_title("10 Most Common merchantId for Label 1")
    ax1.set_xlabel("merchantId")
    ax1.set_ylabel("Frequency")

    label_0_data[label_0_data["merchantId"].isin(most_common_0)][
        "merchantId"
    ].value_counts().plot(kind="bar", ax=ax2, color=colors_0)
    ax2.set_title("10 Most Common merchantId for Label 0")
    ax2.set_xlabel("merchantId")
    ax2.set_ylabel("Frequency")

    # Create the "plots" folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the plot into the "plots" folder
    plot_file = os.path.join("plots", "most_common_merchant.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def main():
    generate_custom_palette(save=True)


if __name__ == "__main__":
    main()
