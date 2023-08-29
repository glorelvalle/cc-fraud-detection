# -*- coding: utf-8 -*-
"""
Created on Thu 4 Aug 2023 17:29:00 CEST

@author: Gloria del Valle
"""
import sys
import os
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

eda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eda"))
sys.path.append(eda_path)

from eda import get_data, get_mcc_category, validate_merchant_zip

pd.set_option("display.max_columns", None)

SEED = 1234


class DataLoader:
    def __init__(self):
        # Load data from csv

        self.data = get_data()
        print(self.data.shape)

        # Preprocess data
        self.data = self.preprocessing()
        print(self.data.shape)

    def compute_frequency(self, label, column):
        """
        Compute frequency of a label in a column.

        Args:
            label (int): label to compute frequency for
            column (str): column to compute frequency for

        Returns:
            frequency (list): list of top 3 most frequent values for the label
        """
        frequency = (
            self.data[self.data["label"] == label][column].value_counts().head(3).index
        )
        return frequency

    def preprocessing(self):
        """
        Preprocess data
        """
        print("Preprocessing data...")

        # Reformat dates
        self.data["reportedTime"] = pd.to_datetime(self.data["reportedTime"])
        self.data["transactionTime"] = pd.to_datetime(self.data["transactionTime"])

        # Validate zip code and fix errors
        self.data = validate_merchant_zip(self.data)

        # Add mcc category
        self.data["mccCategory"] = self.data["mcc"].apply(lambda x: get_mcc_category(x))

        # Add day of the week
        self.data["dayOfWeek"] = self.data["transactionTime"].apply(
            lambda x: x.dayofweek
        )

        # Add hour of the day
        self.data["hourOfDay"] = self.data["transactionTime"].apply(lambda x: x.hour)

        # Add month of the year
        self.data["monthOfYear"] = self.data["transactionTime"].apply(lambda x: x.month)

        # Add day of the month
        self.data["dayOfMonth"] = self.data["transactionTime"].apply(lambda x: x.day)

        # Group by 'accountNumber' and 'transactionTime' (date) and count the transactions per day for each account
        self.data["transactionsPerDay"] = self.data.groupby(
            ["accountNumber", self.data["transactionTime"].dt.date]
        )["transactionAmount"].transform("count")

        # Group by 'accountNumber', 'transactionTime' (date) and 'hour' and count the transactions per hour for each account
        self.data["transactionsPerHour"] = self.data.groupby(
            [
                "accountNumber",
                self.data["transactionTime"].dt.date,
                self.data["transactionTime"].dt.hour,
            ]
        )["transactionAmount"].transform("count")

        # Add ratio of transaction amount to available cash (close to 1 means high risk but not necessarily fraud, we saw that in the EDA)
        self.data["amountRatio"] = (
            self.data["transactionAmount"] / self.data["availableCash"]
        )

        # Add label: fraud = 1, not fraud = 0
        self.data["label"] = self.data["reportedTime"].apply(
            lambda x: 1 if not pd.isna(x) else 0
        )

        # Calculate frequency of each merchant for label 1 transactions
        label_1_merchants = self.compute_frequency(1, "merchantId")

        # Calculate frequency of each merchant for label 0 transactions
        label_0_merchants = self.compute_frequency(0, "merchantId")

        # Calculate frequency of each merchantZip for label 1 transactions
        label_1_merchants_zip = self.compute_frequency(1, "merchantZip")

        # Calculate frequency of each merchantZip for label 0 transactions
        label_0_merchants_zip = self.compute_frequency(0, "merchantZip")

        # Calculate frequency of each merchantCountry for label 1 transactions
        label_1_merchants_country = self.compute_frequency(1, "merchantCountry")

        # Calculate frequency of each merchantCountry for label 0 transactions
        label_0_merchants_country = self.compute_frequency(0, "merchantCountry")

        # Create variable to indicate if the merchant is in the top 3 most frequent merchants for label 1
        self.data["isMostFrequentMerchantId_1"] = self.data["merchantId"].apply(
            lambda x: int(x in label_1_merchants)
        )

        # Create variable to indicate if the merchant is in the top 3 most frequent merchants for label 0
        self.data["isMostFrequentMerchantId_0"] = self.data["merchantId"].apply(
            lambda x: int(x in label_0_merchants)
        )

        # Create variable to indicate if the merchantZip is in the top 3 most frequent merchantZips for label 1
        self.data["isMostFrequentMerchantZip_1"] = self.data["merchantZip"].apply(
            lambda x: int(x in label_1_merchants_zip)
        )

        # Create variable to indicate if the merchantZip is in the top 3 most frequent merchantZips for label 0
        self.data["isMostFrequentMerchantZip_0"] = self.data["merchantZip"].apply(
            lambda x: int(x in label_0_merchants_zip)
        )

        # Create variable to indicate if the merchantCountry is in the top 3 most frequent merchantCountries for label 1
        self.data["isMostFrequentMerchantCountry_1"] = self.data[
            "merchantCountry"
        ].apply(lambda x: int(x in label_1_merchants_country))

        # Create variable to indicate if the merchantCountry is in the top 3 most frequent merchantCountries for label 0
        self.data["isMostFrequentMerchantCountry_0"] = self.data[
            "merchantCountry"
        ].apply(lambda x: int(x in label_0_merchants_country))

        # Drop columns
        self.data.drop(
            columns=[
                "eventId",
                "accountNumber",
                "transactionTime",
                "mcc",
                "availableCash",
                "reportedTime",
                "merchantId",
                "merchantZip",
                "merchantCountry",
            ],
            inplace=True,
            axis=1,
        )
        print(self.data.shape)

        return self.data

    def split_data(self):
        """
        Split data into train and test sets
        """
        # Create label encoder object
        label_encoder = LabelEncoder()

        # Encode labels in column 'label'
        self.data["mccCategory"] = label_encoder.fit_transform(self.data["mccCategory"])

        # One hot encode categorical variables
        one_hot_columns = [
            "mccCategory",
            "posEntryMode",
            # "monthOfYear",
            # "dayOfMonth",
            # "dayOfWeek",
            # "hourOfDay",
        ]

        # One hot encode categorical variables
        self.data = pd.get_dummies(self.data, columns=one_hot_columns)

        # Divide into features and label
        X = self.data.drop(columns=["label"])
        y = self.data["label"]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        # Undersampling the negative class (non-fraudulent transactions)
        # rus = RandomUnderSampler(random_state=SEED, sampling_strategy=0.02)
        # X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

        # Trully important to create synthetic data for the positive class (fraudulent transactions)
        smote = SMOTE(random_state=SEED, sampling_strategy=0.5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Print class distribution before and after SMOTE
        print("Class distribution before SMOTE:")  # and RUS
        print(y.value_counts())

        print("Class distribution after SMOTE (train):")
        print(pd.Series(y_train_resampled).value_counts())

        print("Class distribution after SMOTE (test):")
        print(pd.Series(y_test).value_counts())

        # Save data
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_resampled = X_train_resampled
        self.y_train_resampled = y_train_resampled

        return self
