# -*- coding: utf-8 -*-
"""
Created on Thu 4 Aug 2023 18:45:00 CEST

@author: Gloria del Valle
"""
from preprocessing import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from grid_search import GridSearch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import argparse
import shap
import os


parser = argparse.ArgumentParser(description="Train and evaluate models")
parser.add_argument(
    "--model",
    type=str,
    default="rf",
    help="Model to train and evaluate. Options: rf, xgb",
)
parser.add_argument(
    "--grid-search",
    action="store_true",
    help="Whether to perform grid search or not",
)
parser.add_argument(
    "--shap",
    action="store_true",
    help="Whether to plot shap values or not",
)
parser.add_argument(
    "--plots",
    action="store_true",
    help="Whether to plot extra plots or not",
)
args = parser.parse_args()

# Suppress a specific warning
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data and split into train and test sets
data_loader = DataLoader().split_data()

if args.model == "rf":
    # Initialize and train Random Forest classifier
    model = RandomForestClassifier(
        random_state=42, max_depth=20, max_features="auto", n_estimators=300
    )

    if args.grid_search:
        # Perform grid search
        grid_search = GridSearch(data_loader, model)
        y_pred = grid_search.grid_search()
    else:
        # Train Random Forest classifier
        model.fit(data_loader.X_train_resampled, data_loader.y_train_resampled)
        # Make predictions on the test set
        y_pred = model.predict(data_loader.X_test)

    # Print classification report
    print("Classification report for Random Forest classifier:")
    print(classification_report(data_loader.y_test, y_pred, digits=4))

elif args.model == "xgb":
    # Initialize and train XGBoost classifier
    model = XGBClassifier(
        random_state=42, learning_rate=0.1, max_depth=5, n_estimators=300
    )

    if args.grid_search:
        # Perform grid search
        grid_search = GridSearch(data_loader, model)
        y_pred = grid_search.grid_search()
    else:
        # Train XGBoost classifier
        model.fit(data_loader.X_train, data_loader.y_train)
        # Make predictions on the test set
        y_pred = model.predict(data_loader.X_test)

    # Print classification report
    print("Classification report for XGBoost classifier:")
    print(classification_report(data_loader.y_test, y_pred, digits=4))

# Get probability estimates for the positive class
proba_positive_class = model.predict_proba(data_loader.X_test)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(data_loader.y_test, proba_positive_class)

print(f"ROC AUC score: {roc_auc}")

if args.plots:
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(data_loader.y_test, proba_positive_class)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    # Save plot in folder
    if not os.path.exists("plots/"):
        os.makedirs("plots")
    plt.tight_layout()
    plt.savefig("plots/roc_curve.png")
    plt.close()

    # Calculate confusion matrix
    cm = confusion_matrix(data_loader.y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # Plot new analysis
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Filter data for label 0 (non-fraud)
    label_0_data = data_loader.X[data_loader.y == 0]["amountRatio"]

    num_samples_to_select = 875
    selected_indices = np.random.choice(
        label_0_data.index, num_samples_to_select, replace=False
    )
    balanced_label_0_data = label_0_data.loc[selected_indices]

    # Filter data for label 1 (fraud)
    label_1_data = data_loader.X[data_loader.y == 1]["amountRatio"]

    # Compute the mean value of both distributions
    mean_label_0 = balanced_label_0_data.mean()
    mean_label_1 = label_1_data.mean()

    # Set the colors
    colors = ["#82366a", "#ff3900"]

    # Plot the distribution using Seaborn
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        balanced_label_0_data,
        color=colors[0],
        label="non-fraud (label 0)",
        shade=True,
    )
    sns.kdeplot(label_1_data, color=colors[1], label="fraud (label 1)", shade=True)

    # Plot lines for mean values
    plt.axvline(
        x=mean_label_0, color=colors[0], linestyle="dashed", label="mean (non-fraud)"
    )
    plt.axvline(
        x=mean_label_1, color=colors[1], linestyle="dashed", label="mean (fraud)"
    )

    plt.xlim(0, 0.075)
    plt.title("Distribution of amountRatio")
    plt.xlabel("amountRatio")
    plt.ylabel("density")

    if not os.path.exists("plots/"):
        os.makedirs("plots")
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/distribution_amountRatio.png")
    plt.close()

    # Create a figure with subplots
    plt.figure(figsize=(12, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x="label", y="amountRatio", data=data_loader.data, palette=colors)
    plt.ylim(0, 0.075)
    plt.title("Box plot of amountRatio")
    plt.xlabel("label")
    plt.ylabel("amountRatio")

    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x="label", y="amountRatio", data=data_loader.data, palette=colors)
    plt.ylim(0, 0.5)
    plt.title("Violin plot of amountRatio")
    plt.xlabel("label")
    plt.ylabel("amountRatio")

    # Save plot in folder
    if not os.path.exists("plots/"):
        os.makedirs("plots")
    plt.tight_layout()
    plt.savefig("plots/box_violin_plot.png")
    plt.close()

    # Plot transactionsPerHour
    # Create a figure with subplots
    plt.figure(figsize=(12, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(
        x="label", y="transactionsPerHour", data=data_loader.data, palette=colors
    )
    plt.ylim(0, 40)
    plt.title("Box plot of transactionsPerHour")
    plt.xlabel("label")
    plt.ylabel("transactionsPerHour")

    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(
        x="label", y="transactionsPerHour", data=data_loader.data, palette=colors
    )
    plt.ylim(0, 40)
    plt.title("Violin plot of transactionsPerHour")
    plt.xlabel("label")
    plt.ylabel("transactionsPerHour")

    # Save plot in folder
    if not os.path.exists("plots/"):
        os.makedirs("plots")
    plt.tight_layout()
    plt.savefig("plots/box_violin_plot_transactionsPerHour.png")
    plt.close()

    # Plot density plot
    sns.kdeplot(
        data_loader.X[data_loader.y == 0]["transactionsPerHour"],
        color=colors[0],
        label="non-fraud (label 0)",
        shade=True,
    )
    sns.kdeplot(
        data_loader.X[data_loader.y == 1]["transactionsPerHour"],
        color=colors[1],
        label="fraud (label 1)",
        shade=True,
    )
    plt.title("Density plot of transactionsPerHour")
    plt.xlabel("transactionsPerHour")
    plt.ylabel("density")

    if not os.path.exists("plots/"):
        os.makedirs("plots")
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/density_plot_transactionsPerHour.png")
    plt.close()

if args.shap:
    # Initialize the explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(data_loader.X_test)

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, data_loader.X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig("plots/shap_summary_plot.png")
    plt.close()

    # Plot best 10 features
    shap.summary_plot(shap_values, data_loader.X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("plots/shap_summary_plot_bar.png")
    plt.close()
