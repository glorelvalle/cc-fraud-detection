# Credit Card Fraud Detection problem
This repository contains a solution for credit card fraud detection using traditional ML methods.

## Problem Statement
The bank is experiencing credit card fraud where fraudsters are making unauthorized transactions using the bank's customers' cards. To address this issue, the bank plans to establish a small team of fraud analysts who will review potentially risky transactions and make decisions on whether to allow or block them. The primary goal is to prevent as much fraud as possible while efficiently utilizing the analysts' capacity.

## Data
 - Private data: The bank has provided one year of historical transactional data along with fraud flags. The task is to build a machine learning model that predicts the likelihood of a transaction being marked as fraud.
 - More data coming soon.

## Files
The repository is organized as follows:
```
.
├── eda
│   ├── plots
│   ├── eda.py
│   └── plots.py
├── models
│   ├── plots
│   ├── grid_search.py
│   ├── preprocessing.py
│   └── tree_based_models.py
├── .gitignore
├── README.md
└── requirements.txt
```
Where:

- [eda/](./fs-test/eda): Contains all code for running exploratory data analysis.
  - [plots/](./fs-test/eda/plots): Some plots to facilitate better decision-making.
  - [eda.py](./fs-test/eda/eda.py): The main script to run EDA.
  - [plots.py](./fs-test/eda/plots.py): Script to run all EDA plots.
- [models/](./fs-test/models): Contains all code needed to execute the created models.
  - [plots/](./fs-test/models/plots): Performance plots and feature importance analysis.
  - [grid_search.py](./fs-test/models/grid_search.py): Module to perform grid search for any model.
  - [preprocessing.py](./fs-test/models/preprocessing.py): Preprocessing module that extracts and transforms the data for use by the models.
  - [tree_based_models.py](./fs-test/models/tree_based_models.py): The main script that executes RF/XGBoost.
- [.gitignore](./fs-test/.gitignore)
- [README.md](./fs-test/README.md)
- [requirements.txt](./fs-test/requirements.txt)

## Usage

### Create environment

1. Create a Python 3.9 environment using Conda. Replace `fraud_env` with your desired environment name:
```
conda create -n fraud_env python=3.9
```

2. Activate the newly created environment:
```
conda activate fraud_env
```

3. Install the required packages:
```
pip install -r requirements.txt
```

### Exploratory Data Analysis (EDA)

The eda directory contains code for conducting exploratory data analysis. Plots for making informed decisions are available in the plots subdirectory.

* Run with complete verbosity:
```
python eda.py --verbose
```

* Keep it simple:
```
python eda.py
```

### Models
The models directory contains code related to executing the machine learning models.
These are examples of use:

* Run Random Forest:
```
python tree_based_models.py --model rf
```

* Run RF with extra plots (confusion matrix, roc, and extra interesting features)
```
python tree_based_models.py --model rf --plots
```

* Run RF with GridSearch:
```
python tree_based_models.py --model rf --grid-search
```

* Run XGB and plot SHAP values
```
python tree_based_models.py --model xgb --shap
```

## About

This task presents an exciting opportunity for me to showcase my skills and apply them to a challenging problem. By developing a credit card fraud detection model, I aim to contribute to the prevention of fraudulent transactions and enhance the security of financial transactions for the bank's customers.

----------------
Gloria del Valle