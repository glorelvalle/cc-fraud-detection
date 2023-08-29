# -*- coding: utf-8 -*-
"""
Created on Thu 4 Aug 2023 19:15:00 CEST

@author: Gloria del Valle
"""
from sklearn.model_selection import GridSearchCV
import warnings

# Suppress a specific warning
warnings.filterwarnings("ignore", category=FutureWarning)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 20],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_features": ["auto", "sqrt"],
}


class GridSearch:
    """
    Perform grid search with a determined model
    """

    def __init__(self, data_loader, model):
        """
        Initialize GridSearch class
        """
        self.data_loader = data_loader
        self.model = model

    def grid_search(self):
        """
        Perform grid search with a determined model

        Returns:
            y_pred: predictions on the test set
        """

        # Get the valid hyperparameters for Random Forest
        valid_params = [
            param for param in param_grid if param in self.model.get_params()
        ]

        # Filter the parameter grid to include only valid parameters
        valid_param_grid = {param: param_grid[param] for param in valid_params}

        # Initialize GridSearchCV
        grid_search = GridSearchCV(self.model, valid_param_grid, cv=3, n_jobs=-1)

        # Perform grid search
        grid_search.fit(self.data_loader.X_train, self.data_loader.y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions on the test set
        y_pred = best_model.predict(self.data_loader.X_test)

        # Print best hyperparameters
        print(
            f"Best hyperparameters for {self.model.__class__.__name__} model:",
            grid_search.best_params_,
        )

        return y_pred
