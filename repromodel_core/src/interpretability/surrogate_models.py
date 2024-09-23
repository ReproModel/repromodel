# surrogate_models.py

"""
Surrogate Models Module
-----------------------

This module provides functions to train and evaluate surrogate models
that approximate the predictions of complex black-box models.

Functions:
- train_surrogate_model: Train a surrogate model.
- evaluate_surrogate_model: Evaluate the surrogate model's performance.
- plot_surrogate_performance: Visualize predictions of surrogate and black-box models.

Supported Surrogate Models:
- Decision Tree
- Linear Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             classification_report, mean_absolute_error)
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def train_surrogate_model(black_box_model, X_train, y_train,
                          task_type='regression', model_type='decision_tree',
                          **kwargs):
    """
    Trains a surrogate model to approximate a black-box model.

    Parameters:
    - black_box_model: Trained black-box model (not used directly but for clarity).
    - X_train (pd.DataFrame or np.ndarray): Training features.
    - y_train (pd.Series or np.ndarray): Training targets.
    - task_type (str): 'regression' or 'classification'.
    - model_type (str): Type of surrogate model to use.
    - **kwargs: Additional parameters for the surrogate model.

    Returns:
    - surrogate_model: Trained surrogate model.
    """

    # Input validation
    if task_type not in ['regression', 'classification']:
        raise ValueError("task_type must be 'regression' or 'classification'.")

    if not hasattr(black_box_model, 'predict'):
        raise ValueError("black_box_model must have a predict method.")

    # Generate predictions from the black-box model
    y_black_box_pred = black_box_model.predict(X_train)

    # Select appropriate surrogate model
    if task_type == 'regression':
        model_dict = {
            'decision_tree': DecisionTreeRegressor,
            'linear_regression': LinearRegression,
            'random_forest': RandomForestRegressor,
            'svm': SVR,
            'knn': KNeighborsRegressor
        }
    else:  # classification
        model_dict = {
            'decision_tree': DecisionTreeClassifier,
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'knn': KNeighborsClassifier
        }

    if model_type not in model_dict:
        raise ValueError(f"model_type must be one of {list(model_dict.keys())}.")

    SurrogateModel = model_dict[model_type]
    surrogate_model = SurrogateModel(**kwargs)

    # Train surrogate model to predict black-box predictions
    surrogate_model.fit(X_train, y_black_box_pred)

    return surrogate_model


def evaluate_surrogate_model(black_box_model, surrogate_model, X_test,
                             y_test=None, task_type='regression'):
    """
    Evaluates the surrogate model's performance compared to the black-box model.

    Parameters:
    - black_box_model: Trained black-box model.
    - surrogate_model: Trained surrogate model.
    - X_test (pd.DataFrame or np.ndarray): Test features.
    - y_test (pd.Series or np.ndarray, optional): True test targets.
    - task_type (str): 'regression' or 'classification'.

    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """

    # Predict with black-box and surrogate models
    y_black_box_pred = black_box_model.predict(X_test)
    y_surrogate_pred = surrogate_model.predict(X_test)

    metrics = {}

    if task_type == 'regression':
        # Calculate regression metrics
        mse = mean_squared_error(y_black_box_pred, y_surrogate_pred)
        mae = mean_absolute_error(y_black_box_pred, y_surrogate_pred)
        r2 = r2_score(y_black_box_pred, y_surrogate_pred)
        metrics['MSE'] = mse
        metrics['MAE'] = mae
        metrics['R2'] = r2
        print(f"Surrogate Model Evaluation (Regression):\nMSE: {mse:.4f}, "
              f"MAE: {mae:.4f}, R2: {r2:.4f}")
    else:  # classification
        # Calculate classification metrics
        accuracy = accuracy_score(y_black_box_pred, y_surrogate_pred)
        report = classification_report(y_black_box_pred, y_surrogate_pred)
        metrics['Accuracy'] = accuracy
        metrics['Classification Report'] = report
        print(f"Surrogate Model Evaluation (Classification):\nAccuracy: "
              f"{accuracy:.4f}\n")
        print("Classification Report:\n", report)

    return metrics


def plot_surrogate_performance(black_box_model, surrogate_model, X_test,
                               sample_size=100, task_type='regression'):
    """
    Plots the predictions of the surrogate and black-box models for comparison.

    Parameters:
    - black_box_model: Trained black-box model.
    - surrogate_model: Trained surrogate model.
    - X_test (pd.DataFrame or np.ndarray): Test features.
    - sample_size (int): Number of samples to plot.
    - task_type (str): 'regression' or 'classification'.

    Returns:
    - None
    """

    # Sample data for plotting
    X_sample = X_test[:sample_size]

    # Predictions
    y_black_box_pred = black_box_model.predict(X_sample)
    y_surrogate_pred = surrogate_model.predict(X_sample)

    plt.figure(figsize=(10, 6))

    if task_type == 'regression':
        plt.plot(y_black_box_pred, label='Black-Box Predictions', marker='o')
        plt.plot(y_surrogate_pred, label='Surrogate Predictions', marker='x')
        plt.title('Surrogate vs Black-Box Model Predictions (Regression)')
        plt.ylabel('Predicted Value')
    else:
        plt.scatter(range(len(y_black_box_pred)), y_black_box_pred,
                    label='Black-Box Predictions', marker='o')
        plt.scatter(range(len(y_surrogate_pred)), y_surrogate_pred,
                    label='Surrogate Predictions', marker='x')
        plt.title('Surrogate vs Black-Box Model Predictions (Classification)')
        plt.ylabel('Predicted Class')

    plt.xlabel('Sample Index')
    plt.legend()
    plt.tight_layout()
    plt.show()


'''
# Example usage of surrogate_models.py

from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# For Regression Task
# Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Train black-box model
black_box_model = RandomForestRegressor(n_estimators=100, random_state=42)
black_box_model.fit(X_train, y_train)

# Train surrogate model
surrogate_model = train_surrogate_model(black_box_model, X_train, y_train,
                                        task_type='regression',
                                        model_type='decision_tree',
                                        max_depth=5)

# Evaluate surrogate model
metrics = evaluate_surrogate_model(black_box_model, surrogate_model, X_test,
                                   y_test, task_type='regression')

# Plot performance
plot_surrogate_performance(black_box_model, surrogate_model, X_test,
                           sample_size=50, task_type='regression')

# For Classification Task
# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Train black-box model
black_box_model = RandomForestClassifier(n_estimators=100, random_state=42)
black_box_model.fit(X_train, y_train)

# Train surrogate model
surrogate_model = train_surrogate_model(black_box_model, X_train, y_train,
                                        task_type='classification',
                                        model_type='decision_tree',
                                        max_depth=3)

# Evaluate surrogate model
metrics = evaluate_surrogate_model(black_box_model, surrogate_model, X_test,
                                   y_test, task_type='classification')

# Plot performance
plot_surrogate_performance(black_box_model, surrogate_model, X_test,
                           sample_size=50, task_type='classification')

'''