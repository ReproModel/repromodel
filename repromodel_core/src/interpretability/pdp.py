# pdp.py

"""
Partial Dependence Plots (PDP) Module
-------------------------------------

This module provides functions to compute and plot Partial Dependence Plots (PDPs),
which show the relationship between selected features and the predicted outcome,
while averaging out the effects of all other features.

Functions:
- compute_pdp: Compute partial dependence data for given features.
- plot_pdp: Plot the partial dependence plot for given features.

Features:
- Supports both regression and classification tasks.
- Handles numerical and categorical features.
- Allows for univariate and bivariate PDPs.
- Provides options for customizing plots.

Dependencies:
- scikit-learn
- matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.exceptions import NotFittedError
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def compute_pdp(model, X, features, kind='average', grid_resolution=100):
    """
    Computes partial dependence for given features.

    Parameters:
    - model: Trained model that follows scikit-learn API.
    - X (pd.DataFrame or np.ndarray): Dataset used for computing PDP.
    - features (list or tuple): List of feature names or indices for which to compute PDP.
                                For bivariate PDPs, provide a list of two features.
    - kind (str): 'average' for PDP, 'individual' for ICE curves, or 'both'.
    - grid_resolution (int): Number of points to use in the grid for plotting.

    Returns:
    - pdp_disp: A PartialDependenceDisplay object containing the computed results.
    """

    # Input validation
    if not hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        raise ValueError("Model must have a predict or predict_proba method.")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if not isinstance(features, (list, tuple)):
        features = [features]

    try:
        pdp_disp = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=features,
            kind=kind,
            grid_resolution=grid_resolution,
            response_method='auto',
            percentiles=(0.05, 0.95),
            n_jobs=None
        )
        return pdp_disp
    except NotFittedError as e:
        raise NotFittedError("The model must be fitted before computing PDP.") from e
    except Exception as e:
        raise Exception("An error occurred while computing PDP.") from e


def plot_pdp(pdp_disp, feature_names=None, plot_type='line', centered=False,
             title=None, xlabel=None, ylabel=None, figsize=(10, 6), legend=True,
             show=True, save_path=None):
    """
    Plots the partial dependence plot.

    Parameters:
    - pdp_disp: A PartialDependenceDisplay object returned by compute_pdp.
    - feature_names (list, optional): List of feature names for labeling.
    - plot_type (str): 'line' for line plot, 'contour' for 2D contour plot (bivariate).
    - centered (bool): Whether to center the PDP at zero.
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Figure size.
    - legend (bool): Whether to display the legend.
    - show (bool): Whether to display the plot immediately.
    - save_path (str, optional): Path to save the plot image.

    Returns:
    - fig, ax: Matplotlib figure and axes objects.
    """

    # Customize plot
    fig, ax = plt.subplots(figsize=figsize)
    pdp_disp.plot(ax=ax, line_kw={'linewidth': 2}, contour_kw={'cmap': 'viridis'})

    # Set titles and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Partial Dependence Plot', fontsize=14)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    elif feature_names:
        ax.set_xlabel(feature_names[0], fontsize=12)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel('Partial Dependence', fontsize=12)

    # Center the plot if requested
    if centered:
        y_min, y_max = ax.get_ylim()
        max_abs = max(abs(y_min), abs(y_max))
        ax.set_ylim(-max_abs, max_abs)
        ax.axhline(0, color='grey', linestyle='--', linewidth=1)

    if not legend:
        ax.legend_.remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    return fig, ax


'''
# Example usage of pdp.py

from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from pdp import compute_pdp, plot_pdp

# For Regression Task
# Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute PDP for a single feature
pdp_disp = compute_pdp(model, X_test, features=['LSTAT'])

# Plot PDP
plot_pdp(pdp_disp, title='Partial Dependence of House Price on LSTAT', xlabel='LSTAT')

# For Classification Task
# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute PDP for a single feature
pdp_disp = compute_pdp(model, X_test, features=['petal width (cm)'])

# Plot PDP
plot_pdp(pdp_disp, title='Partial Dependence of Class Probability on Petal Width', xlabel='Petal Width (cm)')


# Example with bivariate PDP
# Compute PDP for two features
pdp_disp = compute_pdp(model, X_test, features=[0, 1], kind='average')

# Plot PDP
plot_pdp(pdp_disp, title='Partial Dependence of Target on Features 0 and 1',
         xlabel='Feature 0', ylabel='Feature 1', plot_type='contour')
'''