# ice.py

"""
Individual Conditional Expectation (ICE) Plots Module
-----------------------------------------------------

This module provides functions to compute and plot Individual Conditional Expectation (ICE) plots,
which show the dependency of the prediction on a feature for individual instances.

Functions:
- compute_ice: Compute ICE data for a given feature.
- plot_ice: Plot the ICE curves for a given feature.

Features:
- Supports both regression and classification tasks.
- Handles numerical and categorical features.
- Offers options to overlay Partial Dependence Plot (PDP) for comparison.
- Provides customization options for plotting.
- Includes performance optimizations for handling large datasets.

Dependencies:
- scikit-learn
- matplotlib
- pandas
- numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from sklearn.exceptions import NotFittedError
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def compute_ice(model, X, feature, grid_resolution=100, percentiles=(0.05, 0.95),
                kind='individual', subsample=1.0, random_state=None):
    """
    Computes ICE curves for a given feature.

    Parameters:
    - model: Trained model that follows scikit-learn API.
    - X (pd.DataFrame or np.ndarray): Dataset used for computing ICE.
    - feature (str or int): Feature name or index for which to compute ICE curves.
    - grid_resolution (int): Number of points to use in the grid for plotting.
    - percentiles (tuple): The lower and upper percentile used to create the extreme values
                           for the feature grid.
    - kind (str): 'individual' for ICE, 'average' for PDP, 'both' for both.
    - subsample (float): Fraction of data to use for ICE computation (between 0 and 1).
    - random_state (int, optional): Seed for random sampling.

    Returns:
    - ice_disp: A PartialDependenceDisplay object containing the computed results.
    """

    # Input validation
    if not hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        raise ValueError("Model must have a predict or predict_proba method.")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if not isinstance(feature, (list, tuple)):
        features = [feature]
    else:
        features = feature

    if subsample < 1.0:
        X = X.sample(frac=subsample, random_state=random_state).reset_index(drop=True)

    try:
        ice_disp = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=features,
            kind=kind,
            grid_resolution=grid_resolution,
            response_method='auto',
            percentiles=percentiles,
            n_jobs=None
        )
        return ice_disp
    except NotFittedError as e:
        raise NotFittedError("The model must be fitted before computing ICE.") from e
    except Exception as e:
        raise Exception("An error occurred while computing ICE.") from e


def plot_ice(ice_disp, feature_name=None, plot_pdp=False, centered=False,
             title=None, xlabel=None, ylabel=None, figsize=(10, 6), legend=True,
             show=True, save_path=None, cmap='viridis'):
    """
    Plots the ICE curves.

    Parameters:
    - ice_disp: A PartialDependenceDisplay object returned by compute_ice.
    - feature_name (str, optional): Feature name for labeling.
    - plot_pdp (bool): Whether to overlay the Partial Dependence Plot.
    - centered (bool): Whether to center the ICE curves at zero.
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Figure size.
    - legend (bool): Whether to display the legend.
    - show (bool): Whether to display the plot immediately.
    - save_path (str, optional): Path to save the plot image.
    - cmap (str): Colormap to use for the ICE curves.

    Returns:
    - fig, ax: Matplotlib figure and axes objects.
    """

    # Customize plot
    fig, ax = plt.subplots(figsize=figsize)

    ice_disp.plot(ax=ax, line_kw={'alpha': 0.2, 'color': 'grey'}, pd_line_kw={'color': 'red', 'linewidth': 2})

    # Center the ICE curves if requested
    if centered:
        ice_values = ice_disp.deciles_[0][1]
        ice_disp.plot(ax=ax, centered=centered, line_kw={'alpha': 0.2, 'color': 'grey'}, pd_line_kw={'color': 'red', 'linewidth': 2})

    # Overlay PDP if requested
    if plot_pdp:
        ice_disp.plot(ax=ax, pd_line_kw={'color': 'red', 'linewidth': 2})

    # Set titles and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Individual Conditional Expectation (ICE) Plot', fontsize=14)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    elif feature_name:
        ax.set_xlabel(feature_name, fontsize=12)
    else:
        ax.set_xlabel('Feature Value', fontsize=12)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel('Predicted Response', fontsize=12)

    if not legend:
        if hasattr(ax, 'legend_') and ax.legend_:
            ax.legend_.remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    return fig, ax

'''
# Example usage of ice.py

from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from ice import compute_ice, plot_ice

# For Regression Task
# Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute ICE for a single feature
ice_disp = compute_ice(model, X_test, feature='LSTAT', subsample=0.5, random_state=42)

# Plot ICE
plot_ice(ice_disp, feature_name='LSTAT', plot_pdp=True, title='ICE Plot for LSTAT', xlabel='LSTAT')

# For Classification Task
# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute ICE for a single feature
ice_disp = compute_ice(model, X_test, feature='petal width (cm)', subsample=0.5, random_state=42)

# Plot ICE
plot_ice(ice_disp, feature_name='Petal Width (cm)', plot_pdp=True, title='ICE Plot for Petal Width', xlabel='Petal Width (cm)')
'''