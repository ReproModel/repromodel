# utilities.py

"""
Utilities Module
----------------

This module provides utility functions for data preprocessing, feature importance computation,
and input validation to support the main modules.

Functions:
- preprocess_data: Preprocesses data (scaling, encoding, handling missing values).
- compute_feature_importance: Computes global feature importance.
- validate_inputs: Validates input data and models.
- sample_data: Samples data for performance optimization.
- get_categorical_features: Identifies categorical features in a dataset.

Dependencies:
- pandas
- numpy
- scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def preprocess_data(X, y=None, categorical_features=None, numerical_features=None,
                    impute_strategy='mean', scale_numerical=True, encode_categorical=True):
    """
    Preprocesses the data (e.g., scaling, encoding, imputing missing values).

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series or np.ndarray, optional): Target variable.
    - categorical_features (list, optional): List of categorical feature names.
    - numerical_features (list, optional): List of numerical feature names.
    - impute_strategy (str): Strategy for imputing missing values ('mean', 'median', 'most_frequent', 'constant').
    - scale_numerical (bool): Whether to scale numerical features.
    - encode_categorical (bool): Whether to encode categorical features.

    Returns:
    - X_preprocessed: Preprocessed feature dataset.
    - y: Target variable (unchanged).
    - preprocessor: Fitted preprocessing pipeline.
    """

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if categorical_features is None:
        categorical_features = get_categorical_features(X)

    if numerical_features is None:
        numerical_features = X.columns.difference(categorical_features)

    # Define transformers
    transformers = []

    if numerical_features.any():
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('scaler', StandardScaler() if scale_numerical else 'passthrough')
        ])
        transformers.append(('num', numerical_transformer, numerical_features))

    if categorical_features.any():
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore') if encode_categorical else 'passthrough')
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit and transform
    X_preprocessed = preprocessor.fit_transform(X)

    # Convert to DataFrame if possible
    if isinstance(X_preprocessed, np.ndarray):
        feature_names = preprocessor.get_feature_names_out()
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

    return X_preprocessed, y, preprocessor


def compute_feature_importance(model, feature_names):
    """
    Computes global feature importance scores.

    Parameters:
    - model: Trained model with feature_importances_ or coef_ attribute.
    - feature_names (list): List of feature names.

    Returns:
    - importance_df: DataFrame containing feature names and their importance scores.
    """

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = np.mean(importances, axis=0)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute.")

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return importance_df.to_dict('records')


def validate_inputs(model, X):
    """
    Validates input data and model.

    Parameters:
    - model: Trained model.
    - X (pd.DataFrame or np.ndarray): Input features.

    Returns:
    - None
    """

    # Check if model is fitted
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a predict method.")

    try:
        model.predict(X.iloc[:1] if isinstance(X, pd.DataFrame) else X[:1])
    except NotFittedError as e:
        raise NotFittedError("The model must be fitted before making predictions.") from e
    except Exception as e:
        raise Exception("An error occurred while validating the model.") from e

    # Check if X is DataFrame or ndarray
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be a pandas DataFrame or numpy ndarray.")

    # Additional validation checks can be added here


def sample_data(X, y=None, sample_size=1000, random_state=None):
    """
    Samples data for performance optimization.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Feature dataset.
    - y (pd.Series or np.ndarray, optional): Target variable.
    - sample_size (int): Number of samples to draw.
    - random_state (int, optional): Seed for random sampling.

    Returns:
    - X_sampled: Sampled feature dataset.
    - y_sampled: Sampled target variable (if provided).
    """

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if sample_size < len(X):
        X_sampled = X.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        if y is not None:
            y_sampled = y[X_sampled.index]
            return X_sampled, y_sampled
        else:
            return X_sampled, None
    else:
        return X, y


def get_categorical_features(X):
    """
    Identifies categorical features in a dataset.

    Parameters:
    - X (pd.DataFrame): Input dataset.

    Returns:
    - categorical_features (list): List of categorical feature names.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return categorical_features

'''
# Example usage of utilities.py

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utilities import preprocess_data, compute_feature_importance, validate_inputs

# Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Introduce missing values for demonstration
X.iloc[5:10, 3] = np.nan

# Preprocess data
X_preprocessed, y, preprocessor = preprocess_data(X, y, impute_strategy='median', scale_numerical=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate inputs
validate_inputs(model, X_test)

# Compute feature importances
importance_df = compute_feature_importance(model, feature_names=X_train.columns.tolist())
'''