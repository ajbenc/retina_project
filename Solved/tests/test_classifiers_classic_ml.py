import pytest
from unittest.mock import patch
import numpy as np
import os
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.classifiers_classic_ml import visualize_embeddings, train_and_evaluate_model

####################################################################################################
################################### Test the Classical ML Models ###################################
####################################################################################################

@pytest.fixture
def sample_embedding_data():
    """
    Fixture to create a mock dataset for testing dimensionality reduction and model training.
    Returns:
        X_train, X_test, y_train, y_test: Training and testing data along with labels.
    """
    # Create a synthetic dataset with 20 samples, 6 features, and 3 classes
    X, y = make_classification(n_samples=20, n_features=6, n_classes=3, random_state=42, n_informative=4)
    
    # Split the dataset into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

@pytest.mark.parametrize("method, plot_type", [
    ('PCA', '2D'),  # PCA reduction to 2D
    ('PCA', '3D'),  # PCA reduction to 3D
])   
def test_visualize_embeddings(method, plot_type, sample_embedding_data):
    """
    Test the dimensionality reduction and embedding visualization.
    This ensures that PCA can reduce embeddings correctly and produce visualizations.
    """
    X_train, X_test, y_train, y_test = sample_embedding_data

    # Mock the plotly figures to avoid actual plotting in test environment
    with patch('plotly.graph_objs.Figure.show'):
        # Test the visualize_embeddings function
        model = visualize_embeddings(X_train, X_test, y_train, y_test, plot_type=plot_type, method=method)
    
    # Check if the PCA model is an instance of the correct class and has the expected number of components
    assert isinstance(model, PCA), "The model should be an instance of PCA"
    if plot_type == '2D':
        assert model.n_components_ == 2, "PCA should reduce data to 2 components"
    elif plot_type == '3D':
        assert model.n_components_ == 3, "PCA should reduce data to 3 components"
        

def test_train_and_evaluate_model(sample_embedding_data):
    """
    Test the training and evaluation of models (Logistic Regression, Random Forest).
    Ensures that models are correctly trained and returned in the expected format.
    """
    X_train, X_test, y_train, y_test = sample_embedding_data

    # Train and evaluate the models
    trained_models = train_and_evaluate_model(X_train, X_test, y_train, y_test, test=False)

    # Verify that trained_models is a list
    assert isinstance(trained_models, list), "The output should be a list of trained models"
    
    # Check that at least two models were trained (Logistic Regression, Random Forest)
    assert len(trained_models) >= 2, "At least two models should be trained"
    
    # Check that the models have Logistic Regression and Random Forest
    models_instances = [model for _, model in trained_models]
    assert any(isinstance(model, LogisticRegression) for model in models_instances), "Logistic Regression model not found"
    assert any(isinstance(model, RandomForestClassifier) for model in models_instances), "Random Forest model not found"

    # Ensure that the trained models are indeed fitted (trained)
    for name, model in trained_models:
        assert hasattr(model, 'fit'), f"{name} should have a fit method"
        assert hasattr(model, 'predict'), f"{name} should have a predict method"
        
        # Check if the model is correctly trained by predicting on the test set
        y_pred = model.predict(X_test)
        assert y_pred is not None, f"{name} should have successfully made predictions"


if __name__ == "__main__":
    pytest.main()