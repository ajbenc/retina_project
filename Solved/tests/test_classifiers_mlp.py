import pytest
import numpy as np
import os
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.classifiers_classic_ml import visualize_embeddings, train_and_evaluate_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from src.classifiers_mlp import MultimodalDataset, train_mlp, create_early_fusion_model


####################################################################################################
##################################### Test the Keras MLP Models ####################################
####################################################################################################


@pytest.fixture
def correlated_sample_data():
    """
    Fixture to create a correlated synthetic dataset using make_classification for testing.
    It generates data with 10 text features and 10 image features.
    Returns:
        train_df (pd.DataFrame): DataFrame with train data.
        test_df (pd.DataFrame): DataFrame with test data.
    """
    # Create synthetic multi-class data with 8 features (4 text-like, 4 image-like)
    X, y = make_classification(n_samples=20, n_features=8, n_informative=6, n_classes=3, random_state=42)

    # Rename features to simulate text and image columns
    feature_names = [f'text_{i}' for i in range(4)] + [f'image_{i}' for i in range(4, 8)]
    
    # Create a DataFrame and assign class labels
    df = pd.DataFrame(X, columns=feature_names)
    df['class_id'] = y

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    return train_df, test_df

@pytest.fixture
def label_encoder(correlated_sample_data):
    """
    Fixture to create a label encoder based on the training data.
    """
    train_df, _ = correlated_sample_data
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['class_id'])
    return label_encoder

def test_multimodal_dataset_image_only(correlated_sample_data, label_encoder):
    """
    Test the MultimodalDataset class with only image data.
    """
    train_df, test_df = correlated_sample_data

    # Image columns (the second 4 features)
    image_columns = [f'image_{i}' for i in range(4, 8)]
    label_column = 'class_id'

    # Create the dataset
    train_dataset = MultimodalDataset(train_df, text_cols=None, image_cols=image_columns, label_col=label_column, encoder=label_encoder)

    # Check if the dataset is correctly instantiated
    assert train_dataset.image_data is not None, "Image data should be instantiated"
    assert train_dataset.text_data is None, "Text data should be None"
    
    # Fetch a batch of data
    (batch_inputs, batch_labels) = train_dataset[0]
    
    assert 'image' in batch_inputs, "Batch should contain image data"
    assert 'text' not in batch_inputs, "Batch should not contain text data"
    assert batch_inputs['image'].shape[1] == len(image_columns), "Image data shape is incorrect"
    assert batch_labels is not None, "Batch should contain labels"
    assert batch_labels.shape[0] == batch_inputs['image'].shape[0], "Labels should match the batch size"

def test_multimodal_dataset_text_only(correlated_sample_data, label_encoder):
    """
    Test the MultimodalDataset class with only text data.
    """
    train_df, test_df = correlated_sample_data

    # Text columns (the first 4 features)
    text_columns = [f'text_{i}' for i in range(4)]
    label_column = 'class_id'

    # Create the dataset
    train_dataset = MultimodalDataset(train_df, text_cols=text_columns, image_cols=None, label_col=label_column, encoder=label_encoder)

    # Check if the dataset is correctly instantiated
    assert train_dataset.text_data is not None, "Text data should be instantiated"
    assert train_dataset.image_data is None, "Image data should be None"
    
    # Fetch a batch of data
    (batch_inputs, batch_labels) = train_dataset[0]
    
    assert 'text' in batch_inputs, "Batch should contain text data"
    assert 'image' not in batch_inputs, "Batch should not contain image data"
    assert batch_inputs['text'].shape[1] == len(text_columns), "Text data shape is incorrect"
    assert batch_labels is not None, "Batch should contain labels"
    assert batch_labels.shape[0] == batch_inputs['text'].shape[0], "Labels should match the batch size"

def test_multimodal_dataset_multimodal(correlated_sample_data, label_encoder):
    """
    Test the MultimodalDataset class with both text and image data.
    """
    train_df, test_df = correlated_sample_data

    # Text and image columns
    text_columns = [f'text_{i}' for i in range(4)]
    image_columns = [f'image_{i}' for i in range(4, 8)]
    label_column = 'class_id'

    # Create the dataset
    train_dataset = MultimodalDataset(train_df, text_cols=text_columns, image_cols=image_columns, label_col=label_column, encoder=label_encoder)

    # Check if the dataset is correctly instantiated
    assert train_dataset.text_data is not None, "Text data should be instantiated"
    assert train_dataset.image_data is not None, "Image data should be instantiated"
    
    # Fetch a batch of data
    (batch_inputs, batch_labels) = train_dataset[0]
    assert 'text' in batch_inputs, "Batch should contain text data"
    assert 'image' in batch_inputs, "Batch should contain image data"
    assert batch_inputs['text'].shape[1] == len(text_columns), "Text data shape is incorrect"
    assert batch_inputs['image'].shape[1] == len(image_columns), "Image data shape is incorrect"
    assert batch_labels is not None, "Batch should contain labels"
    assert batch_labels.shape[0] == batch_inputs['text'].shape[0] == batch_inputs['image'].shape[0], "Labels should match the batch size"



def test_create_early_fusion_model_single_modality_image():
    """
    Test the model creation with only image input or only text input.
    Ensure the architecture matches expectations.
    """
    text_input_size = None
    image_input_size = 4
    output_size = 3

    # Create the model
    model = create_early_fusion_model(text_input_size, image_input_size, output_size, hidden=[128, 64], p=0.3)

    # Check if the model has the expected number of layers
    assert isinstance(model, Model), "Model should be a Keras Model instance"

    # Check that the input and output shapes are consistent
    assert model.input_shape == (None, image_input_size), "Input shape should match image input size"
    assert model.output_shape == (None, output_size), "Output shape should match number of classes"

    # Check that there are the correct number of Dense, Dropout, and BatchNormalization layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]
    batchnorm_layers = [layer for layer in model.layers if isinstance(layer, BatchNormalization)]

    assert len(dense_layers) == 3, "There should be 3 Dense layers (2 hidden + 1 output)"
    assert len(dropout_layers) > 0, "There should be at least 1 Dropout layers"
    assert len(batchnorm_layers) > 0, "There should be at least 1 BatchNormalization layer"

def test_create_early_fusion_model_single_modality_text():
    """
    Test the model creation with only image input or only text input.
    Ensure the architecture matches expectations.
    """
    text_input_size = 4
    image_input_size = None
    output_size = 3

    # Create the model
    model = create_early_fusion_model(text_input_size, image_input_size, output_size, hidden=[128, 64], p=0.3)

    # Check if the model has the expected number of layers
    assert isinstance(model, Model), "Model should be a Keras Model instance"

    # Check that the input and output shapes are consistent
    assert model.input_shape == (None, text_input_size), "Input shape should match text input size"
    assert model.output_shape == (None, output_size), "Output shape should match number of classes"

    # Check that there are the correct number of Dense, Dropout, and BatchNormalization layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]
    batchnorm_layers = [layer for layer in model.layers if isinstance(layer, BatchNormalization)]

    assert len(dense_layers) == 3, "There should be 3 Dense layers (2 hidden + 1 output)"
    assert len(dropout_layers) > 0, "There should be at least 1 Dropout layers"
    assert len(batchnorm_layers) > 0, "There should be at least 1 BatchNormalization layer"


def test_create_early_fusion_model_multimodal():
    """
    Test the model creation with both text and image input.
    Ensure the architecture matches expectations.
    """
    text_input_size = 4
    image_input_size = 4
    output_size = 3

    # Create the model
    model = create_early_fusion_model(text_input_size, image_input_size, output_size, hidden=[128, 64], p=0.3)

    # Check if the model has the expected number of layers
    assert isinstance(model, Model), "Model should be a Keras Model instance"

    # Check that the input and output shapes are consistent
    assert model.input_shape == [(None, text_input_size), (None, image_input_size)], "Input shape should match both text and image input sizes"
    assert model.output_shape == (None, output_size), "Output shape should match number of classes"

    # Check that the concatenation of text and image inputs is present
    assert any(isinstance(layer, Concatenate) for layer in model.layers), "There should be a Concatenate layer for text and image inputs"

    # Check that there are the correct number of Dense, Dropout, and BatchNormalization layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]
    batchnorm_layers = [layer for layer in model.layers if isinstance(layer, BatchNormalization)]

    assert len(dense_layers) == 3, "There should be 3 Dense layers (2 hidden + 1 output)"
    assert len(dropout_layers) > 0, "There should be at least 1 Dropout layers"
    assert len(batchnorm_layers) > 0, "There should be at least 1 BatchNormalization layer"


def test_train_mlp_single_modality_image(correlated_sample_data, label_encoder):
    """
    Test the MLP training with only image data.
    Ensure the model trains and evaluates correctly.
    """
    train_df, test_df = correlated_sample_data

    # Image columns (the second 10 features)
    image_columns = [f'image_{i}' for i in range(4, 8)]
    label_column = 'class_id'

    # Create datasets
    train_dataset = MultimodalDataset(train_df, text_cols=None, image_cols=image_columns, label_col=label_column, encoder=label_encoder)
    test_dataset = MultimodalDataset(test_df, text_cols=None, image_cols=image_columns, label_col=label_column, encoder=label_encoder)

    image_input_size = len(image_columns)
    output_size = len(label_encoder.classes_)

    # Train the model
    model, test_accuracy, f1, macro_auc = train_mlp(
        train_loader=train_dataset,
        test_loader=test_dataset,
        text_input_size=None,
        image_input_size=image_input_size,
        output_size=output_size,
        num_epochs=1,
        set_weights=True,
        adam=True, 
        patience=10,
        save_results=False,
        train_model=False,
        test_mlp_model=False
    )
    
    # Check model
    assert model is not None, "Model should not be None after training."

    # Ensure the model is compiled with the correct loss and optimizer
    assert isinstance(model.loss, CategoricalCrossentropy) or model.loss == 'categorical_crossentropy', f"Loss function should be categorical crossentropy, but got {model.loss}"
    
    # Check model input and output shapes
    assert model.input_shape == (None, image_input_size), "Input shape should match image input size"
    assert model.output_shape == (None, output_size), "Output shape should match number of classes"

    # Check if the model is compiled with the correct optimizer
    assert isinstance(model.optimizer, Adam) or isinstance(model.optimizer, SGD), f"Optimizer should be Adam or SGD, but got {model.optimizer}"


def test_train_mlp_single_modality_text(correlated_sample_data, label_encoder):
    """
    Test the MLP training with only text data.
    Ensure the model trains and evaluates correctly.
    """
    train_df, test_df = correlated_sample_data

    # Text columns (the first 10 features)
    text_columns = [f'text_{i}' for i in range(4)]
    label_column = 'class_id'

    # Create datasets
    train_dataset = MultimodalDataset(train_df, text_cols=text_columns, image_cols=None, label_col=label_column, encoder=label_encoder)
    test_dataset = MultimodalDataset(test_df, text_cols=text_columns, image_cols=None, label_col=label_column, encoder=label_encoder)

    text_input_size = len(text_columns)
    output_size = len(label_encoder.classes_)

    # Train the model
    model, test_accuracy, f1, macro_auc = train_mlp(
        train_loader=train_dataset,
        test_loader=test_dataset,
        text_input_size=text_input_size,
        image_input_size=None,
        output_size=output_size,
        num_epochs=1,
        set_weights=True,
        adam=True, 
        patience=10,
        save_results=False,
        train_model=False,
        test_mlp_model=False
    )
    
    # Check model
    assert model is not None, "Model should not be None after training."

    # Ensure the model is compiled with the correct loss and optimizer
    assert isinstance(model.loss, CategoricalCrossentropy) or model.loss == 'categorical_crossentropy', f"Loss function should be categorical crossentropy, but got {model.loss}"
    
    # Check model input and output shapes
    assert model.input_shape == (None, text_input_size), "Input shape should match text input size"
    assert model.output_shape == (None, output_size), "Output shape should match number of classes"
    
    # Check if the model is compiled with the correct optimizer
    assert isinstance(model.optimizer, Adam) or isinstance(model.optimizer, SGD), f"Optimizer should be Adam or SGD, but got {model.optimizer}"

def test_train_mlp_multimodal(correlated_sample_data, label_encoder):
    """
    Test the MLP training with class weights for an imbalanced dataset.
    Ensure class weights are applied correctly and early stopping works.
    """
    train_df, test_df = correlated_sample_data

    # Text and image columns
    text_columns = [f'text_{i}' for i in range(4)]
    image_columns = [f'image_{i}' for i in range(4, 8)]
    label_column = 'class_id'

    # Create datasets
    train_dataset = MultimodalDataset(train_df, text_cols=text_columns, image_cols=image_columns, label_col=label_column, encoder=label_encoder)
    test_dataset = MultimodalDataset(test_df, text_cols=text_columns, image_cols=image_columns, label_col=label_column, encoder=label_encoder)

    text_input_size = len(text_columns)
    image_input_size = len(image_columns)
    output_size = len(label_encoder.classes_)

    # Train the model
    model, test_accuracy, f1, macro_auc = train_mlp(
        train_loader=train_dataset,
        test_loader=test_dataset,
        text_input_size=text_input_size,
        image_input_size=image_input_size,
        output_size=output_size,
        num_epochs=1,
        set_weights=True,
        adam=True, 
        patience=10,
        save_results=False,
        train_model=False,
        test_mlp_model=False
    )
    
    # Check model
    assert model is not None, "Model should not be None after training."

    # Ensure the model is compiled with the correct loss and optimizer
    assert isinstance(model.loss, CategoricalCrossentropy) or model.loss == 'categorical_crossentropy', f"Loss function should be categorical crossentropy, but got {model.loss}"

    # Check model input and output shapes
    assert model.input_shape == [(None, text_input_size), (None, image_input_size)], "Input shape should match both text and image input sizes"
    assert model.output_shape == (None, output_size), "Output shape should match number of classes"
    
    # Check if the model is compiled with the correct optimizer
    assert isinstance(model.optimizer, Adam) or isinstance(model.optimizer, SGD), f"Optimizer should be Adam or SGD, but got {model.optimizer}"


# Check if the result files are correctly saved
def test_result_files():
    """
    Test if the result files are created for each modality and have the correct format.
    """
    # Get the absolute path of the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths for result files relative to the test file location
    multimodal_results_path = os.path.join(test_dir, "../results/multimodal_results.csv")
    text_results_path = os.path.join(test_dir, "../results/text_results.csv")
    image_results_path = os.path.join(test_dir, "../results/image_results.csv")

    # Check if the files exist
    assert os.path.exists(multimodal_results_path), "Multimodal result file is missing!"
    assert os.path.exists(text_results_path), "Text result file is missing!"
    assert os.path.exists(image_results_path), "Image result file is missing!"

    # Check if the files are not empty and in correct format (CSV)
    for file_path in [multimodal_results_path, text_results_path, image_results_path]:
        df = pd.read_csv(file_path)
        assert not df.empty, f"{file_path} is empty!"
        assert 'Predictions' in df.columns and 'True Labels' in df.columns, f"{file_path} is not in the correct format!"

# Check if the accuracy and F1 scores meet the specified thresholds
def test_model_performance():
    """
    Test if the accuracy and F1 score are above the required thresholds.
    """
    # Get the absolute path of the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths for result files relative to the test file location
    multimodal_results_path = os.path.join(test_dir, "../results/multimodal_results.csv")
    text_results_path = os.path.join(test_dir, "../results/text_results.csv")
    image_results_path = os.path.join(test_dir, "../results/image_results.csv")

    # Load the result files
    multimodal_results = pd.read_csv(multimodal_results_path)
    text_results = pd.read_csv(text_results_path)
    image_results = pd.read_csv(image_results_path)

    # Define the accuracy and F1-score thresholds
    multimodal_accuracy_threshold = 0.85
    multimodal_f1_threshold = 0.80
    text_accuracy_threshold = 0.85
    text_f1_threshold = 0.80
    image_accuracy_threshold = 0.75
    image_f1_threshold = 0.70

    # Calculate accuracy and F1 score for multimodal results
    multimodal_accuracy = accuracy_score(multimodal_results['True Labels'], multimodal_results['Predictions'])
    multimodal_f1 = f1_score(multimodal_results['True Labels'], multimodal_results['Predictions'], average='macro')

    # Calculate accuracy and F1 score for text results
    text_accuracy = accuracy_score(text_results['True Labels'], text_results['Predictions'])
    text_f1 = f1_score(text_results['True Labels'], text_results['Predictions'], average='macro')

    # Calculate accuracy and F1 score for image results
    image_accuracy = accuracy_score(image_results['True Labels'], image_results['Predictions'])
    image_f1 = f1_score(image_results['True Labels'], image_results['Predictions'], average='macro')

    # Check multimodal performance
    assert multimodal_accuracy > multimodal_accuracy_threshold, f"Multimodal accuracy is below {multimodal_accuracy_threshold}"
    assert multimodal_f1 > multimodal_f1_threshold, f"Multimodal F1 score is below {multimodal_f1_threshold}"

    # Check text performance
    assert text_accuracy > text_accuracy_threshold, f"Text accuracy is below {text_accuracy_threshold}"
    assert text_f1 > text_f1_threshold, f"Text F1 score is below {text_f1_threshold}"

    # Check image performance
    assert image_accuracy > image_accuracy_threshold, f"Image accuracy is below {image_accuracy_threshold}"
    assert image_f1 > image_f1_threshold, f"Image F1 score is below {image_f1_threshold}"


if __name__ == "__main__":
    pytest.main()
