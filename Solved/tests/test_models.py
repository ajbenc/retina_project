import pytest
from unittest.mock import patch
import numpy as np
from PIL import Image
import os
import pandas as pd

from src.vision_embeddings_tf import load_and_preprocess_image, FoundationalCVModel, get_embeddings_df

from tensorflow.keras.applications import ResNet50
from transformers import TFConvNextV2Model

import torch
from transformers import AutoTokenizer, AutoModel
from src.nlp_models import HuggingFaceEmbeddings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.classifiers_classic_ml import visualize_embeddings, train_and_evaluate_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from src.classifiers_mlp import MultimodalDataset, train_mlp, create_early_fusion_model

from src.utils import preprocess_data, train_test_split_and_feature_extraction
from sklearn.model_selection import train_test_split

####################################################################################################
#################### Test the foundational CV model and image preprocessing ########################
####################################################################################################
@pytest.fixture
def mock_image(tmp_path):
    """
    Fixture to create a mock image for testing.
    """
    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (300, 300), color='red')
    img.save(img_path)
    return str(img_path)


def test_load_and_preprocess_image(mock_image):
    """
    Test loading and preprocessing of an image.
    """
    # Test the load_and_preprocess_image function
    img = load_and_preprocess_image(mock_image, target_size=(224, 224))
    
    # Check if the output is a numpy array
    assert isinstance(img, np.ndarray), "Output is not a numpy array"
    
    # Check if the image has the correct shape
    assert img.shape == (224, 224, 3), f"Image shape is {img.shape}, expected (224, 224, 3)"
    
    # Check if the pixel values are in the range [0, 1]
    assert img.min() >= 0 and img.max() <= 1, "Image pixel values are not in the range [0, 1]"


@pytest.mark.parametrize("backbone, expected_model_class, expected_output_shape", [
    ('resnet50', type(ResNet50()), (2048,)),  # Keras ResNet50 with 2048 features
    ('convnextv2_tiny', TFConvNextV2Model, (768,)),  # ConvNeXt V2 Tiny from Hugging Face with 768 features
])
def test_foundational_cv_model_generic(backbone, expected_model_class, expected_output_shape):
    """
    Generic test for loading a foundational CV model and making predictions.
    
    This test ensures that:
    - The correct backbone model is loaded.
    - The input shape matches the model's requirements (224x224x3).
    - The output embedding shape matches the expected shape for the backbone.

    Parameters:
    ----------
    backbone : str
        The name of the model backbone to test.
    expected_model_class : class
        The expected class of the loaded backbone model (e.g., ResNet50 or TFConvNextV2Model).
    expected_output_shape : tuple
        The expected shape of the output embedding vector.
    """
    # Initialize the model with the provided backbone
    model = FoundationalCVModel(backbone=backbone, mode='eval')
    
    # Check if the model is an instance of the expected model class
    assert isinstance(model.base_model, expected_model_class), f"Expected model class {expected_model_class}, got {type(model.model)}"
    
    # Create a batch of random images (2 images of shape 224x224x3)
    batch_images = np.random.rand(2, 224, 224, 3)
    
    # Ensure that the input shape matches the model's input requirements
    assert model.model.input_shape == (None, 224, 224, 3), f"Expected input shape (None, 224, 224, 3), got {model.model.input_shape}"

    # Predict the embeddings
    embeddings = model.predict(batch_images)
    
    # Ensure that the output embeddings are a NumPy array
    assert isinstance(embeddings, np.ndarray), "Output embeddings are not a NumPy array"
    
    # Ensure that the output embeddings have the correct shape
    assert embeddings.shape == (2, *expected_output_shape), f"Embedding shape is {embeddings.shape}, expected (2, {expected_output_shape})"

    # Print output shape
    print(f"Output shape for {backbone}: {embeddings.shape} ot type {type(embeddings)}")


@pytest.fixture
def mock_image_folder(tmp_path):
    """
    Fixture to create a mock image folder with a few test images.
    """
    folder_path = tmp_path / "images"
    os.makedirs(folder_path)

    # Create 3 test images
    for i in range(3):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        img_path = folder_path / f"test_image_{i}.jpg"
        Image.fromarray(img).save(img_path)

    return str(folder_path)


@pytest.mark.parametrize("backbone, expected_embedding_size", [
    ('resnet50', 2048),  # ResNet50 with 2048-dimensional embeddings
    ('convnextv2_tiny', 768),  # ConvNeXt V2 Tiny with 768-dimensional embeddings
])
def test_get_embeddings_df(mock_image_folder, tmp_path, backbone, expected_embedding_size):
    """
    Test the get_embeddings_df function by generating embeddings for mock images
    and ensuring the output DataFrame has the correct structure and dimensions.

    Parameters:
    ----------
    mock_image_folder : str
        The path to the folder containing mock test images.
    tmp_path : pathlib.Path
        Temporary path for saving the generated embeddings.
    backbone : str
        The name of the backbone model to use for embedding extraction.
    expected_embedding_size : int
        The expected size of the output embedding vector for the backbone model.
    """
    # Define the directory to save the embeddings CSV file
    output_dir = tmp_path / "embeddings_output"
    os.makedirs(output_dir, exist_ok=True)

    # Call the get_embeddings_df function to generate embeddings
    get_embeddings_df(batch_size=2, path=mock_image_folder, dataset_name='test_dataset', backbone=backbone, directory=str(output_dir))

    # Check if the CSV file has been created
    csv_path = output_dir / "test_dataset" / f"Embeddings_{backbone}.csv"
    assert os.path.exists(csv_path), f"Embeddings CSV file not created at {csv_path}"

    # Load the generated CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check if the DataFrame has the correct number of rows (3 images) and columns (1 for 'ImageName' + embedding size)
    assert df.shape == (3, expected_embedding_size + 1), f"Expected DataFrame shape (3, {expected_embedding_size + 1}), got {df.shape}"

    # Check if the 'ImageName' column contains the correct filenames
    for i in range(3):
        assert f"test_image_{i}.jpg" in df['ImageName'].values, f"Missing test_image_{i}.jpg in DataFrame"

    print(f"Embeddings DataFrame for {backbone} has shape {df.shape} \n{df.head()}")


####################################################################################################
################################## Test the Text Embeddings Model ##################################
####################################################################################################


@pytest.fixture
def mock_text_data(tmp_path):
    """
    Fixture to create a mock CSV file with text data for testing.
    """
    data = {
        "description": ["Product 1 description", "Product 2 description", "Product 3 description"]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_text_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.mark.parametrize("model_name, expected_hidden_size", [
    ('sentence-transformers/all-MiniLM-L6-v2', 384),  # MiniLM with 384 hidden units
    ('bert-base-uncased', 768),  # BERT base with 768 hidden units
])
def test_huggingface_embeddings_generic(model_name, expected_hidden_size, mock_text_data):
    """
    Generic test for loading a Hugging Face model, generating text embeddings, and saving them to a CSV file.

    This test ensures that:
    - The model and tokenizer are properly loaded from Hugging Face.
    - Embeddings are correctly generated for text descriptions.
    - Embeddings are saved in the correct format to a CSV file.

    Parameters:
    ----------
    model_name : str
        The name of the Hugging Face model to test.
    expected_hidden_size : int
        The expected hidden size (dimensionality) of the embeddings generated by the model.
    mock_text_data : str
        Path to the mock CSV file containing text descriptions.
    """
    # Initialize the HuggingFaceEmbeddings model with the provided model name
    model = HuggingFaceEmbeddings(model_name=model_name, path=mock_text_data, device='cpu')

    # Check that the tokenizer and model were loaded correctly
    assert isinstance(model.tokenizer, type(AutoTokenizer.from_pretrained(model_name))), f"Tokenizer should be an instance of {type(AutoTokenizer.from_pretrained(model_name))}"
    assert isinstance(model.model, type(AutoModel.from_pretrained(model_name))), f"Model should be an instance of {type(AutoModel.from_pretrained(model_name))}"

    # Generate embeddings for a sample text
    sample_text = "This is a test description."
    embeddings = model.get_embedding(sample_text)
    
    # Check that the embeddings are a NumPy array with the expected shape
    assert isinstance(embeddings, np.ndarray), "Embeddings should be a NumPy array"
    assert embeddings.shape == (expected_hidden_size,), f"Embeddings shape should be ({expected_hidden_size},), got {embeddings.shape}"

    # Test generating embeddings for a DataFrame and saving to CSV
    output_dir = "Embeddings_test"
    output_file = "test_text_embeddings.csv"
    model.get_embedding_df(column="description", directory=output_dir, file=output_file)

    # Load the saved CSV and check if embeddings are present
    saved_df = pd.read_csv(f"{output_dir}/{output_file}")
    
    assert "embeddings" in saved_df.columns, "The 'embeddings' column should be present in the saved DataFrame"
    assert len(saved_df["embeddings"]) == 3, "The number of embeddings should match the number of descriptions"

    # Check if the embeddings column contains valid embeddings (as a string representation of lists)
    for embedding in saved_df["embeddings"]:
        assert isinstance(eval(embedding), list), "Embeddings should be stored as lists in the CSV file"

    print(f"Embeddings DataFrame for {model_name} has shape {saved_df.shape} \n{saved_df.head()}")
    print(f"The embeddings in the final dataframe are lists of length {len(eval(saved_df['embeddings'][0]))}")

####################################################################################################
######################### Test the Train-Test Split and variable selection #########################
####################################################################################################


@pytest.fixture
def big_fake_data():
    # Create a fake dataset with 100 rows
    num_rows = 100
    num_image_columns = 10
    num_text_columns = 11

    data = {
        'id': np.arange(1, num_rows + 1),
        'image': [f'path/{i}.jpg' for i in range(1, num_rows + 1)],
    }

    # Add image_0 to image_9 columns
    for i in range(num_image_columns):
        data[f'image_{i}'] = np.random.rand(num_rows)

    # Add text_0 to text_10 columns
    for i in range(num_text_columns):
        data[f'text_{i}'] = np.random.rand(num_rows)

    # Add a class_id column
    data['class_id'] = np.random.choice(['label1', 'label2', 'label3'], size=num_rows)

    return pd.DataFrame(data)

def test_train_test_split_and_feature_extraction(big_fake_data):
    # Split the data and extract features and labels
    train_df, test_df, text_columns, image_columns, label_columns = train_test_split_and_feature_extraction(
        big_fake_data, test_size=0.3, random_state=42
    )

    # Check that the correct columns were identified
    assert text_columns == [f'text_{i}' for i in range(11)], "The text embedding columns extraction is incorrect"
    assert image_columns == [f'image_{i}' for i in range(10)], "The image embedding columns extraction is incorrect"
    assert label_columns == ['class_id'], "The label column extraction is incorrect"
    
    # Check the train-test split sizes (30% of 100 rows should be 70 train, 30 test)
    assert len(train_df) == 70, f"Train size should be 70%, but got {len(train_df)}%"
    assert len(test_df) == 30, f"Test size should be 30%, but got {len(test_df)}%"

    # Check random state consistency by ensuring the split results are reproducible
    expected_train_indices = train_df.index.tolist()
    expected_test_indices = test_df.index.tolist()

    # Re-run the function to check for consistency in split
    train_df_recheck, test_df_recheck, _, _, _ = train_test_split_and_feature_extraction(
        big_fake_data, test_size=0.3, random_state=42
    )

    assert expected_train_indices == train_df_recheck.index.tolist(), "Train set indices are not consistent with the random state"
    assert expected_test_indices == test_df_recheck.index.tolist(), "Test set indices are not consistent with the random state"



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
    # Create a synthetic dataset with 1000 samples, 20 features, and 3 classes
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=17, n_classes=3, random_state=42)
    
    # Split the dataset into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

@pytest.mark.parametrize("method, plot_type", [
    ('PCA', '2D'),  # PCA reduction to 2D
    ('t-SNE', '2D'),  # t-SNE reduction to 2D
    ('PCA', '3D'),  # PCA reduction to 3D
    ('t-SNE', '3D'),  # t-SNE reduction to 3D
])   
def test_visualize_embeddings(method, plot_type, sample_embedding_data):
    """
    Test the dimensionality reduction and embedding visualization.
    This ensures that PCA and t-SNE can reduce embeddings correctly and produce visualizations.
    """
    X_train, X_test, y_train, y_test = sample_embedding_data

    # Mock the plotly figures to avoid actual plotting in test environment
    with patch('plotly.graph_objs.Figure.show'):
        # Test the visualize_embeddings function
        model = visualize_embeddings(X_train, X_test, y_train, y_test, plot_type=plot_type, method=method)
    
    # Check if the PCA/t-SNE model is an instance of the correct class and has the expected number of components
    if method == 'PCA':
        assert isinstance(model, PCA), "The model should be an instance of PCA"
        if plot_type == '2D':
            assert model.n_components_ == 2, "PCA should reduce data to 2 components"
        elif plot_type == '3D':
            assert model.n_components_ == 3, "PCA should reduce data to 3 components"
    elif method == 't-SNE':
        assert isinstance(model, TSNE), "The model should be an instance of t-SNE"
        if plot_type == '2D':
            assert model.embedding_.shape[1] == 2, "t-SNE reduced data should have 2 components"
        elif plot_type == '3D':
            assert model.embedding_.shape[1] == 3, "t-SNE reduced data should have 3 components"
                

def test_train_and_evaluate_model(sample_embedding_data):
    """
    Test the training and evaluation of models (Logistic Regression, Random Forest).
    Ensures that models are correctly trained and returned in the expected format.
    """
    X_train, X_test, y_train, y_test = sample_embedding_data

    # Train and evaluate the models
    trained_models = train_and_evaluate_model(X_train, X_test, y_train, y_test)

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
    # Create synthetic multi-class data with 20 features (10 text-like, 10 image-like)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=17, n_classes=3, random_state=42)

    # Rename features to simulate text and image columns
    feature_names = [f'text_{i}' for i in range(10)] + [f'image_{i}' for i in range(10, 20)]
    
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

    # Image columns (the second 10 features)
    image_columns = [f'image_{i}' for i in range(10, 20)]
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

    # Text columns (the first 10 features)
    text_columns = [f'text_{i}' for i in range(10)]
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
    text_columns = [f'text_{i}' for i in range(10)]
    image_columns = [f'image_{i}' for i in range(10, 20)]
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
    image_input_size = 10
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
    text_input_size = 10
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
    text_input_size = 10
    image_input_size = 10
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
    image_columns = [f'image_{i}' for i in range(10, 20)]
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
        num_epochs=10,
        set_weights=True,
        adam=True, 
        patience=10
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

    # Check if the model is trained and evaluated correctly
    assert test_accuracy > 0.5, f"Test accuracy should be greater than 0.5, but got {test_accuracy}"
    assert f1 > 0.5, f"F1 score should be greater than 0.5, but got {f1}"
    assert macro_auc > 0.5, f"Macro AUC should be greater than 0.5, but got {macro_auc}"

def test_train_mlp_single_modality_text(correlated_sample_data, label_encoder):
    """
    Test the MLP training with only text data.
    Ensure the model trains and evaluates correctly.
    """
    train_df, test_df = correlated_sample_data

    # Text columns (the first 10 features)
    text_columns = [f'text_{i}' for i in range(10)]
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
        num_epochs=10,
        set_weights=True,
        adam=True, 
        patience=10
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

    # Check if the model is trained and evaluated correctly
    assert test_accuracy > 0.5, f"Test accuracy should be greater than 0.5, but got {test_accuracy}"
    assert f1 > 0.5, f"F1 score should be greater than 0.5, but got {f1}"
    assert macro_auc > 0.5, f"Macro AUC should be greater than 0.5, but got {macro_auc}"

def test_train_mlp_multimodal(correlated_sample_data, label_encoder):
    """
    Test the MLP training with class weights for an imbalanced dataset.
    Ensure class weights are applied correctly and early stopping works.
    """
    train_df, test_df = correlated_sample_data

    # Text and image columns
    text_columns = [f'text_{i}' for i in range(10)]
    image_columns = [f'image_{i}' for i in range(10, 20)]
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
        num_epochs=10,
        set_weights=True,
        adam=True, 
        patience=10
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

    # Check if the model is trained and evaluated correctly
    assert test_accuracy > 0.5, f"Test accuracy should be greater than 0.5, but got {test_accuracy}"
    assert f1 > 0.5, f"F1 score should be greater than 0.5, but got {f1}"
    assert macro_auc > 0.5, f"Macro AUC should be greater than 0.5, but got {macro_auc}"

if __name__ == "__main__":
    pytest.main()
