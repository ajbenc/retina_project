import pytest
import numpy as np
from PIL import Image
import os
import pandas as pd

from src.vision_embeddings_tf import load_and_preprocess_image, FoundationalCVModel, get_embeddings_df

from tensorflow.keras.applications import ResNet50
from transformers import TFConvNextV2Model


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


if __name__ == "__main__":
    pytest.main()