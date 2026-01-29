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

    # Ensure that the output shape matches the expected output shape without using the model.predict method
    output = model.get_output_shape()

    assert output == (None, *expected_output_shape), f"Expected output shape (None, {expected_output_shape}), got {output}"


if __name__ == "__main__":
    pytest.main()