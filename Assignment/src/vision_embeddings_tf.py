import numpy as np
import pandas as pd
import os
from transformers import TFConvNextV2Model, TFViTModel, TFSwinModel
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, InceptionV3
)
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image.
    
    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired image size.
    
    Returns:
    - np.array: Preprocessed image.
    """
    # TODO: Open the image using PIL Image.open and convert it to RGB format
    img = None
    # TODO: Resize the image to the target size
    img = None
    # TODO: Convert the image to a numpy array and scale the pixel values to [0, 1]
    img = None

    return img


class FoundationalCVModel:
    """
    A Keras module for loading and using foundational computer vision models.

    This class allows you to load and use various foundational computer vision models for tasks like image classification
    or feature extraction. The user can choose between evaluation mode (non-trainable model) and fine-tuning mode (trainable model).

    Attributes:
    ----------
    backbone_name : str
        The name of the foundational CV model to load (e.g., 'resnet50', 'vit_base').
    model : keras.Model
        The compiled Keras model with the selected backbone.
    
    Parameters:
    ----------
    backbone : str
        The name of the foundational CV model to load. The available backbones can include:
        - ResNet variants: 'resnet50', 'resnet101'
        - DenseNet variants: 'densenet121', 'densenet169'
        - InceptionV3: 'inception_v3'
        - ConvNextV2 variants: 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large'
        - Swin Transformer variants: 'swin_tiny', 'swin_small', 'swin_base'
        - Vision Transformer (ViT) variants: 'vit_base', 'vit_large'
    
    mode : str, optional
        The mode of the model, either 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    -------
    __init__(self, backbone, mode='eval'):
        Initializes the model with the specified backbone and mode.
    
    predict(self, images):
        Given a batch of images, performs a forward pass through the model and returns predictions.
        Parameters:
        ----------
        images : numpy.ndarray
            A batch of images to perform prediction on, with shape (batch_size, 224, 224, 3).
        
        Returns:
        -------
        numpy.ndarray
            Model predictions or extracted features for the provided images.
    """
    
    def __init__(self, backbone, mode='eval', input_shape=(224, 224, 3)):
        self.backbone_name = backbone
        
        # Select the backbone from the possible foundational models
        input_layer = Input(shape=input_shape)
        
        
        if backbone == 'resnet50':
            # TODO: Load the ResNet50 model from tensorflow.keras.applications
            self.base_model = None
        elif backbone == 'resnet101':
            # TODO: Load the ResNet101 model from tensorflow.keras.applications
            self.base_model = None
        elif backbone == 'densenet121':
            # TODO: Load the DenseNet121 model from tensorflow.keras.applications
            self.base_model = None
        elif backbone == 'densenet169':
            # TODO: Load the DenseNet169 model from tensorflow.keras.applications
            self.base_model = None
        elif backbone == 'inception_v3':
            # TODO: Load the InceptionV3 model from tensorflow.keras.applications
            self.base_model = None
        elif backbone == 'convnextv2_tiny':
            # TODO: Load the ConvNeXtV2 Tiny model from transformers
            self.base_model = None
        elif backbone == 'convnextv2_base':
            # TODO: Load the ConvNeXtV2 Base model from transformers
            self.base_model = None
        elif backbone == 'convnextv2_large':
            # TODO: Load the ConvNeXtV2 Large model from transformers
            self.base_model = None
        elif backbone == 'swin_tiny':
            # TODO: Load the Swin Transformer Tiny model from transformers
            self.base_model = None
        elif backbone == 'swin_small':
            # TODO: Load the Swin Transformer Small model from transformers
            self.base_model = None
        elif backbone == 'swin_base':
            # TODO: Load the Swin Transformer Base model from transformers
            self.base_model = None
        elif backbone in ['vit_base', 'vit_large']:
            # TODO: Load the Vision Transformer (ViT) model from transformers
            backbone_path = {
                'vit_base': None,
                'vit_large': None,
            }
            self.base_model = None
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")

        
        if mode == 'eval':
            # TODO: Set the model to evaluation mode (non-trainable)
            self.base_model = None
        
        # Take into account the model's input requirements. In models from transformers, the input is channels first, but in models from keras.applications, the input is channels last.
        # Aditionally, the output of the model is different in both cases, we need to get the pooling of the output layer.
        
        # If is a model from transformers:
        if backbone in ['vit_base', 'vit_large', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'swin_tiny', 'swin_small', 'swin_base']:
            # TODO: Adjust the input for channels first models within the model
            # You can use the perm argument of tf.transpose to permute the dimensions of the input tensor
            input_layer_transposed = None
            # TODO: Get the pooling output of the model "pooler_output"
            outputs = None
        # If is a model from keras.applications:
        else:
            # TODO: Get the pooling output of the model
            # In this case the pooling layer is not included in the model, we can use a pooling layer such as GlobalAveragePooling2D
            outputs = None
        
        # TODO: Create the final model with the input layer and the pooling output
        self.model = Model()
    
    def predict(self, images):
        """
        Predict on a batch of images.

        Parameters:
        ----------
        images : numpy.ndarray
            A batch of images of shape (batch_size, 224, 224, 3).

        Returns:
        -------
        numpy.ndarray
            Predictions or features from the model for the given images.
        """
        # TODO: Perform a forward pass through the model and return the predictions
        preds = None
        return preds



class ImageFolderDataset:
    """
    A custom dataset class for loading and preprocessing images from a folder.

    This class helps in loading images from a given folder, automatically filtering valid image files and 
    preprocessing them to a specified shape. It also handles any unreadable or corrupted images by excluding them.

    Attributes:
    ----------
    folder_path : str
        The path to the folder containing the images.
    shape : tuple
        The desired shape (width, height) to which the images will be resized.
    image_files : list
        A list of valid image file names that can be processed.

    Parameters:
    ----------
    folder_path : str
        The path to the folder containing image files.
    shape : tuple, optional
        The target shape to resize the images to. The default value is (224, 224).
    image_files : list, optional
        A pre-provided list of image file names. If not provided, it will automatically detect valid image files
        (with extensions '.jpg', '.jpeg', '.png', '.gif') in the specified folder.

    Methods:
    -------
    clean_unidentified_images():
        Cleans the dataset by removing images that cause an `UnidentifiedImageError` during loading. This helps ensure
        that only valid, readable images are kept in the dataset.
    
    __len__():
        Returns the number of valid images in the dataset after cleaning.

    __getitem__(idx):
        Given an index `idx`, retrieves the image file at that index, loads and preprocesses it, and returns the image 
        along with its filename.
    
    """
    def __init__(self, folder_path, shape=(224, 224), image_files=None):
        """
        Initializes the dataset object by setting the folder path and target image shape. 
        It also optionally accepts a list of image files to be processed, otherwise detects valid images in the folder.

        Parameters:
        ----------
        folder_path : str
            The directory containing the images.
        shape : tuple, optional
            The target shape to resize the images to. Default is (224, 224).
        image_files : list, optional
            A list of image files to load. If not provided, it will auto-detect valid images from the folder.
        """
        self.folder_path = folder_path
        self.shape = shape
        
        # If image files are provided, use them; otherwise, detect image files in the folder
        if image_files:
            self.image_files = image_files
        else:
            # List all files in the folder and filter only image files
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]
        
        # Clean the dataset by removing images that cause errors during loading
        self.clean_unidentified_images()
        
    def clean_unidentified_images(self):
        """
        Clean the dataset by removing images that cannot be opened due to errors (e.g., `UnidentifiedImageError`).
        
        This method iterates over the list of detected image files and attempts to open and convert each image to RGB.
        If an image cannot be opened (e.g., due to corruption or unsupported format), it is excluded from the dataset.

        Any image that causes an error will be skipped, and a message will be printed to indicate which file was skipped.
        """
        cleaned_files = []
        # Iterate over the image files and check if they can be opened
        for img_name in self.image_files:
            img_path = os.path.join(self.folder_path, img_name)
            try:
                # Try to open the image and convert it to RGB format
                Image.open(img_path).convert("RGB")
                # If successful, add the image to the cleaned list
                cleaned_files.append(img_name)
            except:
                print(f"Skipping {img_name} due to error")
                
        # Update the list of image files with only the cleaned files
        self.image_files = cleaned_files
    
    def __len__(self):
        """
        Returns the number of valid images in the dataset after cleaning.

        Returns:
        -------
        int
            The number of images in the cleaned dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves the image and its filename at the specified index.

        Parameters:
        ----------
        idx : int
            The index of the image to retrieve.
        
        Returns:
        -------
        tuple
            A tuple containing the image filename and the preprocessed image as a NumPy array or Tensor.
        
        Raises:
        ------
        IndexError
            If the index is out of bounds for the dataset.
        """
        # Get an item from the list of image files
        img_name = self.image_files[idx]
        # Load and preprocess the image:
        img_path = os.path.join(self.folder_path, img_name)
        img = load_and_preprocess_image(img_path, self.shape)
        # Return the image filename and the preprocessed image
        return img_name, img

def get_embeddings_df(batch_size=32, path="data/images", dataset_name='', backbone="resnet50", directory='Embeddings', image_files=None):
    """
    Generates embeddings for images in a dataset using a specified backbone model and saves them to a CSV file.
    
    This function processes images from a given folder in batches, extracts features (embeddings) using a specified 
    pre-trained computer vision model, and stores the results in a CSV file. The embeddings can be used for 
    downstream tasks such as image retrieval or clustering.

    Parameters:
    ----------
    batch_size : int, optional
        The number of images to process in each batch. Default is 32.
    path : str, optional
        The folder path containing the images. Default is "data/images".
    dataset_name : str, optional
        The name of the dataset to create subdirectories for saving embeddings. Default is an empty string.
    backbone : str, optional
        The name of the backbone model to use for generating embeddings. The default is 'resnet50'.
        Other possible options include models like 'convnext_tiny', 'vit_base', etc.
    directory : str, optional
        The root directory where the embeddings CSV file will be saved. Default is 'Embeddings'.
    image_files : list, optional
        A pre-defined list of image file names to process. If not provided, the function will automatically detect 
        image files in the `path` directory.

    Returns:
    -------
    None
        The function does not return any value. It saves a CSV file containing image names and their embeddings.

    Side Effects:
    ------------
    - Saves a CSV file in the specified directory containing image file names and their corresponding embeddings.
    
    Notes:
    ------
    - The images are loaded and preprocessed using the `ImageFolderDataset` class.
    - The embeddings are generated using a pre-trained model from the `FoundationalCVModel` class.
    - The embeddings are saved as a CSV file with the following structure:
        - `ImageName`: The name of the image file.
        - Columns corresponding to the embedding vector (one column per feature).
    
    Example:
    --------
    >>> get_embeddings_df(batch_size=16, path="data/images", dataset_name='sample_dataset', backbone="resnet50")
    
    This would generate a CSV file with image embeddings from the 'resnet50' backbone model for images in the "data/images" directory.
    """
    
    # Create an instance of the ImageFolderDataset class
    dataset = ImageFolderDataset(folder_path=path, image_files=image_files)
    # Create an instance of the FoundationalCVModel class
    model = FoundationalCVModel(backbone)
    
    img_names = []
    features = []
    # Calculate the number of batches based on the dataset size and batch size
    num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    
    # Process images in batches and extract features
    for i in range(0, len(dataset), batch_size):
        # Get the image files and images for the current batch
        batch_files = dataset.image_files[i:i + batch_size]
        batch_imgs = np.array([dataset[j][1] for j in range(i, min(i + batch_size, len(dataset)))])
        
        # Generate embeddings for the batch of images
        batch_features = model.predict(batch_imgs)
        
        # Append the image names and features to the lists
        img_names.extend(batch_files)
        features.extend(batch_features)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Batch {i // batch_size + 1}/{num_batches} done")
    
    # Create a DataFrame with the image names and embeddings
    df = pd.DataFrame({
        'ImageName': img_names,
        'Embeddings': features
    })
    
    # Split the embeddings into separate columns
    df_aux = pd.DataFrame(df['Embeddings'].tolist())
    df = pd.concat([df['ImageName'], df_aux], axis=1)
    
    # Save the DataFrame to a CSV file
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(f'{directory}/{dataset_name}'):
        os.makedirs(f'{directory}/{dataset_name}')
        
    df.to_csv(f'{directory}/{dataset_name}/Embeddings_{backbone}.csv', index=False)
