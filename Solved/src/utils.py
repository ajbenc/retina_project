import pandas as pd

import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")



######### Merge Datasets #########
def process_embeddings(df, col_name):
    """
    Process embeddings in a DataFrame column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the embeddings column.
    - col_name (str): The name of the column containing the embeddings.

    Returns:
    pd.DataFrame: The DataFrame with processed embeddings.

    Steps:
    1. Convert the values in the specified column to lists.
    2. Extract values from lists and create new columns for each element.
    3. Remove the original embeddings column.

    Example:
    df_processed = process_embeddings(df, 'embeddings')
    """
    # Convert the values in the column to lists
    df[col_name] = df[col_name].apply(eval)

    # Extract values from lists and create new columns
    embeddings_df = pd.DataFrame(df[col_name].to_list(), columns=[f"text_{i+1}" for i in range(df[col_name].str.len().max())])
    df = pd.concat([df, embeddings_df], axis=1)

    # Remove the original "embeddings" column
    df = df.drop(columns=[col_name])

    return df

def rename_image_embeddings(df):
    """
    Rename columns in a DataFrame for image embeddings.

    Args:
    - df (pd.DataFrame): The DataFrame containing columns to be renamed.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.

    Example:
    df_renamed = rename_image_embeddings(df)
    """
    df.columns = [f'image_{int(col)}' if col.isdigit() else col for col in df.columns]

    return df

# Preprocess and merge the dataframes
def preprocess_data(text_data, image_data, text_id="image_id", image_id="ImageName", embeddings_col = 'embeddings'):
    """
    Preprocess and merge text and image dataframes.

    Args:
    - text_data (pd.DataFrame): DataFrame containing text data.
    - image_data (pd.DataFrame): DataFrame containing image data.
    - text_id (str): Column name for text data identifier.
    - image_id (str): Column name for image data identifier.
    - embeddings_col (str): Column name for embeddings data.

    Returns:
    pd.DataFrame: Merged and preprocessed DataFrame.

    This function:
    Process text and image embeddings.
    Convert image_id and text_id values to integers.
    Merge dataframes using id.
    Drop unnecessary columns.

    Example:
    merged_df = preprocess_data(text_df, image_df)
    """
    text_data = process_embeddings(text_data, embeddings_col)
    image_data = rename_image_embeddings(image_data)    

    # drop missing values in image id
    image_data = image_data.dropna(subset=[image_id])
    text_data = text_data.dropna(subset=[text_id])

    text_data[text_id] = text_data[text_id].apply(lambda x: x.split('/')[-1])
    
    # Merge dataframes using image_id
    df = pd.merge(text_data, image_data, left_on=text_id, right_on=image_id)

    # Drop unnecessary columns
    df.drop([image_id, text_id], axis=1, inplace=True)

    return df



class ImageDownloader:
    """
    Image downloader class to download images from URLs.
    
    Args:
    - image_dir (str): Directory to save images.
    - image_size (tuple): Size of the images to be saved.
    - override (bool): Whether to override existing images.
    
    Methods:
    - download_images(df, print_every=1000): Download images from URLs in a DataFrame.
        Args:
        - df (pd.DataFrame): DataFrame containing image URLs.
        - print_every (int): Print progress every n images.
        Returns:
        pd.DataFrame: DataFrame with image paths added.
    
    Example:
    downloader = ImageDownloader()
    df = downloader.download_images(df)
    """
    def __init__(self, image_dir='data/images/', image_size=(224, 224), overwrite=False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.overwrite = overwrite
        
        # Create the directory if it doesn't exist
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
    def download_images(self, df, print_every=1000):
        image_paths = []
        
        i = 0
        for index, row in df.iterrows():
            if i % print_every == 0:
                print(f"Downloading image {i}/{len(df)}")
                i += 1
            
            sku = row['sku']
            image_url = row['image']
            image_path = os.path.join(self.image_dir, f"{sku}.jpg")
            
            if os.path.exists(image_path) and not self.overwrite:
                print(f"Image {sku} is already in the path.")
                image_paths.append(image_path)
                continue
            
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                img.save(image_path)
                #print(f"Downloaded image for SKU: {sku}")
                image_paths.append(image_path)
            except Exception as e:
                print(f"Could not download image for SKU: {sku}. Error: {e}")
                image_paths.append(np.nan)
        
        df['image_path'] = image_paths
        return df
