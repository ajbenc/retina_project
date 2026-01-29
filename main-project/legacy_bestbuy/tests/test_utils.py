import pytest
import numpy as np
import os
import pandas as pd

from src.utils import preprocess_data, train_test_split_and_feature_extraction
from sklearn.model_selection import train_test_split


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
    assert label_columns == ['class_id'], "The label column extraction is incorrect, should be 'class_id'"

    # Check if 'image' is in the columns
    assert 'image' not in image_columns, "'image' column is not part of the embedding columns"
        
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


if __name__ == "__main__":
    pytest.main()