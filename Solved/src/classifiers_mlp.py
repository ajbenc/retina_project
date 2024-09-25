import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from itertools import cycle
from sklearn.preprocessing import LabelEncoder

# Custom Dataset class for Keras
class MultimodalDataset(Sequence):
    """
    Custom Keras Dataset class for multimodal data handling, designed for models that
    take both text and image data as inputs. It facilitates batching and shuffling 
    of data for efficient training in Keras models.

    This class supports loading and batching multimodal data (text and images), as well as handling
    label encoding. It is compatible with Keras and can be used to train models that require both
    text and image inputs. It also supports optional shuffling at the end of each epoch for better
    training performance.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset with text, image, and label columns.
        text_cols (list): List of column names corresponding to text data. Can be a single column or multiple columns.
        image_cols (list): List of column names corresponding to image data (usually file paths or image pixel data).
        label_col (str): Column name corresponding to the target labels.
        encoder (LabelEncoder, optional): A pre-fitted LabelEncoder instance for encoding the labels. 
                                          If None, a new LabelEncoder is fitted based on the provided data.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the dataset at the end of each epoch. Default is True.

    Attributes:
        text_data (np.ndarray): Array of text data from the DataFrame. None if `text_cols` is not provided.
        image_data (np.ndarray): Array of image data from the DataFrame. None if `image_cols` is not provided.
        labels (np.ndarray): One-hot encoded labels corresponding to the dataset's classes.
        encoder (LabelEncoder): Fitted LabelEncoder used to encode target labels.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Flag indicating whether to shuffle the data after each epoch.
        indices (np.ndarray): Array of indices representing the dataset. Used for shuffling batches.

    Methods:
    -------
    __len__():
        Returns the number of batches per epoch based on the dataset size and batch size.
    
    __getitem__(idx):
        Retrieves a single batch of data, including both text and image inputs and the corresponding labels.
        The method returns a tuple in the format ({'text': text_batch, 'image': image_batch}, label_batch),
        where 'text' and 'image' are only included if their respective columns were provided.

    on_epoch_end():
        Updates the index order after each epoch, shuffling if needed.
    """

    def __init__(self, df, text_cols, image_cols, label_col, encoder=None, batch_size=32, shuffle=True):
        """
        Initializes the MultimodalDataset object.

        Args:
            df (pd.DataFrame): The dataset as a DataFrame, containing text, image, and label data.
            text_cols (list): List of column names representing text features.
            image_cols (list): List of column names representing image features (e.g., file paths or pixel data).
            label_col (str): Column name corresponding to the target labels.
            encoder (LabelEncoder, optional): LabelEncoder for encoding the target labels. If None, a new LabelEncoder will be created.
            batch_size (int, optional): Batch size for loading data. Default is 32.
            shuffle (bool, optional): Whether to shuffle the data at the end of each epoch. Default is True.

        Raises:
            ValueError: If both text_cols and image_cols are None or empty.
        """
        if text_cols:
            # TODO: Get the text data from the DataFrame as a NumPy array
            self.text_data = df[text_cols].values
        else:
            # Else, set text data to None
            self.text_data = None
            
        if image_cols:
            # TODO: Get the image data from the DataFrame as a NumPy array
            self.image_data = df[image_cols].values
        else:
            # Else, set image data to None
            self.image_data = None
            
        if not text_cols and not image_cols:
            raise ValueError("At least one of text_cols or image_cols must be provided.")
        
        # TODO: Get the labels from the DataFrame and encode them
        self.labels = df[label_col].values

        # Use provided encoder or fit a new one
        if encoder is None:
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(self.labels)
        else:
            self.encoder = encoder
            self.labels = self.encoder.transform(self.labels)
        
        # One-hot encode labels for multi-class classification
        num_classes = len(self.encoder.classes_)
        self.labels = np.eye(num_classes)[self.labels]
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch based on the dataset size and batch size.

        Returns:
        -------
        int:
            The number of batches per epoch.
        """
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        """
        Retrieves a single batch of data (text and/or image) and the corresponding labels.
        
        Args:
            idx (int): Index of the batch to retrieve.
        
        Returns:
        -------
        tuple:
            A tuple containing the batch of text and/or image inputs and the corresponding labels. 
            The input data is returned as a dictionary with keys 'text' and 'image', depending on the provided columns.
            If no text or image columns were provided, only the other is returned.
        """
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.text_data is not None:
            text_batch = self.text_data[indices]
        if self.image_data is not None:
            image_batch = self.image_data[indices]
        label_batch = self.labels[indices]
        
        if self.text_data is None:
            return {'image': image_batch}, label_batch
        if self.image_data is None:
            return {'text': text_batch}, label_batch
        else:
            return {'text': text_batch, 'image': image_batch}, label_batch

    def on_epoch_end(self):
        """
        Updates the index order after each epoch, shuffling the data if needed.
        
        This method is called at the end of each epoch and will shuffle the data if the `shuffle` flag is set to True.
        """
        self.indices = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indices)

# Early Fusion Model
def create_early_fusion_model(text_input_size, image_input_size, output_size, hidden=[128], p=0.2):
    """
    Creates a multimodal early fusion model combining text and image inputs. The model concatenates the text and 
    image features, passes them through fully connected layers with optional dropout and batch normalization, 
    and produces a multi-class classification output.

    Args:
        text_input_size (int): Size of the input vector for the text data.
        image_input_size (int): Size of the input vector for the image data.
        output_size (int): Number of classes for the output layer (i.e., size of the softmax output).
        hidden (int or list, optional): Specifies the number of hidden units in the dense layers. 
                                        If an integer, a single dense layer with the specified units is created. 
                                        If a list, multiple dense layers are created with the respective units. Default is [128].
        p (float, optional): Dropout rate to apply after each dense layer. Default is 0.2.

    Returns:
        Model (keras.Model): A compiled Keras model with text and image inputs and a softmax output for classification.

    Model Architecture:
        - The model accepts two inputs: one for text features and one for image features.
        - The features are concatenated into a single vector.
        - Dense layers with ReLU activation are applied, followed by dropout and batch normalization (if multiple hidden layers are specified).
        - The output layer uses a softmax activation for multi-class classification.

    Example:
        model = create_early_fusion_model(text_input_size=300, image_input_size=2048, output_size=10, hidden=[128, 64], p=0.3)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    """
    
    
    if text_input_size is None and image_input_size is None:
        raise ValueError("At least one of text_input_size and image_input_size must be provided.")
    
    if text_input_size is not None:
        # TODO: Define text input layer for only text data
        text_input = Input(shape=(text_input_size,), name='text')
    if image_input_size is not None:
        # TODO: Define image input layer for only image data
        image_input = Input(shape=(image_input_size,), name='image')
    
    
    if text_input_size is not None and image_input_size is not None:
        # TODO: Concatenate text and image inputs if both are provided
        x = Concatenate()([text_input, image_input])
    elif text_input_size is not None:
        x = text_input
    elif image_input_size is not None:
        x = image_input

    if isinstance(hidden, int):
        # TODO: Add a single dense layer 
        # Optionally play with activation, dropout and normalization
        x = Dense(hidden, activation='relu')(x)
        x = Dropout(p)(x)
    elif isinstance(hidden, list):
        for h in hidden:
            # TODO: Add multiple dense layers based on the hidden list
            # Optionally play with activation, dropout and normalization
            x = Dense(h, activation='relu')(x)
            x = Dropout(p)(x)
            x = BatchNormalization()(x)

    # TODO: Add the output layer with softmax activation
    output = Dense(output_size, activation='softmax')(x)

    # Create the model
    if text_input_size is not None and image_input_size is not None:
        # TODO: Define the model with both text and image inputs
        model = Model(inputs=[text_input, image_input], outputs=output)
    elif text_input_size is not None:
        # TODO: Define the model with only text input
        model = Model(inputs=text_input, outputs=output)
    elif image_input_size is not None:
        # TODO: Define the model with only image input
        model = Model(inputs=image_input, outputs=output)
    else:
        raise ValueError("At least one of text_input_size and image_input_size must be provided.")
    
    return model

def test_model(y_test, y_pred, y_prob=None, encoder=None):
    """
    Evaluates a trained model's performance using various metrics such as accuracy, precision, recall, F1-score, 
    and visualizations including a confusion matrix and ROC curves.

    Args:
        y_test (np.ndarray): Ground truth one-hot encoded labels for the test data.
        y_pred (np.ndarray): Predicted class labels by the model for the test data (after argmax transformation).
        y_prob (np.ndarray, optional): Predicted probabilities for each class from the model. Required for ROC curves. Default is None.
        encoder (LabelEncoder, optional): A fitted LabelEncoder instance used to inverse transform one-hot encoded and predicted labels to their original categorical form.

    Returns:
        accuracy (float): Accuracy score of the model on the test data.
        precision (float): Weighted precision score of the model on the test data.
        recall (float): Weighted recall score of the model on the test data.
        f1 (float): Weighted F1 score of the model on the test data.

    This function performs the following steps:
        - Inverse transforms the one-hot encoded `y_test` and predicted `y_pred` values to their original labels using the provided LabelEncoder.
        - Computes the confusion matrix and plots it as a heatmap using Seaborn.
        - If `y_prob` is provided, computes and plots the ROC curves for each class.
        - Prints the classification report, which includes precision, recall, F1-score, and support for each class.
        - Returns the overall accuracy, weighted precision, recall, and F1-score of the model.

    Visualizations:
        - Confusion Matrix: A heatmap of the confusion matrix comparing the true labels with the predicted labels.
        - ROC Curves: Plots ROC curves for each class if predicted probabilities are provided (`y_prob`).

    Example:
        accuracy, precision, recall, f1 = test_model(y_test, y_pred, y_prob, encoder)
    """
    y_test_binarized = y_test
    y_test = encoder.inverse_transform(np.argmax(y_test, axis=1))
    y_pred = encoder.inverse_transform(y_pred)
    

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    if y_prob is not None:
        fig, ax = plt.subplots(figsize=(15, 15))
        
        colors = cycle(["aqua", "darkorange", "cornflowerblue"])

        for i, color in zip(range(y_prob.shape[1]), colors):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {i}')

        ax.plot([0, 1], [0, 1], "k--")
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show()

    cr = classification_report(y_test, y_pred)
    print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1


    
def train_mlp(train_loader, test_loader, text_input_size, image_input_size, output_size, num_epochs=50, report=False, lr=0.001, set_weights=True, adam=False, p=0.0, seed=1, patience=40, save_results=True, train_model=True, test_mlp_model=True):
    """
    Trains a multimodal early fusion model using both text and image data.

    The function handles the training process of the model by combining text and image features,
    computes class weights if needed, applies an optimizer (SGD or Adam), and implements early stopping 
    to prevent overfitting. The model is evaluated on the test set, and key performance metrics are computed.

    Args:
        train_loader (MultimodalDataset): Keras-compatible data loader for the training set with both text and image data.
        test_loader (MultimodalDataset): Keras-compatible data loader for the test set with both text and image data.
        text_input_size (int): The size of the input vector for the text data.
        image_input_size (int): The size of the input vector for the image data.
        output_size (int): Number of output classes for the softmax layer.
        num_epochs (int, optional): Number of training epochs. Default is 50.
        report (bool, optional): Whether to generate a detailed classification report and display metrics. Default is False.
        lr (float, optional): Learning rate for the optimizer. Default is 0.001.
        set_weights (bool, optional): Whether to compute and apply class weights to handle imbalanced datasets. Default is True.
        adam (bool, optional): Whether to use the Adam optimizer instead of SGD. Default is False.
        p (float, optional): Dropout rate for regularization in the model. Default is 0.0.
        seed (int, optional): Seed for random number generators to ensure reproducibility. Default is 1.
        patience (int, optional): Number of epochs with no improvement on validation loss before early stopping. Default is 40.

    Returns:
        None

    Side Effects:
        - Trains the early fusion model and saves the best weights based on validation loss.
        - Generates plots showing the training and validation accuracy over epochs.
        - If `report` is True, calls `test_model` to print detailed evaluation metrics and plots.

    Training Process:
        - The function creates a fusion model combining text and image inputs.
        - Class weights are computed to balance the dataset if `set_weights` is True.
        - The model is trained using categorical cross-entropy loss and the chosen optimizer (Adam or SGD).
        - Early stopping is applied based on validation loss to prevent overfitting.
        - After training, the model is evaluated on the test set, and accuracy, F1-score, and AUC are calculated.

    Example:
        train_mlp(train_loader, test_loader, text_input_size=300, image_input_size=2048, output_size=10, num_epochs=30, lr=0.001, adam=True, report=True)

    Notes:
        - `train_loader` and `test_loader` should be instances of `MultimodalDataset` or compatible Keras data loaders.
        - If the dataset is imbalanced, setting `set_weights=True` is recommended to ensure better model performance on minority classes.
    """
    
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
      
    # Create an instance of the early fusion model  
    # TODO: Create an early fusion model using the provided input sizes and output size
    model = create_early_fusion_model(text_input_size, image_input_size, output_size, p=p)

    # Compute class weights for imbalanced datasets
    if set_weights:
        class_indices = np.argmax(train_loader.labels, axis=1)
        # TODO: Compute class weights using the training labels
        # You should use the `compute_class_weight` function from scikit-learn.
        class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
        class_weights = {i: weight for i, weight in enumerate(class_weights)}

    # TODO: Choose the loss function for multi-class classification
    loss = CategoricalCrossentropy()

    # Choose the optimizer
    if adam:
        # TODO: Use the Adam optimizer with the specified learning rate
        optimizer = Adam(lr=lr)
    else:
        # TODO: Use the SGD optimizer with the specified learning rate
        optimizer = SGD(lr=lr)

    # TODO: Compile the model with the chosen optimizer and loss function
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # TODO: Define an early stopping callback with the specified patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # TODO: Train the model using the training data and validation data
    # Use the class weights if set_weights
    # Use the early stopping callback
    # Use the number of epochs specified
    if train_model:
        history = model.fit(train_loader, validation_data=test_loader, epochs=num_epochs, class_weight=class_weights, callbacks=[early_stopping])

    if test_mlp_model:
        # Test the model on the test set
        y_true, y_pred, y_prob = [], [], []
        for batch in test_loader:
            features, labels = batch
            if len(features) == 1:
                text = features['text'] if 'text' in features else features['image']
                preds = model.predict(text)
            else:
                text, image = features['text'], features['image']
                preds = model.predict([text, image])
            y_true.extend(labels)
            y_pred.extend(np.argmax(preds, axis=1))
            y_prob.extend(preds)

        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

        test_accuracy = accuracy_score(np.argmax(y_true, axis=1), y_pred)
        f1 = f1_score(np.argmax(y_true, axis=1), y_pred, average='macro')
        
        auc_scores = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
        macro_auc = auc_scores

        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        if report:
            test_model(y_true, y_pred, y_prob, encoder=train_loader.encoder)
        
        # Store results in a dataframe and save in the results folder
        if text_input_size is not None and image_input_size is not None:
            model_type = 'multimodal'
        elif text_input_size is not None:
            model_type = 'text'
        elif image_input_size is not None:
            model_type = 'image'
        
        if save_results:
            results = pd.DataFrame({'Predictions': y_pred, 'True Labels': np.argmax(y_true, axis=1)})
            # create results folder if it does not exist
            os.makedirs('results', exist_ok=True)
            results.to_csv(f"results/{model_type}_results.csv", index=False)
    else:
        test_accuracy, f1, macro_auc = None, None, None
        
    return model, test_accuracy, f1, macro_auc