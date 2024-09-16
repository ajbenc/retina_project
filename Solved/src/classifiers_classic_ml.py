import pandas as pd

# Metrics
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, accuracy_score, precision_score, recall_score, f1_score, auc

# Plots:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from itertools import cycle

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

## Embeddings Visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

def visualize_embeddings(X_train, X_test, y_train, y_test, plot_type='2D', method='PCA'): 
    """
    Visualizes high-dimensional embeddings (e.g., text or image embeddings) using dimensionality reduction techniques (PCA or t-SNE) 
    and plots the results in 2D or 3D using Plotly for interactive visualizations.

    Args:
        X_train (np.ndarray): Training data embeddings of shape (n_samples, n_features).
        X_test (np.ndarray): Test data embeddings of shape (n_samples, n_features).
        y_train (np.ndarray): True labels for the training data.
        y_test (np.ndarray): True labels for the test data.
        plot_type (str, optional): Type of plot to generate, either '2D' or '3D'. Default is '2D'.
        method (str, optional): Dimensionality reduction method to use, either 'PCA' or 't-SNE'. Default is 'PCA'.

    Returns:
        None

    Side Effects:
        - Displays an interactive 2D or 3D scatter plot of the reduced embeddings, with points colored by their class labels.
    
    Notes:
        - PCA is a linear dimensionality reduction method, while t-SNE is non-linear and captures more complex relationships.
        - Perplexity is set to 10 for t-SNE. It can be tuned if necessary for better visualization of data clusters.
        - The function raises a `ValueError` if an invalid method is specified.
        - The function uses Plotly to display interactive plots.

    Example:
        visualize_embeddings(X_train, X_test, y_train, y_test, plot_type='3D', method='t-SNE')

    Visualization Details:
        - For 3D visualization, the reduced embeddings are plotted in a 3D scatter plot, with axes labeled as 'col1', 'col2', and 'col3'.
        - For 2D visualization, the embeddings are plotted in a 2D scatter plot, with axes labeled as 'col1' and 'col2'.
        - Class labels are represented by different colors in the scatter plots.
    """
    perplexity = 10

    if plot_type == '3D':
        if method == 'PCA':
            # TODO: Create an instance of PCA for 3D visualization and fit it on the training data
            red = PCA(n_components=3, random_state=42).fit(X_train)
            # TODO: Use the trained model to transform the test data
            reduced_embeddings = red.transform(X_test)
        elif method == 't-SNE':
            # TODO: Implement t-SNE for 3D visualization
            red = TSNE(n_components=3, perplexity=perplexity, random_state=42).fit(X_train)
            # TODO: Use the model to train and transform the test data
            reduced_embeddings = red.fit_transform(X_test)
        else:
            raise ValueError("Invalid method. Please choose either 'PCA' or 't-SNE'.")
        
            
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['col1', 'col2', 'col3'])
        df_reduced['Class'] = y_test

        # 3D scatter plot
        fig = px.scatter_3d(df_reduced, x='col1', y='col2', z='col3', color='Class', title='3D')
    
    else:
        if method == 'PCA':
            # TODO: Create an instance of PCA for 2D visualization and fit it on the training data
            red = PCA(n_components=2, random_state=42).fit(X_train)
            # TODO: Use the trained model to transform the test data
            reduced_embeddings = red.transform(X_test)
        elif method == 't-SNE':
            # TODO: Implement t-SNE for 2D visualization
            red = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit(X_train)
            # TODO: Use the model to train and transform the test data
            reduced_embeddings = red.fit_transform(X_test)
        else:
            raise ValueError("Invalid method. Please choose either 'PCA' or 't-SNE'.")
        
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['col1', 'col2'])
        df_reduced['Class'] = y_test

        # 2D scatter plot
        fig = px.scatter(df_reduced, x='col1', y='col2', color='Class', title='2D')
    
    fig.update_layout(
        title=f"Embeddings - {method} {plot_type} Visualization",
        scene=dict()
    )
    
    fig.show()
    
    return red


def test_model(X_test, y_test, model):
    """
    Evaluates a trained model on a test set by computing key performance metrics and visualizing the results.

    The function generates a confusion matrix, plots ROC curves (for binary or multi-class classification), 
    and prints the classification report. It also computes overall accuracy, weighted precision, weighted recall, 
    and weighted F1-score for the test data.

    Args:
        X_test (np.ndarray): Test set feature data.
        y_test (np.ndarray): True labels for the test set.
        model (sklearn-like model): A trained machine learning model with `predict` and `predict_proba` methods.

    Returns:
        tuple:
            - accuracy (float): Overall accuracy of the model on the test set.
            - precision (float): Weighted precision score across all classes.
            - recall (float): Weighted recall score across all classes.
            - f1 (float): Weighted F1-score across all classes.

    Side Effects:
        - Displays a confusion matrix as a heatmap.
        - Plots ROC curves for binary or multi-class classification.
        - Prints the classification report with precision, recall, F1-score, and support for each class.

    Example:
        accuracy, precision, recall, f1 = test_model(X_test, y_test, trained_model)

    Notes:
        - If `y_test` is multi-dimensional (e.g., one-hot encoded), it will be squeezed to 1D.
        - For binary classification, a single ROC curve is plotted. For multi-class classification, 
          an ROC curve is plotted for each class with a unique color.
        - Weighted precision, recall, and F1-score are computed to handle class imbalance in multi-class classification.

    """

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_test = y_test.squeeze() if y_test.ndim > 1 else y_test
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix')
    plt.show()

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))

    # Binary classification
    if y_pred_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        ax.plot(fpr, tpr, color='aqua', lw=2, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
        ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    # Multiclass classification
    else: 
        y_onehot_test = pd.get_dummies(y_test).values
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "purple", "pink", "brown", "black"])

        for class_id, color in zip(range(y_onehot_test.shape[1]), colors):
            fpr, tpr, _ = roc_curve(y_onehot_test[:, class_id], y_pred_proba[:, class_id])
            ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve for class {class_id} (area = {auc(fpr, tpr):.2f})')

    ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    ax.set_axisbelow(True)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.show()
        
    cr = classification_report(y_test, y_pred)
    print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1



def train_and_evaluate_model(X_train, X_test, y_train, y_test, models=None):
    """
    Trains and evaluates multiple machine learning models on a given dataset, then visualizes the data embeddings
    using PCA before training. This function trains each model on the training data, evaluates them on the test data, 
    and computes performance metrics (accuracy, precision, recall, and F1-score).

    Args:
        X_train (np.ndarray): Feature matrix for the training data.
        X_test (np.ndarray): Feature matrix for the test data.
        y_train (np.ndarray): True labels for the training data.
        y_test (np.ndarray): True labels for the test data.
        models (list of tuples, optional): A list of tuples, where each tuple contains the model name as a string and
                                           the corresponding scikit-learn model instance. 
                                           If None, default models include Random Forest, Decision Tree, and Logistic Regression.

    Returns:
        list: A list of trained model tuples, where each tuple contains the model name and the trained model instance.

    Side Effects:
        - Displays a PCA 2D visualization of the embeddings using the `visualize_embeddings` function.
        - Trains each model on the training set.
        - Prints evaluation metrics (accuracy, precision, recall, F1-score) for each model on the test set.
        - Displays confusion matrix and ROC curve for each model using the `test_model` function.

    Example:
        models = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    Notes:
        - The `models` argument can be customized to include any classification models from scikit-learn.
        - The function uses PCA for the embedding visualization. You can modify the `visualize_embeddings` function call for other visualization methods or dimensionality reduction techniques.
        - Default models include Random Forest, Decision Tree, and Logistic Regression.
    """
    
    visualize_embeddings(X_train, X_test, y_train, y_test, plot_type='2D', method='PCA')
    
    if not(models):
        # TODO: Implement the ML models
        # The models should be a list of tuples, where each tuple contains the model name and the model instance
        # Example: models = [ ('Model 1', Model1()), ('Model2', Model2()), ... ('ModelN', ModelN()) ]
        models = [
            ('Random Forest', RandomForestClassifier()),
            ('Logistic Regression', LogisticRegression(multi_class='ovr')),
        ]

    for name, model in models:
        
        print('#'*20, f' {name} ', '#'*20)
        # TODO: Train the model on the training
        model.fit(X_train, y_train)
        
        # TODO: Evaluate the model on the test set using the test_model function
        accuracy, precision, recall, f1 = test_model(X_test, y_test, model)
        
    return models
