from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import KDTree

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

    def _euclidean_distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        a (array-like): First point.
        b (array-like): Second point.

        Returns:
        distance (float): Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((a - b) ** 2))

class KNNRegressionModel(MachineLearningModel):
    """
    Class for KNN regression model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
       
    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by averaging the target variable of the k nearest neighbors.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
            k_neighbors_indices = np.argsort(distances)[:self.k]
            k_neighbors_values = self.y_train[k_neighbors_indices]
            predictions.append(np.mean(k_neighbors_values))
        return np.array(predictions)
       

    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the Mean Squared Error (MSE) between the true and predicted values.
        The MSE is calculated as the average of the squared differences between the true and predicted values.        

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        return np.mean((np.array(y_true) - np.array(y_predicted)) ** 2)

class KNNClassificationModel(MachineLearningModel):
    """
    Class for KNN classification model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by taking the mode (majority) of the target variable of the k nearest neighbors.
        
        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
            k_neighbors_indices = np.argsort(distances)[:self.k]
            k_neighbors_values = self.y_train[k_neighbors_indices]
            unique_labels, counts = np.unique(k_neighbors_values, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            predictions.append(majority_label)
        return np.array(predictions)
        

    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the total number of correct predictions only.
        Do not use any other evaluation metric.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        return np.sum(np.array(y_true) == np.array(y_predicted)) / len(y_true)


class FastKNNClassificationModel(MachineLearningModel):
    """
    Class for Fast KNN classification model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.tree = None
        self.y_train = None

    
    def fit(self, X, y):
        """
        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.tree = KDTree(X)
        self.y_train = np.array(y)

    
    def predict(self, X):
        """
        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X = np.array(X)
        predictions = []

        distances, indexes = self.tree.query(X, k=self.k)
        for index in indexes:
            k_neighbors_values = self.y_train[index]
            unique_labels, counts = np.unique(k_neighbors_values, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            predictions.append(majority_label)
        return np.array(predictions)
        
    

    def evaluate(self, y_true, y_predicted):
        return np.sum(np.array(y_true) == np.array(y_predicted)) / len(y_true)


        