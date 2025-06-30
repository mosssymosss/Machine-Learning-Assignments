import numpy as np
from ROCAnalysis import ROCAnalysis

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.model = model
        self.selected_features = []
        self.best_cost = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_split(X, y)
    

    def create_split(self, X, y):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        X = np.array(X)
        y = np.array(y)

        np.random.seed(6969) 
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        train_size = int(0.8 * len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        return X_train, X_test, y_train, y_test
    

    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """

        if not features:
            return 0.0
        
        X_train_selected = self.X_train[:, features]
        X_test_selected = self.X_test[:, features]

        self.model.fit(X_train_selected, self.y_train)

        y_pred = self.model.predict(X_test_selected)
        roc = ROCAnalysis(y_pred, self.y_test)

        return roc.f_score()
        

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        
        num_features = self.X.shape[1]
        remaining_features = list(range(num_features))
        self.best_cost = 0.0
        self.selected_features = []

        f_score = self.train_model_with_features([])
        best_model_scores = [f_score]
        best_model_features = [[]]

        for i in range(num_features):
            best_feature = None
            best_f_score = 0.0

            for feature in remaining_features:
                current_features = self.selected_features + [feature]
                current_f_score = self.train_model_with_features(current_features)

                if current_f_score > best_f_score:
                    best_f_score = current_f_score
                    best_feature = feature

            if best_feature is not None:
                self.selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_model_scores.append(best_f_score)
                best_model_features.append(list(self.selected_features))

                if best_f_score > self.best_cost:
                    self.best_cost = best_f_score

            else:
                break

        
        best_idx = np.argmax(best_model_scores)
        self.selected_features = best_model_features[best_idx]

                
    def fit(self):
        """
        Fits the model using the selected features.
        """
        self.forward_selection()

        if not self.selected_features:
            return

        self.model.fit(self.X[:, self.selected_features], self.y)
       
        

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        X_test = np.array(X_test)
        
        if not self.selected_features:
            return np.zeros(X_test.shape[0])

        X_test_selected = X_test[:, self.selected_features]
        return self.model.predict(X_test_selected)
