import numpy as np
from tree import DecisionTreeClassifier


class EnsembleClassifier(DecisionTreeClassifier):
    """
    Ensemble method using bagging (i.e. bootstrap aggregating).
    """

    def __init__(self, n_estimators = 100):
        super().__init__()
        self.n_estimators = n_estimators
        self.trees = np.full(n_estimators, None, dtype=object)
        self.X = None
        self.y = None

    def fit(self, X, y):

        """
        Fits ensemble of decision trees to training data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
        """

        # Exception handling
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array")
        if np.ndim(X) != 2:
            raise ValueError("X must be a 2D array")
        if np.ndim(y) != 1:
            raise ValueError("y must be 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            # Generate a new bootstrap sample for each tree
            indices = np.random.choice(n_samples, size = n_samples, replace = True)
            X_bootstrap = X[indices, :]
            y_bootstrap = y[indices]

            tree = DecisionTreeClassifier()
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees[i] = tree

        self.X = X
        self.y = y

    def partial_fit(self, X_chunk, y_chunk):

        """
        Updates ensemble of decision trees using chunk of training data.

        Args:
            X_chunk (np.ndarray): Training data chunk.
            y_chunk (np.ndarray): Training labels chunk.
        """
        
        if self.X is None:
            self.X = X_chunk
            self.y = y_chunk

        else:
            self.X = np.concatenate((self.X, X_chunk))
            self.y = np.concatenate((self.y, y_chunk))

        self.fit(self.X, self.y)

    def predict(self, X):

        """
        Predicts labels from data using fitted ensemble of decision trees.

        Args:
            X (np.ndarray): Data to predict labels for.

        Returns:
            np.ndarray: Predicted labels.
        """

        predictions = np.array([tree.predict(X) for tree in self.trees])
        print(predictions)
        return np.array([np.bincount(predictions[:, i]).argmax() for i in range(X.shape[0])])
