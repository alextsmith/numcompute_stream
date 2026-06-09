import numpy as np


class Pipeline():

    """
    Provides a pipeline for preprocessing, model fitting and prediction.
    """

    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe
    
    def partial_fit(self, X_chunk, y_chunk):

        """
        Applies preprocessing procedure and fits model for a chunk of data.

        Args:
            X_chunk (np.ndarray): Data chunk.
            y_chunk (np.ndarray): Labels chunk.
        """

        # Exception handling
        if not isinstance(X_chunk, np.ndarray):
            raise TypeError("X_chunk must be a NumPy array")
        if not isinstance(y_chunk, np.ndarray):
            raise TypeError("y_chunk must be a NumPy array")
        if np.ndim(X_chunk) != 2:
            raise ValueError("X_chunk must be a 2D array")
        if np.ndim(y_chunk) != 1:
            raise ValueError("y_chunk must be 1D array")

        # Perform steps in pipeline    

        for name, operation in self.pipe:

            if name == "model":
                operation.partial_fit(X_chunk, y_chunk)
            else:
                X_chunk = operation.fit_transform(X_chunk)

    def predict(self, X):

        """
        Predicts labels from data using fitted pipeline.

        Args:
            X (np.ndarray): Data to predict labels for.

        Returns:
            np.ndarray: Predicted labels.
        """

        # Exception handling
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array")
        if np.ndim(X) != 2:
            raise ValueError("X must be a 2D array")

        # Predict using fitted pipeline
        for name, operation in self.pipe:
            if name == "model":
                return operation.predict(X)
            else:
                X = operation.transform(X)