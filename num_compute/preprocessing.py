import numpy as np
from stats import StreamingStats


class StandardScaler(StreamingStats):
    """
    Applies standard scaling to streaming data.
    """

    def __init__(self):

        super().__init__()

    def partial_fit(self, X_chunk):

        """
        Incrementally updates streaming statistics using a new batch of data.

        Args:
        X_chunk (np.ndarray): A 2D array with of shape (n_samples, n_features) 
        containing data chunk. If previous data chunks have been used to
        update streaming stats, n_features must match the number of features
        of the previous chunks.
        """

        self.update_stats(X_chunk)

    def transform(self, X):

        """
        Transforms data using StandardScaler for mean and standard deviation
        computed from streaming data.

        Args:
        X (np.ndarray): A 2D array with of shape (n_samples, n_features) 
        containing data to be scaled. 
        """

        return (X - self.mean) / self.std

    def fit_transform(self, X):

        """
        Computes mean and standard deviation on streaming data and transforms
        data using StandardScaler.

        Args:
        X (np.ndarray): A 2D array with of shape (n_samples, n_features) 
        containing data to be scaled.
        """

        self.partial_fit(X)
        return self.transform(X)
    

class Imputer(StreamingStats):

    """
    Imputes missing values with mean of that feature.
    """

    def __init__(self):

        super().__init__()

    def partial_fit(self, X_chunk):

        """
        Updates feature means using a new batch of data.

        Args:
        X_chunk (np.ndarray): A 2D array with of shape (n_samples, n_features) 
        containing data chunk. If previous data chunks have been used to
        update streaming stats, n_features must match the number of features
        of the previous chunks.
        """

        self.update_stats(X_chunk)

    def transform(self, X):

        """
        Transforms data using Imputer for mean computed from streaming data.

        Args:
        X (np.ndarray): A 2D array with the same number of features as the input 
        chunks.
        """

        return np.where(np.isnan(X), self.mean, X)


    def fit_transform(self, X):

        """
        Calculates mean on streaming data and transforms data using Imputer.

        Args:
        X (np.ndarray): A 2D array containing data to be imputed.
        """

        self.partial_fit(X)
        return self.transform(X)
    

class OneHotEncoder():

    """
    One hot encoding for streaming data. Maps categorical variables to integer
    encoding. 
    """

    def __init__(self):
        self.encoding = dict()
        self.num_unique = 0
        

    def partial_fit(self, X_chunk):

        """
        Updates encoding using a new batch of data.

        Args:
            X_chunk (np.ndarray): A array containing a chunk of data.
        """

        # Exception handling
        if not isinstance(X_chunk, np.ndarray):
            raise TypeError("X_chunk must be a numpy array")
    
        nan_mask = np.isin(X_chunk, [np.nan, "nan"])
        unique_values = np.unique(X_chunk[~nan_mask])
        new_values = unique_values[~np.isin(unique_values, list(self.encoding.keys()))]
        self.encoding.update(zip(new_values, np.arange(self.num_unique, self.num_unique + len(new_values))))
        self.num_unique += len(new_values)

    def transform(self, X):

        """
        Encoders categorical variables to integer encoding.

        Args:
            X (np.ndarray): A array containing data to be encoded.
        """

        # Exception handling
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        # Need to add handle_unknown = "ignore" functionality
        # Need more robust nan functionality for arrays with strings

        encoding_func = np.vectorize(self.encoding.get)
        nan_mask = np.isin(X, [np.nan, "nan"])
        encoded_arr = np.full(X.shape, np.nan)
        encoded_arr[~nan_mask] = encoding_func(X[~nan_mask]).astype(float)
        encoded_arr[nan_mask] = np.nan

        return encoded_arr.astype(int)

    def fit_transform(self, X):

        """
        
        """

        self.partial_fit(X)
        return self.transform(X)