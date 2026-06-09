import numpy as np


class StreamingStats:
    """
    Calculates statistics for streaming data for each feature in a 2D array.
    """

    def __init__(self):

        self.n_features = None
        self._n = None
        self._mean = None
        self._M2 = None

    def update_stats(self, X_chunk):

        """
        Updates streaming statistics using a new batch of data.

        Args:
        X_chunk (np.ndarray): A 2D array with of shape (n_samples, n_features) 
        containing data chunk. If previous data chunks have been used to
        update streaming stats, n_features must match the number of features
        of the previous chunks.
        """

        # Exception handling
        if not isinstance(X_chunk, np.ndarray):
            raise TypeError("X_chunk must be a numpy array")
        if np.ndim(X_chunk) != 2:
            raise ValueError("X_chunk must be a 2D array")
        if self.n_features is None:
            self.n_features = X_chunk.shape[1]
            self._n = np.zeros(self.n_features)
            self._mean = np.zeros(self.n_features)
            self._M2 = np.zeros(self.n_features)
        elif self.n_features != X_chunk.shape[1]:
            raise ValueError("X_chunk must have the same number of features as the previous chunks")

        # Determine chunk size
        chunk_size = np.sum(~np.isnan(X_chunk), axis = 0)

        # Calculate batch statistics
        mean_chunk = np.nanmean(X_chunk, axis = 0)
        M2_chunk = np.nansum(((X_chunk - mean_chunk) ** 2), axis = 0)

        # Update stats
        delta = mean_chunk - self._mean
        self._mean += delta * chunk_size / (self._n + chunk_size)
        self._M2 += M2_chunk + delta ** 2 * self._n * chunk_size / (self._n + chunk_size)
        self._n += chunk_size


    @property
    def n(self):
        return self._n


    @property
    def mean(self):
        return self._mean.copy()


    @property
    def var(self):

        """Population variance"""

        self._M2[self._n < 2] = np.nan

        return self._M2 / self._n


    @property
    def std(self):

        """Population standard deviation"""

        return np.sqrt(self.var)