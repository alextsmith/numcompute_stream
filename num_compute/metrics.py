import numpy as np


class StreamingMetrics():

    """
    Calculates classification metrics for streaming data.
    """

    def __init__(self):

        super().__init__()

        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.n = 0

    def update(self, y_true_chunk, y_pred_chunk):

        """
        Updates metrics using a new batch of data.

        Args:
            y_true_chunk (np.ndarray): A array containing a chunk of true labels.
            y_pred_chunk (np.ndarray): A array containing a chunk of predicted labels.
        """

        # Exception handling
        if not isinstance(y_true_chunk, np.ndarray):
            raise TypeError("y_true_chunk must be a numpy array")
        if not isinstance(y_pred_chunk, np.ndarray):
            raise TypeError("y_pred_chunk must be a numpy array")
        if np.ndim(y_true_chunk) != 1:
            raise ValueError("y_true_chunk must be a 1D array")
        if np.ndim(y_pred_chunk) != 1:
            raise ValueError("y_pred_chunk must be a 1D array")
        if y_true_chunk.shape != y_pred_chunk.shape:
            raise ValueError("y_true_chunk and y_pred_chunk must have the same shape")

        # Update
        self.true_pos += np.sum((y_true_chunk == 1) & (y_pred_chunk == 1))
        self.true_neg += np.sum((y_true_chunk == 0) & (y_pred_chunk == 0))
        self.false_pos += np.sum((y_true_chunk == 0) & (y_pred_chunk == 1))
        self.false_neg += np.sum((y_true_chunk == 1) & (y_pred_chunk == 0))
        self.n += len(y_true_chunk)


    def reset(self):

        """
        Resets metrics.
        """

        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.n = 0

    @property
    def accuracy(self):

        if self.n == 0:
            return np.nan

        return (self.true_pos + self.true_neg) / self.n


    @property
    def precision(self):

        if self.true_pos + self.false_pos == 0:
            return np.nan

        return (self.true_pos) / (self.true_pos + self.false_pos)


    @property
    def recall(self):

        if self.true_pos + self.false_neg == 0:
            return np.nan

        return (self.true_pos) / (self.true_pos + self.false_neg)


    @property
    def confusion_matrix(self):
        return np.array([[self.true_pos, self.false_neg], [self.false_pos, self.true_neg]])