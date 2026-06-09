import numpy as np
from metrics import StreamingMetrics


class StreamTrainer(StreamingMetrics):

    """
    Applies pipeline to stream of data and updates classification metrics.
    """

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.logs = None

    def fit_chunk(self, X_chunk, y_chunk):

        """
        Updates pipeline using chunk of data.

        Args:
            X_chunk (np.ndarray): Data chunk.
            y_chunk (np.ndarray): Labels chunk.
        """

        self.pipeline.partial_fit(X_chunk, y_chunk)

    def score_chunk(self, X_chunk, y_chunk):

        """
        Updates metrics for a chunk of data.

        Args:
            X_chunk (np.ndarray): Data chunk.
            y_chunk (np.ndarray): Labels chunk.
        """

        y_pred_chunk = self.pipeline.predict(X_chunk)

        for name, operation in self.pipeline.pipe:

            if name == "encoder":
                y_chunk = operation.fit_transform(y_chunk)


        self.update(y_chunk, y_pred_chunk)

        if self.logs is None:
            self.logs = np.array([[self.accuracy, self.precision, self.recall]])
        else:
            self.logs = np.vstack((self.logs, np.array([[self.accuracy, self.precision, self.recall]])))

    def reset_logs(self):

        """
        Resets logs.
        """
        self.logs = None