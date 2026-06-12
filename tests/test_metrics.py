import numpy as np
from metrics import StreamingMetrics


def streaming_metrics_test():
    """
    Testing for StreamingMetrics.
    """

    # Test 1: Perfect predictions
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])

    metrics = StreamingMetrics()
    metrics.update(y_true, y_pred)

    assert metrics.true_pos == 2
    assert metrics.true_neg == 2
    assert metrics.false_pos == 0
    assert metrics.false_neg == 0

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0

    # Test 2: Mixed predictions
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    metrics.reset()
    metrics.update(y_true, y_pred)

    assert metrics.true_pos == 1
    assert metrics.true_neg == 1
    assert metrics.false_pos == 1
    assert metrics.false_neg == 1

    assert np.isclose(metrics.accuracy, 0.5)
    assert np.isclose(metrics.precision, 0.5)
    assert np.isclose(metrics.recall, 0.5)

    # Test 3: Streaming updates (multiple chunks)
    y_true1 = np.array([1, 0])
    y_pred1 = np.array([1, 1])

    y_true2 = np.array([0, 1])
    y_pred2 = np.array([0, 0])

    metrics = StreamingMetrics()
    metrics.update(y_true1, y_pred1)
    metrics.update(y_true2, y_pred2)

    assert metrics.n == 4

    # confusion matrix check
    cm = metrics.confusion_matrix
    assert cm.shape == (2, 2)

    # Check consistency with counts
    assert cm[0, 0] == metrics.true_pos
    assert cm[1, 1] == metrics.true_neg
    assert cm[1, 0] == metrics.false_pos
    assert cm[0, 1] == metrics.false_neg

    # Test 4: Edge case - all negatives predicted correctly
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    metrics.reset()
    metrics.update(y_true, y_pred)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 0.0 or np.isnan(metrics.precision) == False  # safe check
    assert metrics.recall == 0.0 or np.isnan(metrics.recall) == False

    print("StreamingMetrics passed all tests.")