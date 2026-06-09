import numpy as np
from metrics import StreamingMetrics
from stream import StreamTrainer
from pipeline import Pipeline
from preprocessing import StandardScaler
from tree import DecisionTreeClassifier


def stream_trainer_test():
    """
    Testing for StreamTrainer.
    """

    # Test 1: Single chunk fit and score
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier())
    ])

    trainer = StreamTrainer(pipeline)

    trainer.fit_chunk(X, y)
    trainer.score_chunk(X, y)

    assert trainer.n == len(y)
    assert trainer.logs is not None
    assert trainer.logs.shape == (1, 3)

    assert 0 <= trainer.accuracy <= 1
    assert 0 <= trainer.precision <= 1
    assert 0 <= trainer.recall <= 1

    # Test 2: Multiple chunks accumulate metrics
    X1 = np.array([
        [1.0],
        [2.0]
    ])

    y1 = np.array([0, 0])

    X2 = np.array([
        [3.0],
        [4.0]
    ])

    y2 = np.array([1, 1])

    trainer = StreamTrainer(
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier())
        ])
    )

    trainer.fit_chunk(X1, y1)
    trainer.score_chunk(X1, y1)

    trainer.fit_chunk(X2, y2)
    trainer.score_chunk(X2, y2)

    assert trainer.logs.shape == (2, 3)
    assert trainer.n == 4

    # Test 3: Logs contain accuracy, precision, recall
    last_log = trainer.logs[-1]

    assert np.isclose(last_log[0], trainer.accuracy)
    assert np.isclose(last_log[1], trainer.precision)
    assert np.isclose(last_log[2], trainer.recall)

    print("StreamTrainer passed all tests.")