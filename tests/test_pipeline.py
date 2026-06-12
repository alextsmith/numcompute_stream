import numpy as np
from pipeline import Pipeline
from preprocessing import StandardScaler, Imputer
from tree import DecisionTreeClassifier


def pipeline_test():
    """
    Testing for Pipeline.
    """

    # Test 1: Fit and predict
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier())
    ])

    pipe.partial_fit(X, y)

    y_pred = pipe.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape

    # Should fit perfectly on training data
    assert np.array_equal(y_pred, y)

    # Test 2: Multiple chunks
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

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier())
    ])

    pipe.partial_fit(X1, y1)
    pipe.partial_fit(X2, y2)

    X_full = np.vstack((X1, X2))
    y_full = np.concatenate((y1, y2))

    y_pred = pipe.predict(X_full)

    assert y_pred.shape == y_full.shape

    # Test 3: Imputer + model
    X = np.array([
        [1.0],
        [np.nan],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imputer", Imputer()),
        ("model", DecisionTreeClassifier())
    ])

    pipe.partial_fit(X, y)

    y_pred = pipe.predict(X)

    assert y_pred.shape == y.shape

    print("Pipeline passed all tests.")