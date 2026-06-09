import numpy as np
from tree import DecisionTreeClassifier
from ensemble import EnsembleClassifier


def ensemble_classifier_test():
    """
    Testing for EnsembleClassifier.
    """

    np.random.seed(42)

    # Test 1: Fit creates correct number of trees
    X = np.array([
        [0],
        [1],
        [2],
        [3]
    ])

    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=5)
    model.fit(X, y)

    assert len(model.trees) == 5
    assert all(tree is not None for tree in model.trees)

    # Test 2: Prediction shape
    y_pred = model.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == model.y.shape

    # Test 3: Perfectly separable data
    X = np.array([
        [0],
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = EnsembleClassifier(n_estimators=25)
    model.fit(X, y)

    y_pred = model.predict(X)

    accuracy = np.mean(y_pred == y)

    assert accuracy >= 0.8

    # Test 4: partial_fit
    X1 = np.array([
        [0],
        [1]
    ])

    y1 = np.array([0, 0])

    X2 = np.array([
        [2],
        [3]
    ])

    y2 = np.array([1, 1])

    model = EnsembleClassifier(n_estimators=10)

    model.partial_fit(X1, y1)

    assert model.X.shape == X1.shape
    assert model.y.shape == y1.shape

    model.partial_fit(X2, y2)

    assert model.X.shape[0] == 4
    assert model.y.shape[0] == 4

    y_pred = model.predict(model.X)

    assert y_pred.shape == model.y.shape

    # Test 5: Single class dataset
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([1, 1, 1])

    model = EnsembleClassifier(n_estimators=10)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert np.all(y_pred == 1)

    print("EnsembleClassifier passed all tests.")