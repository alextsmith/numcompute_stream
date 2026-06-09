import numpy as np
from tree import DecisionTreeClassifier


def decision_tree_classifier_test():
    """
    Testing for DecisionTreeClassifier.
    """

    # Test 1: Perfectly separable data
    X = np.array([
        [0],
        [1],
        [2],
        [3]
    ])

    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y_pred = tree.predict(X)

    assert np.array_equal(y_pred, y)

    # Test 2: Single-class data
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([1, 1, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y_pred = tree.predict(X)

    assert np.array_equal(y_pred, y)

    # Test 3: max_depth = 0
    X = np.array([
        [0],
        [1],
        [2],
        [3]
    ])

    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit(X, y)

    y_pred = tree.predict(X)

    # Majority class prediction
    assert np.all(y_pred == 0)

    # Test 4: partial_fit
    X1 = np.array([[0], [1]])
    y1 = np.array([0, 0])

    X2 = np.array([[2], [3]])
    y2 = np.array([1, 1])

    tree = DecisionTreeClassifier()

    tree.partial_fit(X1, y1)
    tree.partial_fit(X2, y2)

    X_full = np.vstack((X1, X2))
    y_full = np.concatenate((y1, y2))

    y_pred = tree.predict(X_full)

    assert np.array_equal(y_pred, y_full)

    # Test 5: prediction shape
    X = np.array([
        [0],
        [1],
        [2]
    ])

    y = np.array([0, 0, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y_pred = tree.predict(X)

    assert y_pred.shape == y.shape

    print("DecisionTreeClassifier passed all tests.")