import numpy as np


class DecisionTreeClassifier:

    """
    Decision tree classifier for streaming data.
    """

    def __init__(self, max_depth = None, min_samples_split = 2, max_features = None, criterion = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.tree = None
        self.X = None
        self.y = None

    def fit(self, X, y):

        """
        Fits decision tree to training data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.

        """

        self.tree = self._grow_tree(X, y, 0)

    def partial_fit(self, X_chunk, y_chunk):

        """
        Updates decision tree using chunk of training data.

        Args:
            X_chunk (np.ndarray): Training data chunk.
            y_chunk (np.ndarray): Training labels chunk.
        """

        if self.X is None:
            self.X = X_chunk
            self.y = y_chunk

        else:
            self.X = np.concatenate((self.X, X_chunk))
            self.y = np.concatenate((self.y, y_chunk))

        self.fit(self.X, self.y)

    def predict(self, X):

        """
        Predicts labels from data using fitted decision tree.

        Args:
            X (np.ndarray): Data to predict labels for.

        Returns:
            np.ndarray: Predicted labels.
        """

        return np.array([self._predict_one(x, self.tree) for x in X])

    def _compute_impurity(self, y):

        """
        Computes impurity of a node (either gini impurity or entropy).

        Args:
            y (np.ndarray): Node labels.

        Returns:
            float: Impurity of node.
        """

        probs = y / np.sum(y)

        if self.criterion == 'gini':
            return (1 - np.sum(probs ** 2))
        elif self.criterion == 'entropy':
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

    def _impurity_gain(self, parent_counts, left_counts, right_counts):

        """
        Computes impurity gain of a split.

        Args:
            parent_counts (np.ndarray): Parent node labels.
            left_counts (np.ndarray): Left child node labels.
            right_counts (np.ndarray): Right child node labels.

        Returns:
            float: Impurity gain of split.
        """

        parent_total = np.sum(parent_counts)
        left_total = np.sum(left_counts)
        right_total = np.sum(right_counts)

        if left_total == 0 or right_total == 0:
            return 0

        parent_impurity = self._compute_impurity(parent_counts)
        left_impurity = self._compute_impurity(left_counts)
        right_impurity = self._compute_impurity(right_counts)

        weighted_child = (left_total / parent_total) * left_impurity + (right_total / parent_total) * right_impurity
        impurity_gain = parent_impurity - weighted_child

        return impurity_gain


    def _find_best_split(self, X, y):

        """
        Finds best split for a node.

        Args:
            X (np.ndarray): Node data.
            y (np.ndarray): Node labels.

        Returns:
            tuple: Best feature and threshold for split.
        """

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        parent_impurity = self._compute_impurity(y)
        best_gain = 0
        best_feature = None
        best_threshold = None
        parent_counts = np.bincount(y, minlength=n_classes)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                left_counts = np.bincount(y[left_indices], minlength = n_classes)
                right_counts = np.bincount(y[right_indices], minlength = n_classes)

                if not np.any(left_indices) or not np.any(right_indices):
                    continue

                gain = self._impurity_gain(parent_counts, left_counts, right_counts)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):

        """
        Recursively grows decision tree.

        Args:
            X (np.ndarray): Node data.
            y (np.ndarray): Node labels.
            depth (int): Depth of node in tree.


        """

        values, counts = np.unique(y, return_counts=True)
        prediction = values[np.argmax(counts)]

        if len(values) == 1 or len(y) < self.min_samples_split:
            return prediction

        if self.max_depth is not None and depth >= self.max_depth:
              return prediction

        feature, threshold = self._find_best_split(X, y)

        if feature is None:
            return prediction

        left = X[:, feature] <= threshold

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._grow_tree(X[left], y[left], depth + 1),
            "right": self._grow_tree(X[~left], y[~left], depth + 1),
        }

    def _predict_one(self, x, node):

        """
        Predicts label for a single data point.

        Args:
            x (np.ndarray): Data point.
            node (dict): Node in tree.

        Returns:
            int: Predicted label.
        """

        if not isinstance(node, dict):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])
