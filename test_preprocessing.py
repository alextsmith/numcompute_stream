import numpy as np
from stats import StreamingStats
from preprocessing import StandardScaler, Imputer, OneHotEncoder


def standard_scaler_test():
    """
    Testing for StandardScaler.
    """

    # Test 1: Single chunk 
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert scaler.n is not None
    assert np.allclose(scaler.mean, np.mean(X, axis=0))
    assert np.allclose(scaler.std, np.std(X, axis=0))

    # Mean of scaled data should be approx 0
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0)

    # Std of scaled data should be approx 1
    assert np.allclose(np.std(X_scaled, axis=0), 1.0)

    # Test 2: 
    X1 = np.array([[1.0, 2.0],
                   [3.0, 4.0]])

    X2 = np.array([[5.0, 6.0],
                   [7.0, 8.0]])

    scaler = StandardScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    X_full = np.vstack([X1, X2])

    X_scaled = scaler.transform(X_full)

    assert np.allclose(scaler.mean, np.mean(X_full, axis=0))
    assert np.allclose(scaler.std, np.std(X_full, axis=0))

    assert np.allclose(np.mean(X_scaled, axis=0), 0.0)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0)

    # Test 3: Std = 0 edge case
    X = np.array([
        [7.0, 7.0],
        [7.0, 7.0],
        [7.0, 7.0]
    ])

    scaler = StandardScaler()
    scaler.partial_fit(X)

    X_scaled = scaler.transform(X)

    assert np.allclose(scaler.mean, np.array([7.0, 7.0]))

    assert np.all(np.isnan(X_scaled))

    print("StandardScaler passed all tests.")


def imputer_test():
    """
    Testing for Imputer.
    """

    # Test 1: Basic mean imputation
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    imputer = Imputer()
    X_out = imputer.fit_transform(X)

    expected_mean = np.nanmean(X, axis=0)

    assert np.allclose(imputer.mean, expected_mean)

    # Check there are no NaNs
    assert np.isnan(X_out).sum() == 0
    
    assert X_out[0, 1] == expected_mean[1]

    # Test 2: Multiple chunks
    X1 = np.array([[1.0, np.nan],
                   [3.0, 4.0]])

    X2 = np.array([[5.0, 6.0],
                   [7.0, np.nan]])

    imputer = Imputer()
    imputer.partial_fit(X1)
    imputer.partial_fit(X2)

    X_full = np.vstack([X1, X2])
    X_out = imputer.transform(X_full)

    assert np.allclose(imputer.mean, np.nanmean(X_full, axis=0))
    assert np.isnan(X_out).sum() == 0

    # Test 3: No missing values
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    imputer = Imputer()
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, X)
    assert np.allclose(imputer.mean, np.mean(X, axis=0))

    print("Imputer passed all tests.")


def one_hot_encoder_test():
    """
     Testing for OneHotEncoder.
    """

    # Test 1: Basic encoding
    X = np.array([
        ["cat"],
        ["dog"],
        ["cat"]
    ])

    encoder = OneHotEncoder()
    X_encoder = encoder.fit_transform(X)

    assert X_encoder.shape == X.shape
    assert len(encoder.encoding) == 2
    assert np.isnan(X_encoder).sum() == 0

    # Test 2: Multiple chunks
    X1 = np.array([["a"], ["b"]])
    X2 = np.array([["b"], ["c"]])

    encoder = OneHotEncoder()
    encoder.partial_fit(X1)
    encoder.partial_fit(X2)

    assert len(encoder.encoding) == 3

    X_full = np.vstack([X1, X2])
    X_encoder = encoder.transform(X_full)

    assert np.isnan(X_encoder).sum() == 0

    # Test 3: Unknown handling behavior (current: NaN)
    X_train = np.array([["x"], ["y"]])
    X_test = np.array([["z"]])

    encoder = OneHotEncoder()
    encoder.partial_fit(X_train)
    X_out = encoder.transform(X_test)

    assert np.isnan(X_out).all()

    print("OneHotEncoder passed all tests.")