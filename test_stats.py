import numpy as np
from stats import StreamingStats


def streaming_stats_test():
    """
    Testing for StreamingStats.
    """

    # Test 1: Single chunk
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    stats = StreamingStats()
    stats.update_stats(X)

    assert np.allclose(stats.n, np.array([3, 3]))
    assert np.allclose(stats.mean, np.array([3.0, 4.0]))
    assert np.allclose(stats.var, np.var(X, axis=0))
    assert np.allclose(stats.std, np.std(X, axis=0))

    # Test 2: Multiple chunks
    X1 = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    X2 = np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    stats = StreamingStats()
    stats.update_stats(X1)
    stats.update_stats(X2)

    X_full = np.vstack([X1, X2])

    assert np.allclose(stats.n, np.array([4, 4]))
    assert np.allclose(stats.mean, np.mean(X_full, axis=0))
    assert np.allclose(stats.var, np.var(X_full, axis=0))
    assert np.allclose(stats.std, np.std(X_full, axis=0))

    # Test 3: Missing values
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    stats = StreamingStats()
    stats.update_stats(X)

    assert np.allclose(stats.n, np.array([3, 2]))
    assert np.allclose(stats.mean, np.nanmean(X, axis=0))
    assert np.allclose(stats.var, np.nanvar(X, axis=0))
    assert np.allclose(stats.std, np.nanstd(X, axis=0))

    # Test 4: Constant values
    X = np.array([
        [7.0, 7.0],
        [7.0, 7.0],
        [7.0, 7.0]
    ])

    stats = StreamingStats()
    stats.update_stats(X)

    assert np.allclose(stats.mean, np.array([7.0, 7.0]))
    assert np.allclose(stats.var, np.array([0.0, 0.0]))
    assert np.allclose(stats.std, np.array([0.0, 0.0]))

    print("StreamingStats passed all tests.")