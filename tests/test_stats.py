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

    # Test 5: Many chunks produce the same result as one large chunk
    rng = np.random.default_rng(42)
    X_full = rng.standard_normal((100, 5))

    stats_single = StreamingStats()
    stats_single.update_stats(X_full)

    stats_chunked = StreamingStats()

    for i in range(0, 100, 10):
        stats_chunked.update_stats(X_full[i:i+10])

    assert np.allclose(stats_single.n, stats_chunked.n)
    assert np.allclose(stats_single.mean, stats_chunked.mean)
    assert np.allclose(stats_single.var, stats_chunked.var)
    assert np.allclose(stats_single.std, stats_chunked.std)

    # Test 6: Exception handling
    stats = StreamingStats()

    try:
        stats.update_stats([[1.0, 2.0], [3.0, 4.0]]) 
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        stats.update_stats(np.array([1.0, 2.0, 3.0]))  # 1D array
        assert False, "Expected ValueError"
    except ValueError:
        pass

    stats.update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
    try:
        stats.update_stats(np.array([[1.0, 2.0, 3.0]]))  # Mismatched features
        assert False, "Expected ValueError"
    except ValueError:
        pass

    print("StreamingStats passed all tests.")


streaming_stats_test()
