"""Keep pytest focused on automated tests rather than legacy manual scripts."""

collect_ignore = [
    "test_benchmark.py",
    "test_decomposition.py",
    "test_lin_log.py",
    "test_tiling_performance_3d.py",
]
