"""Mudra quantitative trading and machine learning framework package."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Pre-load onnxruntime before numpy/OpenMP engines to avoid Windows 0xC0000005 crashes.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass
