# tests/conftest.py
"""
Pytest configuration and session fixtures for Mudra.
Ensures clean C-runtime and OpenMP initialization across OS environments.
"""

# Pre-load onnxruntime before numerical engines (XGBoost/OpenMP) to prevent Windows DLL access violations
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass
