"""USAF: Ultra Sparse Adaptive Fine-Tuning - Python package setup.

All package metadata lives in pyproject.toml (PEP 621). This shim exists
only so legacy "python setup.py" invocations keep working; it declares no
metadata of its own, which removes the previous source of drift between the
two files (they disagreed on the license - MIT here vs. Apache-2.0
everywhere else - and on the transformers floor).
"""
from setuptools import setup

setup()
