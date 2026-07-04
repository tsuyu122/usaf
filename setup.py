"""USAF: Ultra Sparse Adaptive Fine-Tuning — Python package setup."""
from setuptools import setup, find_packages

setup(
    name="usaf",
    version="0.2.0",
    description="Ultra Sparse Adaptive Fine-Tuning for MoE models",
    author="USAF Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "safetensors",
        "numpy",
        "psutil",
    ],
    entry_points={
        "console_scripts": [
            "usaf=usaf.train:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
