"""
Setup script for the lightningDS package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lightningDS",
    version="0.1.0",
    author="Bitcoin Data Labs",
    author_email="lnsorukumar@gmail.com",
    description="A data science library for analyzing Lightning Network data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sorukumar/lightningDS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "seaborn>=0.11.0",
        "networkx>=2.5",
        "bokeh>=2.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ],
        "community": [
            "python-louvain>=0.15",
        ],
    },
)
