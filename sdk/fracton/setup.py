#!/usr/bin/env python3
"""
Setup script for Fracton - Infodynamics Computational Modeling Language
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements-dev.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

setup(
    name="fracton",
    version="0.1.0",
    description="Infodynamics Computational Modeling Language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dawn Field Institute",
    author_email="research@dawnfield.institute",
    url="https://github.com/dawnfield-institute/dawn-field-theory",
    
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    
    extras_require={
        "dev": requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    
    entry_points={
        "console_scripts": [
            "fracton-test=tests.run_tests:main",
        ],
    },
    
    keywords="infodynamics consciousness recursion entropy modeling",
    project_urls={
        "Documentation": "https://github.com/dawnfield-institute/dawn-field-theory/tree/main/sdk/fracton",
        "Source": "https://github.com/dawnfield-institute/dawn-field-theory",
        "Tracker": "https://github.com/dawnfield-institute/dawn-field-theory/issues",
    },
)
