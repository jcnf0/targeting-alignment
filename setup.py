from setuptools import find_packages, setup

# Core dependencies required for basic functionality
CORE_DEPS = [
    "accelerate==1.7.0",
    "fschat==0.2.36",
    "numpy==1.26.3",
    "pandas==2.2.3",
    "protobuf==6.31.0",
    "pynvml==12.0.0",  # To measure GPU VRAM usage
    "scikit-learn==1.6.1",
    "sentencepiece==0.2.0",
    "torch==2.7.0",
    "torchvision==0.22.0",  # Fixes torchvision version for imports
    "transformers==4.52.3",
    "typing==3.7.4.3",
]

# Optional dependencies for visualization and interactive usage
VIZ_DEPS = ["matplotlib", "seaborn"]

# Hardware-specific Polars dependencies
POLARS_DEPS = {
    "polars": ["polars"],  # Default Polars
    "polars-lts": ["polars-lts-cpu"],  # LTS CPU version
}

setup(
    name="clfextract",
    version="0.1.0",
    description="Source code for Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs",
    author="Jean-Charles Noirot Ferrand",
    author_email="jcnf@cs.wisc.edu",
    python_requires=">=3.10",
    url="https://github.com/jcnf0/targeting-alignment",
    packages=find_packages(),
    install_requires=CORE_DEPS,
    extras_require={
        # Individual feature sets
        "viz": VIZ_DEPS,
        "polars": POLARS_DEPS["polars"],
        "polars-lts": POLARS_DEPS["polars-lts"],
        # Convenience combinations
        "full": VIZ_DEPS + POLARS_DEPS["polars"],
        "full-lts": VIZ_DEPS + POLARS_DEPS["polars-lts"],
        # Development dependencies
        "dev": [
            "pytest",
            "debugpy",
            "black",
            "isort",
            "flake8",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
