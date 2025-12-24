from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="predep-analysis",
    version="0.1.0",
    author="Danilo Couto de Souza, Mateus Cruz",
    author_email="",
    description="PREDEP analysis for MoV-precipitation relationships",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PREDEP-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "bokeh>=3.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "predep-run=scripts.run_analysis:main",
            "predep-test=scripts.test_predep:main",
        ],
    },
    include_package_data=True,
)