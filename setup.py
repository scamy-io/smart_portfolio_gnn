from setuptools import setup, find_packages

setup(
    name="smart_portfolio_gnn",
    version="0.1.0",
    description="Real-time portfolio rebalancing using heterogeneous temporal graph neural networks on live stock relationship graphs.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric",
        "pandas",
        "numpy",
        "yfinance",
        "scikit-learn",
        "streamlit",
        "plotly",
        "kafka-python-ng",
        "watchdog"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "black",
            "isort",
            "flake8"
        ]
    }
)
