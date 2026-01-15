from setuptools import setup, find_packages


setup(
    name="ins_pricing",
    version="0.2.9",
    description="Reusable modelling, pricing, governance, and reporting utilities.",
    author="meishi125478",
    license="Proprietary",
    python_requires=">=3.9",
    packages=find_packages(where="..", include=[
                           "ins_pricing", "ins_pricing.*"]),
    package_dir={"": ".."},
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.4",
    ],
    extras_require={
        "bayesopt": [
            "torch>=1.13",
            "optuna>=3.0",
            "xgboost>=1.6",
            "scikit-learn>=1.1",
            "statsmodels>=0.13",
            "joblib>=1.2",
            "matplotlib>=3.5",
        ],
        "plotting": [
            "matplotlib>=3.5",
            "scikit-learn>=1.1",
        ],
        "explain": [
            "torch>=1.13",
            "shap>=0.41",
            "scikit-learn>=1.1",
        ],
        "geo": [
            "contextily>=1.3",
            "matplotlib>=3.5",
        ],
        "gnn": [
            "torch>=1.13",
            "pynndescent>=0.5",
            "torch-geometric>=2.3",
        ],
        "all": [
            "torch>=1.13",
            "optuna>=3.0",
            "xgboost>=1.6",
            "scikit-learn>=1.1",
            "statsmodels>=0.13",
            "joblib>=1.2",
            "matplotlib>=3.5",
            "shap>=0.41",
            "contextily>=1.3",
            "pynndescent>=0.5",
            "torch-geometric>=2.3",
        ],
    },
    include_package_data=True,
    package_data={"ins_pricing": ["**/*.json", "**/*.md"]},
    exclude_package_data={"ins_pricing": ["examples/*", "examples/**/*"]},
)
