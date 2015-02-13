from setuptools import find_packages, setup

setup(
    name="esl",
    version="0.1.0",
    description="Outils pour les notebooks The Elements of Statistical Learning",
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "scikit-learn"],
)
