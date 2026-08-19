from setuptools import setup, find_packages

setup(
    name="fedpareto",
    version="0.1.0",
    description="FedPARETO research implementation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
)
