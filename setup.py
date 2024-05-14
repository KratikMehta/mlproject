from typing import List

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List[str]:
    """This function returns the list of requiremnts"""
    with open(file_path) as f:
        requirements = [req.strip() for req in f.readlines() if req != "-e ."]
    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Kratik Mehta",
    author_email="kratikmehta57@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
