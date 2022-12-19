from typing import Dict

from setuptools import find_packages, setup

version: Dict[str, str] = {}
with open("requirements.txt", "r") as f:
    requirements = [x for x in f.read().splitlines() if "#" not in x]

with open("redkg/version.py") as f:
    exec(f.read(), version)

setup(
    name="redkg",
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    version=version["version"],
    description="ReDKG library",
    author="NCCR Team, ITMO Univerisy",
    install_requires=requirements,
)
