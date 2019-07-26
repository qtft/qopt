from setuptools import setup

NAME = "qtft-qopt"
VERSION = "0.0.1.dev1"
DESCRIPTION = "Quantum-inspired optimization algorithms"
AUTHOR = "Quantum Technology Foundation of Thailand"
EMAIL = "contact@qtft.org"
HOMEPAGE_URL = "https://qtft.org/"
LICENSE = "LGPL-2.1"

# TODO: Add more information

# Current command to build packages and push to PyPI:
# $ python setup.py sdist bdist_wheel
# $ twine dist/*

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=HOMEPAGE_URL,
    license=LICENSE,
    classifiers=[
        'Development Status :: 1 - Planning',
    ],
    packages=[],
)
