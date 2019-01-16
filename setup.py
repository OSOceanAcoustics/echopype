import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in echopype/version.py
ver_file = os.path.join('echopype', 'version.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name = "echopype",
            maintainer = "Wu-Jung Lee",
            maintainer_email = "leewujung@gmail.com",
            description = DESCRIPTION,
            long_description = LONG_DESCRIPTION,
            url = "https://github.com/OSOceanAcoustics/echopype",
            download_url = "",
            license = "Apache 2.0",
            classifiers = CLASSIFIERS,
            author = "Wu-Jung Lee",
            author_email = "leewujung@gmail.com",
            platforms = "OS Independent",
            version = VERSION,
            packages = PACKAGES,
            package_data = PACKAGE_DATA,
            install_requires = REQUIRES,
            requires = REQUIRES,
            test_requires = ['tox'])


if __name__ == '__main__':
    setup(**opts)
