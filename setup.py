from __future__ import absolute_import, division, print_function
from setuptools import setup, find_packages

# Long description will go up on the pypi page
with open("README.md") as file:
    LONG_DESCRIPTION = file.read()

# Dependencies.
with open("requirements.txt") as f:
    requirements = f.readlines()
INSTALL_REQUIRES = [t.strip() for t in requirements]

opts = dict(
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    package_dir={"": "."},
    install_requires=INSTALL_REQUIRES,
    tests_require=["tox", "pandas"],
    py_modules=["_echopype_version"],
    use_scm_version={
        "fallback_version": "unknown",
        "local_scheme": "node-and-date",
        "write_to": "_echopype_version.py",
        "write_to_template": 'version = "{version}"\n',
    },
    setup_requires=["setuptools>=30.3.0", "wheel", "setuptools_scm"],
)


if __name__ == "__main__":
    setup(**opts)
