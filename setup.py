from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

# Long description will go up on the pypi page
with open("README.md") as file:
    LONG_DESCRIPTION = file.read()

# Dependencies.
with open("requirements.txt") as f:
    requirements = f.readlines()

with open("requirements-dev.txt") as f:
    dev_reqs = f.readlines()

EXTRA_REQUIRES = {
    "dev": dev_reqs,
}

INSTALL_REQUIRES = [t.strip() for t in requirements]

opts = dict(
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires=">=3.6",
    py_modules=["_echopype_version"],
    use_scm_version={
        "fallback_version": "unknown",
        "local_scheme": "node-and-date",
        "write_to": "_echopype_version.py",
        "write_to_template": 'version = "{version}"\n',
    },
    setup_requires=["setuptools>=45", "wheel", "setuptools_scm"],
)


if __name__ == "__main__":
    setup(**opts)
