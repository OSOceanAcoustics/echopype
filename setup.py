import os
from setuptools import setup, find_packages
import versioneer
PACKAGES = find_packages()

# Get version and release info, which is all stored in echopype/version.py
ver_file = os.path.join('echopype', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Long description will go up on the pypi page
with open('README.rst') as file:
    LONG_DESCRIPTION = file.read()
    
# Dependencies.
with open('requirements.txt') as f:
    requirements = f.readlines()
INSTALL_REQUIRES = [t.strip() for t in requirements]

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=versioneer.get_version(),
            cmdclass=versioneer.get_cmdclass(),
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=INSTALL_REQUIRES,
            requires=REQUIRES,
            test_requires=TEST_REQUIRES,
            scripts=SCRIPTS)


if __name__ == '__main__':
    setup(**opts)
