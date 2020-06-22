from __future__ import absolute_import, division, print_function
from os.path import join as pjoin
from setuptools import setup, find_packages
import versioneer

# Long description will go up on the pypi page
with open('README.md') as file:
    LONG_DESCRIPTION = file.read()
    
# Dependencies.
with open('requirements.txt') as f:
    requirements = f.readlines()
INSTALL_REQUIRES = [t.strip() for t in requirements]

opts = dict(name='echopype',
            maintainer='Wu-Jung Lee',
            maintainer_email='leewujung@gmail.com',
            description='Enhancing the interoperability and scalability in analyzing ocean sonar data',
            long_description=LONG_DESCRIPTION,
            long_description_content_type='text/markdown',
            url='https://github.com/OSOceanAcoustics/echopype',
            download_url='',
            license='Apache License, Version 2.0',
            classifiers=['Development Status :: 3 - Alpha',
                         'Environment :: Console',
                         'Intended Audience :: Science/Research',
                         'License :: OSI Approved :: Apache Software License',
                         'Operating System :: OS Independent',
                         'Programming Language :: Python',
                         'Topic :: Scientific/Engineering'],
            author='Wu-Jung Lee',
            author_email='leewujung@gmail.com',
            platforms='OS Independent',
            version=versioneer.get_version(),
            cmdclass=versioneer.get_cmdclass(),
            packages=find_packages(),
            package_data={'echopype': [pjoin('data', '*')]},
            install_requires=INSTALL_REQUIRES,
            tests_require='tox',)


if __name__ == '__main__':
    setup(**opts)
