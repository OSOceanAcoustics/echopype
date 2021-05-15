from __future__ import absolute_import, division, print_function
from setuptools import setup, find_packages

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
            packages=find_packages(),
            include_package_data=True,
            install_requires=INSTALL_REQUIRES,
            py_modules=["_echopype_version"],
            use_scm_version={
                "fallback_version": "unknown",
                "local_scheme": "node-and-date",
                "write_to": "_echopype_version.py",
                "write_to_template": 'version = "{version}"\n',
            },
            setup_requires=["setuptools>=45", "wheel", "setuptools_scm"],)


if __name__ == '__main__':
    setup(**opts)
