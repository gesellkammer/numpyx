"""
numpyx. Some accelerated funcs for numpy

"""
import sys
import os
# from distutils.core import setup
from setuptools import setup, Extension
# from Cython.Build import cythonize

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload -r pypi')
    sys.exit()
elif sys.argv[-1] == 'testpublish':
    os.system('python setup.py sdif upload -r pypitest')
    sys.exit()


setup(
    name = "numpyx",
    setup_requires=[
        'cython'
    ],
    ext_modules=[
        Extension(
            'numpyx',
            sources=['numpyx.pyx'],
        ),
    ],
    # ext_modules = cythonize('numpyx.pyx'),  # accepts a glob pattern

    # metadata
    version          = "0.1.5",
    url              = 'https://github.com/gesellkammer/numpyx',
    download_url     = 'https://github.com/gesellkammer/numpyx', 
    author           = 'eduardo moguillansky',
    author_email     = 'eduardo.moguillansky@gmail.com',
    maintainer       = '',
    maintainer_email = '',
    install_requires = ["numpy", "cython"],
    description = "Utility functions for numpy, written in cython",
    license="GPL v3",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)
