"""
numpyx. Some accelerated funcs for numpy

"""
import sys
import os
from setuptools import setup, Extension


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload -r pypi')
    sys.exit()
elif sys.argv[-1] == 'testpublish':
    os.system('python setup.py sdif upload -r pypitest')
    sys.exit()

class numpy_get_include(str):
    def __str__(self):
        import numpy
        return numpy.get_include()
        
setup(
    name = "numpyx",
    python_requires=">=3.6",
    setup_requires=['cython', 'numpy'],
    install_requires = ["numpy"],
    ext_modules=[
        Extension(
            'numpyx',
            sources=['numpyx.pyx'],
            include_dirs=[numpy_get_include()]
        )
    ],

    version          = "0.3.1",
    url              = 'https://github.com/gesellkammer/numpyx',
    download_url     = 'https://github.com/gesellkammer/numpyx',
    author           = 'eduardo moguillansky',
    author_email     = 'eduardo.moguillansky@gmail.com',
    description = "Utility functions for numpy, written in cython",
    license="GPL v3",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
