"""
numpyx. Some accelerated funcs for numpy

"""
import sys
import os
from distutils.core import setup
from Cython.Build import cythonize

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


setup(
    name = "numpyx",
    ext_modules = cythonize('numpyx.pyx'),  # accepts a glob pattern

    # metadata
    version          = "0.1.0",
    url              = 'https://github.com/gesellkammer/numpyx',
    download_url     = 'https://github.com/gesellkammer/numpyx', 
    author           = 'eduardo moguillansky',
    author_email     = 'eduardo.moguillansky@gmail.com',
    maintainer       = '',
    maintainer_email = '',
    description = "Utility functions for numpy, written in cython",
    license="GPL",
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)