from setuptools import setup, Extension
import os

VERSION = "1.3.0"

class numpy_get_include(str):
    def __str__(self):
        import numpy
        return numpy.get_include()
        
# read the contents of your README file
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.rst')) as f:
    long_description = f.read()
    
setup(
    name = "numpyx",
    python_requires=">=3.8",
    setup_requires=['cython', 
                    'numpy>=1.22'],
    install_requires = ["numpy>=1.22"],
    ext_modules=[
        Extension(
            'numpyx',
            sources=['numpyx.pyx'],
            include_dirs=[numpy_get_include()]
        )
    ],

    version          = VERSION,
    url              = 'https://github.com/gesellkammer/numpyx',
    download_url     = 'https://github.com/gesellkammer/numpyx',
    author           = 'eduardo moguillansky',
    author_email     = 'eduardo.moguillansky@gmail.com',
    description = "Utility functions for numpy, written in cython",
    long_description = long_description,
    license="GPL v3",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ]
)
