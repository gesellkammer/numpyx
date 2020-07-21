"""
numpyx. Some accelerated funcs for numpy

"""
from setuptools import setup, Extension

VERSION = "0.3.2"

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

    version          = VERSION,
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
