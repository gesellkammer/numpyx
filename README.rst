A small package with fast numpy routines written in cython

Documentation
-------------

https://numpyx.readthedocs.io

Installation
------------

    pip install numpyx

-----


Functions in this package
-------------------------

All functions here are specialized for double arrays only

Short-cut functions
~~~~~~~~~~~~~~~~~~~

These functions are similar to numpy functions but are faster by
exiting out of a loop when one element satisfies the given condition


* any_less_than
* any_less_or_equal_than
* any_greater_than
* any_greater_or_equal_than
* any_equal_to
* array_is_sorted
* allequal

minmax1d
~~~~~~~~

Calculate min. and max. value in one go

searchsorted1
~~~~~~~~~~~~~

like search sorted, but for 1d double arrays. It is faster than the more generic numpy version


searchsorted2
~~~~~~~~~~~~~

like search sorted but allows to search across any column of a 2d array


nearestidx
~~~~~~~~~~

Return the index of the item in an array which is nearest to a given value. The
array does not need to be sorted (this is a simple linear search)


nearestitem
~~~~~~~~~~~

For any value of an array, search the nearest item in another array and put its
value in the output result


weightedavg
~~~~~~~~~~~

Weighted averageof a time-series


trapz
~~~~~

trapz integration specialized for contiguous / double arrays. Quite faster than generic numpy/scipy 