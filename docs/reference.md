# Reference


---------


## allequal


Check if all elements in A == to its corresponding element in B


```python

def allequal(A: np.ndarray, B: np.ndarray) -> bool

```


Exits early if any inequality is found.



**Args**

* **A** (`np.ndarray`): a 1D double array
* **B** (`np.ndarray`): a 1D double array

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if all items in A are equal to their corresponding items in B


---------


## any\_equal\_to


Is any value of a == scalar?


```python

def any_equal_to(a: np.ndarray, scalar: float, tolerance: float = 0.0) -> bool

```


To query if any value of a is different from a scalar just
use any_less_than



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to
* **tolerance** (`float`):  (default: 0.0)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a == scalar


---------


## any\_greater\_or\_equal\_than


Is any value of a >= scalar?


```python

def any_greater_or_equal_than(a: np.ndarray, scalar: float) -> bool

```



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a >= scalar


---------


## any\_greater\_than


Is any value of a > scalar?


```python

def any_greater_than(a: np.ndarray, scalar: float) -> bool

```



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a > scalar


---------


## any\_less\_or\_equal\_than


Is any value of a <= scalar?


```python

def any_less_or_equal_than(a: np.ndarray, scalar: float) -> bool

```



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a <= scalar


---------


## any\_less\_than


Is any value in a < scalar?


```python

def any_less_than(a: np.ndarray, scalar: float) -> bool

```



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a is < than scalar


---------


## array\_is\_sorted


Is the array sorted?


```python

def array_is_sorted(xs: np.ndarray, allowduplicates: bool = True) -> bool

```



**Args**

* **xs** (`np.ndarray`): a numpy float array
* **allowduplicates** (`bool`): if true (default), duplicate values are still
    considered sorted (default: True)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if the array is sorted


---------


## minmax1d


Calculate min. and max. of a double 1D-array in one pass


```python

def minmax1d(a: np.ndarray) -> tuple[float, float]

```



**Args**

* **a** (`np.ndarray`): a 1D double array

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`tuple[float, float]`) The min and max values within a


---------


## nearestidx


Return the index of the element in A which is nearest to x


```python

def nearestidx(A: np.ndarray, x: np.ndarray, sorted: bool = False) -> int

```



**Args**

* **A** (`np.ndarray`): the array to query (1D double array)
* **x** (`np.ndarray`): the value to search the nearest item
* **sorted** (`bool`): True if A is sorted (default: False)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`int`) The index in A whose element is closest to x


---------


## nearestitem


For each value in V, return the element in A which is nearest


```python

def nearestitem(A: np.ndarray, V: np.ndarray, out: np.ndarray | None = None
                ) -> np.ndarray

```


to it.

### Example

```python

>>> import numpy as np
>>> A = np.array([1., 2., 3., 4., 5.])
>>> V = np.array([0.3, 1.1, 3.4, 10.8])
>>> nearestitem(A, V)
array([1., 1., 3., 5.])
```



**Args**

* **A** (`np.ndarray`): a 1D double array. The values to choose from
* **V** (`np.ndarray`): a 1D double array. The values to snap to A
* **out** (`np.ndarray | None`): if given, the values selected from A will
    be put here. It can't be A itself, but could be V (default: None)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`np.ndarray`) An array of the same shape as V with values of A, each of each is the nearest value of A to each value of B


---------


## searchsorted1


Like searchsorted, but optimized for 1D double arrays


```python

def searchsorted1(a: np.ndarray, v: float | np.ndarray, out=None
                  ) -> float | np.ndarray

```



**Args**

* **a** (`np.ndarray`): array to be searched
* **v** (`float | np.ndarray`): value/values to "insert" in a
* **out**: if v is a numpy array, an array `out` can be passed          which
    will hold the result.  (default: None)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`float | np.ndarray`) If v is a scalar, returns an integer, otherwise an array with the same shape as `v`


---------


## searchsorted2


Like searchsorted, but for 2D arrays


```python

def searchsorted2(xs: np.ndarray, col: int, x: float) -> int

```


Only one column is used for searching



**Args**

* **xs** (`np.ndarray`): a 2D double array to search
* **col** (`int`): indicates which column to use to compare
* **x** (`float`): value to "insert" in xs

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`int`) the index where x would be inserted to keep xs sorted


---------


## table\_interpol\_linear


Interpolate between rows of a 2D matrix


```python

def table_interpol_linear(table: np.ndarray, xs: (np.ndarray) -> np.ndarray

```


Given a 2D-array (`table`) with multidimensional Y measurements sampled
at possibly irregular X, `table_interpol_linear` will interpolate between
adjacent rows of `table` for each value of `xs`. `xs` contains the x values at which
to interpolate rows of `table`


### Example

```python

>>> A = np.array([[0, 0, 1, 2, 3,   4]
...               [1, 0, 2, 4, 6,   8]
...               [2, 0, 4, 8, 12, 16]], dtype=float)
>>> xs = np.array([0.5, 1.5, 2.2])
>>> table_interpol_linear(A, xs)
array([[0.5, 0., 1.5, 3., 4.5, 6. ]
       [1.5, 0., 3.,  6., 9.,  12.]
       [2.,  0., 4.,  8., 12., 16.]])
```

The resampled table has no `x` column, which would be a 
copy of the sampling points `xs`, and thus has one column
less than the table. To build a table with the given xs as
first column, do:

```python
>>> resampled = table_interpol_linear(table, xs)
>>> table2 = np.hstack((xs.reshape(xs.shape[0], 1), resampled))
```



**Args**

* **table** (`np.ndarray`): a 2D array where each row has the form [x_i, a, b,
    c, ...]. The first         value of the row is the x coordinate (or time-
    stamp) and the rest of the         row contains multiple measurements
    corresponding to this x.
* **xs** (`(np.ndarray`): a 1D array with x values to query the table. For each
    value in `xs`,         a whole row of values will be generated from adjacent
    rows in `table`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`np.ndarray`) An array with the interpolated rows. The result will have as many rows as `xs`, and one column less than the columns of `table`


---------


## trapz


A trapz integration routinge optimized for doubles


```python

def trapz(Y: np.ndarray, X: np.ndarray) -> float

```



**Args**

* **Y** (`np.ndarray`): a 1D double array with y coordinates
* **X** (`np.ndarray`): a 1D double array with x coordinates

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`float`) The surface beneath the curve defined by the points X, Y


---------


## weightedavg


Weighted average of a time-series


```python

def weightedavg(Y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> float

```


### Example

```python

>>> # Given a time-series of the fundamental frequency of a sound together
>>> # with its amplitude, calculate an average using the amplitude as weight
>>> import numpy as np
>>> freqs = np.array([444., 442., 443.])
>>> times = np.array([0.,   1.,   2.])
>>> amps  = np.array([0.1,  0.3,  0.2])
>>> weightedavg(freqs, times, amps)
442.6667

```



**Args**

* **Y** (`np.ndarray`): values
* **X** (`np.ndarray`): times corresponding to the Y values
* **weights** (`np.ndarray`): weight for each value

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`float`) The weighted average (a scalar)