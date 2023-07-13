# Reference


---------


| Function  | Description  |
| :-------  | :----------- |
| `allequal` | Check if all elements in A == to its corresponding element in B |
| `any_equal_to` | Is any value of a == scalar? |
| `any_greater_or_equal_than` | Is any value of a >= scalar? |
| `any_greater_than` | Is any value of a > scalar? |
| `any_less_or_equal_than` | Is any value of a <= scalar? |
| `any_less_than` | Is any value in a < scalar? |
| `aranged` | aranged(double start, double stop, double step) |
| `argmax1d` | Like argmax but only for 1D double arrays |
| `array_is_sorted` | Is the array sorted? |
| `minmax1d` | Calculate min. and max. of a double 1D-array in one pass |
| `nearestidx` | Return the index of the element in A which is nearest to x |
| `nearestitem` | For each value in V, return the element in A which is nearest |
| `searchsorted1` | Like searchsorted, but optimized for 1D double arrays |
| `searchsorted2` | Like searchsorted, but for 2D arrays |
| `table_interpol_linear` | Interpolate between rows of a 2D matrix |
| `trapz` | A trapz integration routine optimized for doubles |
| `viterbi_core` | Core Viterbi algorithm. |
| `weightedavg` | Weighted average of a time-series |


---------


## allequal


```python

allequal(double[:] A, double[:] B, float tolerance=0.)

```


Check if all elements in A == to its corresponding element in B


Exits early if any inequality is found.



**Args**

* **A** (`np.ndarray`): a 1D double array
* **B** (`np.ndarray`): a 1D double array
* **tolerance** (`float`): The tolerance to considere two values equal
    (*default*: `0.0`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if all items in A are equal to their corresponding items in B


---------


## any\_equal\_to


```python

any_equal_to(double[:] a, double scalar, double tolerance=0)

```


Is any value of a == scalar?


To query if any value of a is different from a scalar just
use any_less_than



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to
* **tolerance** (`float`):  (*default*: `0.0`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a == scalar


---------


## any\_greater\_or\_equal\_than


```python

any_greater_or_equal_than(double[:] a, double scalar)

```


Is any value of a >= scalar?



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a >= scalar


---------


## any\_greater\_than


```python

any_greater_than(double[:] a, double scalar)

```


Is any value of a > scalar?



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a > scalar


---------


## any\_less\_or\_equal\_than


```python

any_less_or_equal_than(double[:] a, double scalar)

```


Is any value of a <= scalar?



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a <= scalar


---------


## any\_less\_than


```python

any_less_than(double[:] a, double scalar)

```


Is any value in a < scalar?



**Args**

* **a** (`np.ndarray`): a 1D double array
* **scalar** (`float`): the scalar to compare to

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if any value in a is < than scalar


---------


## aranged


```python

def aranged(start, stop, step) -> None

```


aranged(double start, double stop, double step)



**Args**

* **start**:
* **stop**:
* **step**:


---------


## argmax1d


```python

argmax1d(double[:] xs)

```


Like argmax but only for 1D double arrays



**Args**

* **xs** (`ndarray`): a 1D double array

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the index of the highest element in xs


---------


## array\_is\_sorted


```python

array_is_sorted(double[:] xs, bool allowduplicates=True)

```


Is the array sorted?



**Args**

* **xs** (`np.ndarray`): a numpy float array
* **allowduplicates** (`bool`): if true (default), duplicate values are still
    considered sorted (*default*: `True`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`bool`) True if the array is sorted


---------


## minmax1d


```python

minmax1d(double[:] a)

```


Calculate min. and max. of a double 1D-array in one pass



**Args**

* **a** (`np.ndarray`): a 1D double array

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`tuple[float, float]`) The min and max values within a


---------


## nearestidx


```python

nearestidx(double[:] A, double x, bool sorted=False)

```


Return the index of the element in A which is nearest to x



**Args**

* **A** (`np.ndarray`): the array to query (1D double array)
* **x** (`np.ndarray`): the value to search the nearest item
* **sorted** (`bool`): True if A is sorted (*default*: `False`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`int`) The index in A whose element is closest to x


---------


## nearestitem


```python

nearestitem(double[:] A, double[:] V, out=None)

```


For each value in V, return the element in A which is nearest


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
    be put here. It can't be A itself, but could be V (*default*: `None`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`np.ndarray`) An array of the same shape as V with values of A, each of each is the nearest value of A to each value of B


---------


## searchsorted1


```python

searchsorted1(a, v, out=None)

```


Like searchsorted, but optimized for 1D double arrays



**Args**

* **a** (`np.ndarray`): array to be searched
* **v** (`float | np.ndarray`): value/values to "insert" in a
* **out**: if v is a numpy array, an array `out` can be passed          which
    will hold the result.  (*default*: `None`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`float | np.ndarray`) If v is a scalar, returns an integer, otherwise an array with the same shape as `v`


---------


## searchsorted2


```python

searchsorted2(xs, col, x)

```


Like searchsorted, but for 2D arrays


Only one column is used for searching



**Args**

* **xs** (`np.ndarray`): a 2D double array to search
* **col** (`int`): indicates which column to use to compare
* **x** (`float`): value to "insert" in xs

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`int`) the index where x would be inserted to keep xs sorted


---------


## table\_interpol\_linear


```python

table_interpol_linear(double[:, ::1] table, double[:] xs)

```


Interpolate between rows of a 2D matrix


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


```python

trapz(double[:] Y, double[:] X)

```


A trapz integration routine optimized for doubles



**Args**

* **Y** (`np.ndarray`): a 1D double array with y coordinates
* **X** (`np.ndarray`): a 1D double array with x coordinates

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`float`) The surface beneath the curve defined by the points X, Y


---------


## viterbi\_core


```python

viterbi_core(double[:, ::1] log_prob, double[:, ::1] log_trans, double[::1] log_p_init)

```


Core Viterbi algorithm.


Used internally by algorithms like pyin to perform
viderbi decoding 



**Args**

* **log_prob** (`np.ndarray [shape=(T, m)]`): ``log_prob[t, s]`` is the
    conditional         log-likelihood ``log P[X = X(t) | State(t) = s]``
* **log_trans** (`np.ndarray [shape=(m, m)]`): The log transition matrix
    ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
* **log_p_init** (`np.ndarray [shape=(m,)]`): log of the initial state
    distribution

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;state, logp


---------


## weightedavg


```python

weightedavg(double[:] Y, double[:] X, double[:] weights)

```


Weighted average of a time-series


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