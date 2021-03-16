#cython: embedsignature=True
#cython: infer_types=True
#cython: c_string_type=str, c_string_encoding=ascii
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY


def any_less_than(double[:] a not None, double scalar):
    """
    Is any value of a < scalar?
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x
    cdef int out = 0
    with nogil:
        for i in range(size):
            if a[i] < scalar:
                out = 1
                break
    return bool(out)

    
def any_less_or_equal_than(double[:] a not None, double scalar):
    """
    Is any value of a <= scalar?
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x
    cdef int out = 0
    with nogil:
        for i in range(size):
            if a[i] <= scalar:
                out = 1
                break
    return bool(out)


def any_greater_than(double[:] a not None, double scalar):
    """
    Is any value of a > scalar?
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x
    cdef int out = 0
    with nogil:
        for i in range(size):
            if a[i] > scalar:
                out = 1
                break
    return bool(out)


def any_greater_or_equal_than(double[:] a not None, double scalar):
    """
    Is any value of a >= scalar?
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x
    cdef int out = 0
    with nogil:
        for i in range(size):
            if a[i] >= scalar:
                out = 1
                break
    return bool(out)


def any_equal_to(double[:] a not None, double scalar):
    """
    Is any value of a == scalar?

    To query if any value of a is different from a scalar just
    use any_less_than
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x
    cdef int out = 0
    with nogil:
        for i in range(size):
            if a[i] == scalar:
                out = 1
                break
    return bool(out)


def minmax1d(double[:] a not None):
    """
    Calculate min. and max. of a double 1D-array in one go
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x0 = INFINITY
    cdef double x1 = -INFINITY
    cdef double x
    with nogil:
        for i in range(size):
            x = a[i]
            if x < x0:
                x0 = x
            elif x > x1:
                x1 = x
    return x0, x1


def array_is_sorted(double [:]xs, int allowduplicates=1):
    """
    Is the array sorted?
    
    Args:
        xs: a numpy float array
        allowduplicates: if true (default), duplicate values are still
            considered sorted
    """
    cdef double x0 = -INFINITY
    cdef double x
    cdef int out = 1
    with nogil:
        if not allowduplicates:
            for i in range(xs.shape[0]):
                x = xs[i]
                if x < x0:
                    out = 0
                    break
                x0 = x
        else:
            for i in range(xs.shape[0]):
                x = xs[i]
                if x <= x0:
                    out = 0
                    break
                x0 = x
    return bool(out)


cdef inline void _interpol(double[:, :] table, int idx, double delta, double[:, ::1] out, 
                           int outidx, int numcols) nogil:
    cdef:
        int i
        double y0, y1, y
    for i in range(1, numcols):
        y0 = table[idx, i]
        y1 = table[idx+1, i]
        y = y0 + (y1 - y0) * delta
        out[outidx, i - 1] = y

cdef inline void _putrow(double[:, ::1] out, int outidx, double[:, :] table, int tableidx) nogil:
    cdef int i
    cdef int out_numcols = out.shape[1]
    for i in range(1, table.shape[1]):
        out[outidx, i-1] = table[tableidx, i]


def searchsorted1(a, v, out=None):
    """
    Like searchsorted, but for 1d double arrays

    Args:
        a: array to be searched
        v: value/values to "insert" in a
        out: if v is a numpy array, an array `out` can be passed
             which will hold the result. 
    """
    if isinstance(v, np.ndarray):
        if out is None:
            out = np.empty((v.shape[0],), dtype="long")
        _searchsorted1x(a, v, out)
        return out
    elif isinstance(v, float):
        return _searchsorted1(a, v)
    else:
        raise TypeError(f"v: expected numpy array or float, got {type(v)}")


cdef int _searchsorted1(double[:] A, double x) nogil: 
    cdef:
        int imin = 0
        int imax = A.shape[0]
        int imid
    while imin < imax:
        imid = imin + ((imax - imin) / 2)
        if A[imid] < x:
            imin = imid + 1
        else:
            imax = imid
    return imin


cdef void _searchsorted1x(double[:] A, double[:] V, long[:] out) nogil:
    cdef:
        int imin = 0
        int imaxidx = A.shape[0]
        int imid
        int i
        double x
    for i in range(imaxidx):
        imin = 0 
        imax = imaxidx
        x = V[i]
        while imin < imax:
            imid = imin + ((imax - imin) / 2)
            if A[imid] < x:
                imin = imin + 1
            else:
                imax = imid
        out[i] = imin 
    

cdef int _searchsorted2(double[:, :] xs, int col, double x) nogil:    
    cdef:
        int imin = 0
        int imax = xs.shape[0]
        int imid
    with nogil:
        while imin < imax:
            imid = imin + ((imax - imin) / 2)
            if xs[imid, col] < x:
                imin = imid + 1
            else:
                imax = imid
    return imin


def searchsorted2(xs, col, x):
    """
    Like searchsorted, but for 2d arrays, where only 
    one column is used for searching

    Args:
        xs: data
        col: indicates which column to use to compare
        x : value to "insert" in xs

    Returns:
        the index where x would be inserted to keep xs sorted
    """
    return _searchsorted2(xs, col, x)


def table_interpol_linear(double[:, ::1] table, double[:] xs):
    """
    Given a 2D-array (`table`) with multidimensional Y measurements sampled
    at possibly irregular X, `table_interpol_linear` will interpolate between
    adjacent rows of `table` for each value of `xs`. `xs` contains the x values at which
    to interpolate rows of `table`
    
    Args:
        table: a 2D array where each row has the form [x_i, a, b, c, ...]. The first
            value of the row is the x coordinate (or time-stamp) and the rest of the
            row contains multiple measurements corresponding to this x.
        xs: a 1D array with x values to query the table. For each value in `xs`,
            a whole row of values will be generated from adjacent rows in `table`

    Returns:
        an array with the interpolated rows. The result will have as many rows as `xs`,
        and one column less than the columns of `table`
        
    Example
    ~~~~~~~

    
        >>> A = np.array([[0, 0, 1, 2, 3,   4]
        ...               [1, 0, 2, 4, 6,   8]
        ...               [2, 0, 4, 8, 12, 16]], dtype=float)
        >>> xs = np.array([0.5, 1.5, 2.2])
        >>> table_interpol_linear(A, xs)
        array([[0.5, 0., 1.5, 3., 4.5, 6. ]
               [1.5, 0., 3.,  6., 9.,  12.]
               [2.,  0., 4.,  8., 12., 16.]])

    NB: the resampled table has no `x` column, which would be a 
        copy of the sampling points `xs`, and thus has one column
        less than the table. To build a table with the given xs as
        first column, do:

        >>> resampled = table_interpol_linear(table, xs)
        >>> table2 = np.hstack((xs.reshape(xs.shape[0], 1), resampled))
    """
    cdef:
        double[:, ::1] out = np.empty((xs.shape[0], table.shape[1] - 1))
        int idx 
        int lastidx = 0
        int table_numrows = table.shape[0]
        int table_numcols = table.shape[1]
        int table_numrows_minus_2 = table_numrows - 2
        double t0 = table[0, 0]
        double t1 = table[table_numrows - 1, 0]
        double last_t0 = -1
        double last_t1 = -2
        double last_diff = 1
        double x, delta
        int i = 0
        int error = 0
        int numxs = xs.shape[0]
        double [:] ts = table[:,0]
    while i < numxs:
        x = xs[i]
        if x > t0:
            break
        _putrow(out, i, table, 0)
        i += 1
    with nogil:
        for i in range(i, numxs):
            x = xs[i]
            if last_t0 <= x < last_t1:
                delta = (x - last_t0) / last_diff
                _interpol(table, lastidx, delta, out, i, table_numcols)
            elif lastidx < table_numrows_minus_2 and last_t1 <= x < table[lastidx+2, 0]:
                last_t0 = last_t1
                last_t1 = table[lastidx+2, 0]
                last_diff = last_t1 - last_t0
                lastidx = lastidx + 1
                delta = (x - last_t0) / last_diff
                _interpol(table, lastidx, delta, out, i, table_numcols)
            elif x >= t1:
                _putrow(out, i, table, table_numrows - 1)
            else:
                # idx = _searchsorted2(table, 0, x) - 1
                idx = _searchsorted1(ts, x) - 1
                last_t0 = table[idx, 0]
                last_t1 = table[idx+1, 0]
                last_diff = last_t1 - last_t0
                if last_diff == 0:
                    error = 1
                    break
                delta = (x - last_t0) / last_diff
                _interpol(table, idx, delta, out, i, table_numcols)
                lastidx = idx
    if error == 1:
        raise RuntimeError("Values along the 0 axis should be sorted")
    return np.asarray(out)


def nearestidx(double[:] A not None, double x, bint sorted=False):
    """
    Return the index of the element in A which is nearest to x
    """
    cdef int size = A.shape[0]
    cdef double smallest = INFINITY
    cdef int idx = 0
    cdef double diff
    if not sorted:
        for i in range(size):
            diff = abs(A[i] - x)
            if diff == 0:
                return i
            if diff < smallest:
                smallest = diff
                idx = i
    else:
        idx = _searchsorted1(A, x)
        if abs(A[idx] - x) < abs(A[idx+1] -x):
            return idx
        return idx + 1 
    return idx


def nearestitem(double[:] A not None, double[:] V not None, out=None):
    """
    For each value in V, return the element in A which is nearest
    to it.

    Args:
        A: a 1D double array. The values to choose from
        V: a 1D double array. The values to snap to A
        out: if given, the values selected from A will be put here. It can't
            be A itself, but could be V

    Returns:
        an array of the same shape as V with values of A, each of each is
        the nearest value of A to each value of B

    Example
    ~~~~~~~

    >>> import numpy as np
    >>> A = np.array([1., 2., 3., 4., 5.])
    >>> V = np.array([0.3, 1.1, 3.4, 10.8])
    >>> nearestitem(A, V)
    array([1., 1., 3., 5.])
    """
    cdef long[:] Idxs = np.searchsorted(A, V)
    cdef int i, idx
    cdef double a0, a1, v
    cdef double[:] O 
    cdef int Vsize = V.shape[0]
    cdef int maxidx = A.shape[0] - 1
    if out is not None:
        O = out
    else:
        O = np.empty((Vsize,), dtype=np.double)
    for i in range(Vsize):
        idx = Idxs[i]
        if idx > maxidx:
            idx = maxidx
        a1 = A[idx]
        a0 = A[idx - 1] if idx >= 1 else a1
        v = V[i]
        if v - a0 < a1 - v:
            O[i] = a0
        else:
            O[i] = a1
    return np.asarray(O)


def weightedavg(double[:] Y not None, double [:] X not None, double[:] weights not None):
    """
    Weighted average of a time-series

    Args:
        Y: values
        X: times corresponding to the Y values
        weights: weight for each value

    Returns:
        the weighted average (a scalar)

    Example
    ~~~~~~~

    >>> # Given a time-series of the fundamental frequency of a sound together
    >>> # with its amplitude, calculate an average using the amplitude as weight
    >>> import numpy as np
    >>> freqs = np.array([444., 442., 443.])
    >>> times = np.array([0.,   1.,   2.])
    >>> amps  = np.array([0.1,  0.3,  0.2])
    >>> weightedavg(freqs, times, amps)
    442.6667
    
    """
    if Y.is_c_contig() and X.is_c_contig() and weights.is_c_contig():
        return _weightedavg_contiguous(&Y[0], &X[0], &weights[0], len(Y))
    return _weightedavg(Y, X, weights)


cdef double _weightedavg(double[:] Y, double[:] X, double[:] weights):
    cdef int i
    cdef double x0, x1, y0, y1, dx, yavg, wavg, w0, w1
    cdef double accum = 0
    cdef double accumw = 0
    x0 = X[0]
    y0 = Y[0]
    w0 = weights[0]
    for i in range(1, X.shape[0]):
        x1 = X[i]
        y1 = Y[i]
        w1 = weights[i]
        dx = x1 - x0
        yavg = (y0 + y1) * 0.5
        wavg = (w0 + w1) * 0.5 * dx
        accum += yavg * wavg
        accumw += wavg
        x0 = x1 
        y0 = y1
        w0 = w1
    return accum / accumw


cdef double _weightedavg_contiguous(double* Y, double* X, double* weights, int size):
    cdef int i
    cdef double x0, x1, y0, y1, dx, yavg, wavg, w0, w1
    cdef double accum = 0
    cdef double accumw = 0
    x0 = X[0]
    y0 = Y[0]
    w0 = weights[0]
    for i in range(1, size):
        x1 = X[i]
        y1 = Y[i]
        w1 = weights[i]
        dx = x1 - x0
        yavg = (y0 + y1) * 0.5
        wavg = (w0 + w1) * 0.5 * dx
        accum += yavg * wavg
        accumw += wavg
        x0 = x1 
        y0 = y1
        w0 = w1
    return accum / accumw


def allequal(double[:] A not None, double[:] B not None):
    """
    Check if all elements in A are equal to its corresponding element in B,
    exit early if any inequality is found.
    """
    cdef int i
    for i in range(A.shape[0]):
        if A[i] != B[i]:
            return False
    return True


def trapz(double[:] Y not None, double[:] X not None):
    """
    A trapz integration routinge optimized for doubles
    """
    if Y.is_c_contig() and X.is_c_contig():
        return _trapz_contiguous(&Y[0], &X[0], len(Y))
    return _trapz(Y, X)


cdef double _trapz(double[:] Y, double[:] X):
    cdef int i
    cdef double x0, x1, y0, y1, area, dx
    x0 = X[0]
    y0 = Y[0]
    cdef double total = 0
    for i in range(1, X.shape[0]):
        x1 = X[i]
        y1 = Y[i]
        dx = x1 - x0
        area = (y0 + y1) * 0.5 * dx
        total += area
        x0 = x1 
        y0 = y1
    return total


cdef double _trapz_contiguous(double *Y, double *X, int size):
    cdef int i
    cdef double x0, x1, y0, y1, area, dx
    x0 = X[0]
    y0 = Y[0]
    cdef double total = 0
    for i in range(1, size):
        x1 = X[i]
        y1 = Y[i]
        dx = x1 - x0
        area = (y0 + y1) * 0.5 * dx
        total += area
        x0 = x1 
        y0 = y1
    return total
    
    

