#cython: language_level=3
#cython: binding=True
#cython: embedsignature=True
#cython: infer_types=True
#cython: c_string_type=str, c_string_encoding=ascii
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY, fabs, ceil
from libc cimport stdint


def any_less_than(double[:] a not None, double scalar):
    """
    Is any value in a < scalar?

    Args:
        a (np.ndarray): a 1D double array
        scalar (float): the scalar to compare to

    Returns:
        (bool) True if any value in a is < than scalar
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

    Args:
        a (np.ndarray): a 1D double array
        scalar (float): the scalar to compare to

    Returns:
        (bool) True if any value in a <= scalar
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

    Args:
        a (np.ndarray): a 1D double array
        scalar (float): the scalar to compare to

    Returns:
        (bool) True if any value in a > scalar

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

    Args:
        a (np.ndarray): a 1D double array
        scalar (float): the scalar to compare to

    Returns:
        (bool) True if any value in a >= scalar
    
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


def any_equal_to(double[:] a not None, double scalar,  double tolerance=0):
    """
    Is any value of a == scalar?

    To query if any value of a is different from a scalar just
    use any_less_than

    Args:
        a (np.ndarray): a 1D double array
        scalar (float): the scalar to compare to

    Returns:
        (bool) True if any value in a == scalar
    
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x
    cdef int out = 0
    if tolerance == 0:
        with nogil:
            for i in range(size):
                if a[i] == scalar:
                    out = 1
                    break
    else:
        with nogil:
            for i in range(size):
                if fabs(a[i] - scalar) < tolerance:
                    out = 1
                    break
    return bool(out)


def minmax1d(double[:] a not None):
    """
    Calculate min. and max. of a double 1D-array in one pass

    Args:
        a (np.ndarray): a 1D double array

    Returns:
        (tuple[float, float]) The min and max values within a
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


def array_is_sorted(double [:]xs, bint allowduplicates=True):
    """
    Is the array sorted?
    
    Args:
        xs (np.ndarray): a numpy float array
        allowduplicates (bool): if true (default), duplicate values are still
            considered sorted

    Returns:
        (bool) True if the array is sorted
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
    Like searchsorted, but optimized for 1D double arrays

    Args:
        a (np.ndarray): array to be searched
        v (float | np.ndarray): value/values to "insert" in a
        out: if v is a numpy array, an array `out` can be passed
             which will hold the result. 

    Returns:
        (float | np.ndarray) If v is a scalar, returns an integer, 
        otherwise an array with the same shape as `v`
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
    Like searchsorted, but for 2D arrays

    Only one column is used for searching

    Args:
        xs (np.ndarray): a 2D double array to search
        col (int): indicates which column to use to compare
        x (float): value to "insert" in xs

    Returns:
        (int) the index where x would be inserted to keep xs sorted
    """
    return _searchsorted2(xs, col, x)


def table_interpol_linear(double[:, ::1] table, double[:] xs):
    """
    Interpolate between rows of a 2D matrix

    Given a 2D-array (`table`) with multidimensional Y measurements sampled
    at possibly irregular X, `table_interpol_linear` will interpolate between
    adjacent rows of `table` for each value of `xs`. `xs` contains the x values at which
    to interpolate rows of `table`
    
    Args:
        table (np.ndarray): a 2D array where each row has the form [x_i, a, b, c, ...]. The first
            value of the row is the x coordinate (or time-stamp) and the rest of the
            row contains multiple measurements corresponding to this x.
        xs ((np.ndarray): a 1D array with x values to query the table. For each value in `xs`,
            a whole row of values will be generated from adjacent rows in `table`

    Returns:
        (np.ndarray) An array with the interpolated rows. The result will have as many 
        rows as `xs`, and one column less than the columns of `table`
        
    
    Example
    -------

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
    
    Args:
        A (np.ndarray): the array to query (1D double array)
        x (np.ndarray): the value to search the nearest item 
        sorted (bool): True if A is sorted

    Returns:
        (int) The index in A whose element is closest to x
    """
    cdef int size = A.shape[0]
    if size == 0:
        raise ValueError("array is empty")
    elif size == 1:
        return 0
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
        return idx
    else:
        idx = _searchsorted1(A, x)
        if idx >= size - 1:
            return size - 1

        if abs(A[idx] - x) < abs(A[idx+1] -x):
            return idx
        return idx + 1
    

def nearestitem(double[:] A not None, double[:] V not None, out=None):
    """
    For each value in V, return the element in A which is nearest
    to it.

    Args:
        A (np.ndarray): a 1D double array. The values to choose from
        V (np.ndarray): a 1D double array. The values to snap to A
        out (np.ndarray | None): if given, the values selected from A will 
            be put here. It can't be A itself, but could be V

    Returns:
        (np.ndarray) An array of the same shape as V with values of A, each of 
        each is the nearest value of A to each value of B

    Example
    -------

    ```python

    >>> import numpy as np
    >>> A = np.array([1., 2., 3., 4., 5.])
    >>> V = np.array([0.3, 1.1, 3.4, 10.8])
    >>> nearestitem(A, V)
    array([1., 1., 3., 5.])
    ```
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
        Y (np.ndarray): values
        X (np.ndarray): times corresponding to the Y values
        weights (np.ndarray): weight for each value

    Returns:
        (float) The weighted average (a scalar)

    Example
    -------

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


def allequal(double[:] A not None, double[:] B not None, float tolerance=0.):
    """
    Check if all elements in A == to its corresponding element in B

    Exits early if any inequality is found.
    
    Args:
        A (np.ndarray): a 1D double array
        B (np.ndarray): a 1D double array
        tolerance (float): The tolerance to considere two values equal
        
    Returns:
        (bool) True if all items in A are equal to their corresponding items in B
    """
    cdef int i
    cdef int out = 1
    if tolerance == 0:
        with nogil:
            for i in range(A.shape[0]):
                if A[i] != B[i]:
                    out = 0
                    break
    else:
        with nogil:
            for i in range(A.shape[0]):
                if fabs(A[i] - B[i]) < tolerance:
                    out = 0
                    break
    return bool(out)


def trapz(double[:] Y not None, double[:] X not None):
    """
    A trapz integration routine optimized for doubles
     
    Args:
        Y (np.ndarray): a 1D double array with y coordinates
        X (np.ndarray): a 1D double array with x coordinates

    Returns:
        (float) The surface beneath the curve defined by the points X, Y
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

cdef int _argmax1d(double[:] xs):
    """
    Like argmax but only for 1D double arrays
    """
    cdef size_t i
    cdef size_t idx = 0
    cdef double m = xs[0]
    cdef double x
    for i in range(1, xs.shape[0]):
        x = xs[i]
        if x > m:
            m = x
            idx = i
    return idx


cdef int _argmax1d_row(double[:, ::1] xs, int row):
    cdef size_t i
    cdef size_t idx = 0
    cdef double m = xs[row, 0]
    cdef double x
    for i in range(1, xs.shape[1]):
        x = xs[row, i]
        if x > m:
            m = x
            idx = i
    return idx



def argmax1d(double[:] xs):
    """
    Like argmax but only for 1D double arrays

    Args:
        xs (ndarray): a 1D double array

    Returns:
        the index of the highest element in xs
    """
    cdef size_t i
    cdef size_t idx = 0
    cdef double m = xs[0]
    cdef double x
    for i in range(xs.shape[0]):
        x = xs[i]
        if x > m:
            m = x
            idx = i
    return idx


def aranged(double start, double stop, double step):
    cdef size_t i = 0
    cdef int numitems = int((stop - start) / step)
    cdef double x
    # cdef double[::1] out = np.empty((numitems,), dtype=np.double)
    cdef np.ndarray[double, ndim=1] out = np.empty((numitems,), dtype=np.double)
    for i in range(numitems):
        out[i] = start + i * step
    return np.asarray(out)



def viterbi_core(double[:, ::1] log_prob, double[:, ::1] log_trans, double[::1] log_p_init):
    """
    Core Viterbi algorithm.

    Used internally by algorithms like pyin to perform
    viderbi decoding 

    Args:
        log_prob (np.ndarray [shape=(T, m)]): ``log_prob[t, s]`` is the conditional 
            log-likelihood ``log P[X = X(t) | State(t) = s]``
        log_trans (np.ndarray [shape=(m, m)]): The log transition matrix
            ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
        log_p_init (np.ndarray [shape=(m,)]): log of the initial state distribution

    Returns:
        state, logp
    
    """
    cdef int n_steps = log_prob.shape[0]
    cdef int n_states = log_prob.shape[1]
    # n_steps, n_states = log_prob.shape

    # cdef np.ndarray[np.uint64_t, ndim=1] state = np.zeros(n_steps, dtype=np.uint64)
    cdef np.uint64_t[::1] state = np.zeros(n_steps, dtype=np.uint64)
    cdef double[:, ::1] value = np.zeros((n_steps, n_states), dtype=np.float64)
    cdef np.uint64_t[:, ::1] ptr = np.zeros((n_steps, n_states), dtype=np.uint64)
    cdef np.ndarray log_trans_T = log_trans.T
    cdef size_t t, j, col, i0, j0
    cdef double[:, ::1] trans_out = np.zeros_like(log_trans.T, dtype=np.float64)
    

    # factor in initial state distribution
    # value[0] = log_prob[0] + log_p_init
    for j in range(log_p_init.shape[0]):
        value[0, j] = log_prob[0, j] + log_p_init[j]

    for t in range(1, n_steps):
        # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
        #    assume at time t-1 we were in state k
        #    transition k -> j

        # Broadcast over rows:
        #    Tout[k, j] = V[t-1, k] * A[k, j]
        #    then take the max over columns
        # We'll do this in log-space for stability


        # trans_out = value[t - 1] + log_trans_T
        for i0 in range(log_trans_T.shape[0]):
            for j0 in range(log_trans_T.shape[1]):
                trans_out[i0, j0] = log_trans_T[i0, j0] + value[t-1, j0]
        
        for j in range(n_states):
            # ptr[t, j] = np.argmax(trans_out[j])
            ptr[t, j] = _argmax1d_row(trans_out, j)
            # value[t, j] = log_prob[t, j] + trans_out[j, ptr[t, j]]
            col = ptr[t, j]
            value[t, j] = log_prob[t, j] + trans_out[j, col]
            
    # Now roll backward

    # Get the last state
    # state[-1] = np.argmax(value[-1])
    state[-1] = _argmax1d(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    return state, logp
