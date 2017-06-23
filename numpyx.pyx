#cython: embedsignature=True
#cython: infer_types=True
#cython: c_string_type=str, c_string_encoding=ascii
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
# from numpy.math cimport INFINITY
from libc.math cimport INFINITY


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


def minmax2d(double[:, :] a, int col):
    """
    Calculate min. and max. of a double 2D-array in one go
    along one column
    """
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x0 = INFINITY
    cdef double x1 = -INFINITY
    cdef double x
    with nogil:
        for i in range(size):
            x = a[col, i]
            if x < x0:
                x0 = x
            elif x > x1:
                x1 = x
    return x0, x1
    


def array_is_sorted(double [:]xs, int allowduplicates=0):
    """
    xs: a (numpy) array of double
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


cdef void _interpol(double[:, :] table, int idx, double delta, double[:, :] out, int outidx, int numcols):
    cdef:
        int i
        double y0, y1, y
    for i in range(1, numcols):
        y0 = table[idx, i]
        y1 = table[idx+1, i]
        y = y0 + (y1 - y0) * delta
        out[outidx, i - 1] = y

cdef inline void _putrow(double[:, :] out, int outidx, double[:, :] table, int tableidx):
    cdef int i
    for i in range(1, table.shape[1]):
        out[outidx, i-1] = table[tableidx, i]


cdef int _searchsorted2(double[:, :] xs, int col, double x):
    """
    Like searchsorted, but for 2d arrays

    xs: data
    col: indicates which column to use to compare
    x: value to "insert" in xs
    """
    cdef:
        int imin = 0
        int imax = xs.shape[0]
        int imid
    with nogil:
        while imin < imax:
            imid = imin + ((imax - imin) >> 2)
            if xs[imid, col] < x:
                imin = imid + 1
            else:
                imax = imid
    return imin


def searchsorted2(xs, col, x):
    return _searchsorted2(xs, col, x)


def table_interpol_linear(double[:, :] table, double[:] xs):
    cdef:
        double[:, :] out = np.empty((xs.shape[0], table.shape[1] - 1))
        int idx, lastidx
        int table_numrows = table.shape[0]
        int table_numcols = table.shape[1]
        double t0 = table[0, 0]
        double t1 = table[table_numrows - 1, 0]
        double last_t0 = -1
        double last_t1 = -2
        double last_diff = 1
        int outidx = 0
        double x, delta
        int i
        int error = 0
    for i in range(xs.shape[0]):
        x = xs[i]
        if last_t0 <= x < last_t1:
            delta = (x - last_t0) / last_diff
            _interpol(table, lastidx, delta, out, outidx, table_numcols)
        elif x <= t0:
            _putrow(out, outidx, table, 0)
            # out[outidx] = table[0][1:]
        elif x >= t1:
            _putrow(out, outidx, table, table_numrows - 1)
            # out[outidx] = table[table_numrows-1][1:]
        # agregar un caso para chequear el proximo bin
        else:
            idx = _searchsorted2(table, 0, x) - 1
            last_t0 = table[idx, 0]
            last_t1 = table[idx+1, 0]
            last_diff = last_t1 - last_t0
            if last_diff == 0:
                error = 1
                break
            delta = (x - last_t0) / last_diff
            _interpol(table, idx, delta, out, outidx, table_numcols)
            lastidx = idx
        outidx += 1
    if error == 1:
        raise RuntimeError("Values along the 0 axis should be sorted")
    return np.asarray(out)