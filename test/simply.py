import numpyx
import numpy as np

A = np.arange(0, 10, 0.5)
V = np.array([0., 0.6, 1.0, 1.1, 2.4])
out = np.empty((V.shape[0],), dtype=int)

for i in range(10):
    out2 = numpyx.searchsorted1(A, V, out)
    assert out2 is out
    print(out)
    print(V)
    print(A[out])
