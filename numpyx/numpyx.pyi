import numpy as np

def any_less_than(a: np.ndarray, scalar: float) -> bool: ...

def any_less_or_equal_than(a: np.ndarray, scalar: float) -> bool: ...

def any_greater_than(a: np.ndarray, scalar: float) -> bool: ...

def any_greater_or_equal_than(a: np.ndarray, scalar: float) -> bool: ...

def any_equal_to(a: np.ndarray, scalar: float, tolerance: float = 0.) -> bool: ...

def minmax1d(a: np.ndarray) -> tuple[float, float]: ...

def array_is_sorted(xs: np.ndarray, allowduplicates: bool = True) -> bool: ...

def searchsorted1(a: np.ndarray, v: float | np.ndarray, out: np.ndarray | None = None) -> float | np.ndarray: ...

def searchsorted2(xs: np.ndarray, col: int, x: float) -> int: ...

def table_interpol_linear(table: np.ndarray, xs: np.ndarray) -> np.ndarray: ...

def nearestidx(A: np.ndarray, x: float, sorted=False) -> int: ...

def nearestitemsorted(A: np.ndarray, V: np.ndarray) -> np.ndarray: ...

def nearestitem(A: np.ndarray, x: float, sorted=False) -> float: ...

def weightedavg(Y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> float: ...

def allequal(A: np.ndarray, B: np.ndarray, tolerance=0.) -> bool: ...

def trapz(Y: np.ndarray, X: np.ndarray) -> float: ...

def argmax1d(xs: np.ndarray) -> int: ...

def aranged(start: float, stop: float, step: float) -> np.ndarray: ...

def amp_follow(samples: np.ndarray, attack=0.01, release=0.01, chunksize=256) -> np.ndarray: ...
