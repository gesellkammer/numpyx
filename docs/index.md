# Home

Welcome to the **numpyx** documentation!

**numpyx** is a small package with fast miscellaneous numpy routines written in cython

## Installation

Binary wheels are provided for all major platforms:

``` bash
pip install numpyx
```

## Quick Introduction

### Short circuit functions

In **numpy**, searching for a value or checking if any value is equal or less than a given
scalar implies querying the entire array. **numpyx** provides functions like `any_less_than` or
`any_equal_to` which exit the loop once the condition is met.

### Optimized versions

**numpyx** provides versions of some functions, like `searchsorted` or `trapz` which are optimized for
a given type of arrays (1D double arrays, 2D double arrays). 

## Reference

See [reference](reference.md)

