# setar-trees
A Python implementation of the R Setar Tree models from Godahewa et al.

Objectives
- porting the default Setar-Trees and Setar-Forest to Python code
    - Writing Python code + tests when critical
    - Defining a sensical API
    - Writing tests + documentation
    - Accelerating the code with Numba
    - benchmarking the models
- Adding extensions to the model
    - Extra-Setar
    - Boosted-Setar?
    - Probabilistic-Setar?


Learnings
- R is frustrating to work with:
    - Lists are 1-indexed
    - `Array[:, -1]` means all columns except the first one ???
    - `XtX = list(); XtX.left = matrix(0,p,p)` is possible, like... why not?
    - True is T and False is F, like why would you want to have explicit naming in your code anyway?
    - length(1) = 1 ...
