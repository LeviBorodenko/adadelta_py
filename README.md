# adadelta_py
Clean & dependency-free implementation of the ADADELTA
algorithm as presented in [this paper](https://arxiv.org/pdf/1212.5701.pdf)

## basic usage
The adadelta() function will need at least 2 things:
  1. A gradient function that takes a vector and outputs
  the gradient of your objective function at that vector.
  (Vectors are just lists)
  2. A initial guess to initialise the algorithm.
Given those 2, adadelta() will try minimise your function.

## advanced usage
Check out the docstring to see all the other parameters
that can be tweaked and further details.

Enjoy.
