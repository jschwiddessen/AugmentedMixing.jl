# AugmentedMixing

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jschwiddessen.github.io/AugmentedMixing.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jschwiddessen.github.io/AugmentedMixing.jl/dev/)
[![Build Status](https://github.com/jschwiddessen/AugmentedMixing.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jschwiddessen/AugmentedMixing.jl/actions/workflows/CI.yml?query=branch%3Amain)

__AugmentedMixing.jl__ is a Julia implementation of the **Augmented Mixing Method** for solving general large-scale SDPs of the form

```math
\begin{array}{rl}
\text{minimize} \quad & \displaystyle \sum_{i=1}^q \langle C_i, X_i \rangle \\[1.2ex]
\text{subject to} \quad & \displaystyle \sum_{i=1}^q \mathcal{A}_i(X_i) = a, \\\\
& \displaystyle \sum_{i=1}^q \mathcal{B}_i(X_i) \geq b, \\\\
& X_i \in \mathcal{S}_+^{n_i}, \quad i = 1,\ldots, q.
\end{array}
```

This method features a Burer-Monteiro factorization-based algorithm in which all factorization matrices are updated in a column-wise fashion and is in particular designed to handle a large number of inequality constraints.

## Installation
This package is registered in the Julia General registry. To install it, do the following:

From the Julia REPL, press `]` to enter Pkg mode and run:
```julia-repl
pkg> add AugmentedMixing
```
**Requirements:** Julia v1.11 or later.

## Example
Let us consider the following SDP which is equivalent to the basic Max-Cut relaxation of some graph:
```math
\begin{array}{rl}
\text{minimize} \quad & \displaystyle \langle C , X \rangle \\[1.2ex]
\text{subject to} \quad & \displaystyle X_{ii} = 1, \quad i = 1,\ldots, 5, \\\\
& X \in \mathcal{S}_+^{5},
\end{array}
```
where
```math
C \;=\;
\begin{bmatrix}
3 & -1 & -1 & 0 & -1 \\
-1 & 2 & -1 & 0 & 0 \\
-1 & -1 & 3 & -1 & 0 \\
0 & 0 & -1 & 1 & 0 \\
-1 & 0 & 0 & 0 & 1
\end{bmatrix}.
```

In Julia, we can formulate and solve this SDP in the following way:

```julia
using AugmentedMixing
using SparseArrays, LinearAlgebra

As = [sparse([i], [i], [1.0], 5, 5) for i in 1:5] # Vector of sparse constraint matrices
b  = ones(5) # Right-hand side vector
C = [-3.0  1    1   0   1;
      1   -2    1   0   0;
      1    1   -3   1   0;
      0    0    1  -1   0;
      1    0    0   0  -1]

# Build SDP instance that can be passed to augmented_mixing().
# The last argument needs to be set to the index of the first inequality constraint in the SDP.
# However, since our SDP does not involve any inequalities,
# we set the last argument to the number of equality constraints plus one.
sdp = SdpData(As, b, C, length(b) + 1)

# Run Augmented Mixing Method with default parameters
Xs, y, Zs = augmented_mixing(sdp)
```

You will see something like this:

```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
coefficient ranges of original SDP:
A range: [1e+00, 1e+00]
b range: [1e+00, 1e+00]
C range: [1e+00, 3e+00]

Creating the scaled SDP took 0.00 seconds

coefficient ranges after scaling:
A range: [1e+00, 1e+00]
b range: [4e-01, 4e-01]
C range: [2e-01, 5e-01]

This is a sparse SDP with 5 nonzeros in A.
Density of A: 4 %

n = 5
m = 5
equality constraints: 5
inequality constraints: 0
k = 4

Creating "data" took 0.00 seconds

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|   iter      secs|             primal                dual|    gap    pinf    dinf   compl  compl*|     mu|     evals     avg    max|  nnz(y)|  p_rat   d_rat     sum   ratio|
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|      0       0.0|-1.399420383764e+01  0.000000000000e+00|5.2e-01 3.8e-01     Inf     Inf 5.2e-01|2.2e+00|         0     0.0      0|       0|0.0e+00 0.0e+00 0.0e+00     NaN|
|      1       0.0|-2.549435515802e+01  0.000000000000e+00|6.6e-01 2.9e-01     Inf     Inf 6.6e-01|2.2e+00|       134    26.8     33|       0|6.0e-01 1.6e+00 2.2e+00 3.7e-01|
|      2       0.0|-1.773225218418e+01 -1.618219304342e+01|3.3e-02 6.6e-02     Inf     Inf 2.4e-02|2.2e+00|       214    21.4     33|       5|1.2e-01 1.3e+00 1.4e+00 9.6e-02|
|      3       0.0|-1.689541001608e+01 -1.718107066740e+01|6.1e-03 6.4e-03     Inf     Inf 3.8e-03|2.1e+00|       270    18.0     33|       5|1.3e-02 2.8e-01 2.9e-01 4.7e-02|
|      4       0.0|-1.697772317001e+01 -1.699857915460e+01|4.4e-04 5.7e-03     Inf     Inf 1.4e-04|2.0e+00|       317    15.8     33|       5|1.2e-02 2.2e-02 3.4e-02 5.7e-01|
|      5       0.0|-1.705703558048e+01 -1.694124711077e+01|2.5e-03 3.9e-03     Inf     Inf 1.3e-03|2.0e+00|       366    14.6     33|       5|6.1e-03 1.7e-02 2.4e-02 3.5e-01|
|      6       0.0|-1.701924564732e+01 -1.699265606542e+01|5.7e-04 8.2e-04     Inf     Inf 1.6e-04|1.9e+00|       410    13.7     33|       5|1.7e-03 9.3e-03 1.1e-02 1.8e-01|
|      7       0.0|-1.699594793320e+01 -1.700751645470e+01|2.5e-04 2.9e-04     Inf     Inf 1.6e-04|1.9e+00|       453    12.9     33|       5|4.8e-04 3.5e-03 4.0e-03 1.4e-01|
|      8       0.0|-1.699673524774e+01 -1.700201291789e+01|1.1e-04 1.7e-04     Inf     Inf 4.3e-05|1.8e+00|       487    12.2     33|       5|3.0e-04 6.8e-04 9.8e-04 4.4e-01|
|      9       0.0|-1.699981177741e+01 -1.699925745816e+01|1.2e-05 5.1e-05     Inf     Inf 1.6e-05|1.8e+00|       521    11.6     33|       5|1.1e-04 4.6e-04 5.6e-04 2.3e-01|
|     10       0.0|-1.700034576581e+01 -1.699953231776e+01|1.7e-05 2.0e-05     Inf     Inf 9.9e-06|1.7e+00|       552    11.0     33|       5|3.8e-05 1.5e-04 1.9e-04 2.5e-01|
|     20       0.0|-1.699999999944e+01 -1.699999997723e+01|4.7e-10 1.8e-09     Inf     Inf 4.8e-10|1.3e+00|       870     6.4      7|       5|3.6e-09 8.8e-09 1.2e-08 4.1e-01|
|     30       0.0|-1.699999999999e+01 -1.700000000000e+01|3.2e-13 6.5e-13     Inf     Inf 9.7e-14|9.8e-01|      1167     5.9      7|       5|1.0e-12 1.1e-12 2.1e-12 9.4e-01|
|     40       0.0|-1.700000000000e+01 -1.700000000000e+01|1.2e-16 1.9e-16     Inf     Inf 1.7e-18|8.2e-01|      1797    12.6    181|       5|3.8e-16 1.0e-15 1.4e-15 3.8e-01|
|     50       0.0|-1.700000000000e+01 -1.700000000000e+01|1.2e-16 3.8e-17 0.0e+00 8.1e-17 3.8e-17|7.5e-01|      2754    19.1     94|       5|1.1e-16 1.2e-16 2.3e-16 9.4e-01|
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## References
This package is based on the following preprint:

Daniel Brosch, Jan Schwiddessen, Angelika Wiegele. (2025). _The Augmented Mixing Method: Computing High-Accuracy Primal-Dual Solutions to Large-Scale SDPs via Column Updates._ [Manuscript submitted for publication].

## Citation
If you use this package in your academic work, please cite the following:
```
@misc{brosch2025augmented,
      title={The Augmented Mixing Method: Computing High-Accuracy Primal-Dual Solutions to Large-Scale SDPs via Column Updates}, 
      author={Daniel Brosch and Jan Schwiddessen and Angelika Wiegele},
      year={2025},
      eprint={2507.20386},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2507.20386}, 
}
```
