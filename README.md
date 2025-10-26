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
