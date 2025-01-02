[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2410.08060.svg)](https://doi.org/10.48550/arXiv.2410.08060)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Optimal Transportation by Orthogonal Coupling Dynamics

In this repository, we present an implementation of orthogonal coupling dynamics to solve the optimal transport problem. This git repository has been used to produce results in the following paper:

Mohsen Sadr, Peyman Mohajerin Esfehani, Hossein Gorji. "Optimal Transportation by Orthogonal Coupling Dynamics." 2024, preprint at [arXiv:2410.08060](https://doi.org/10.48550/arXiv.2410.08060).

## Usage

Here, we provide the most accessible (and not fastest) implementation of the OCD in NumPy. Simply, first import the library via

```
import sys
import os
src_path = os.path.abspath(os.path.join(os.getcwd(), '[path/to/src]'))
sys.path.append(src_path)
from OCD import *
```

Given samples of two marginals X, Y, you can find the optimal pairing by first finding the optimal regularization parameter $epsilon$. Then, call the OCD solver via

```
eps0 = find_opt_eps2(X0, Y0, log_eps_range=[-3,0], nepss = 400, perc=0.9998)
print("epsilon = ", eps0)

dt = 0.1
Nt = 200
tol = 1e-6

X_ocd, Y_ocd, dists, err_m2X, err_m2Y = ocd_map_RK4(X0, Y0, dt=dt, Nt=200, sigma=eps0, tol=tol)
```

For examples of how this implementation can be used, see the Jupyter Notebooks in ```examples/``` directory.
