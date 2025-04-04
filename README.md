# Learning elastoplasticity with implicit layers

These scripts are supplementary materials to the paper:

> Learning elastoplasticity with implicit layers, by Jérémy Bleyer

## Requirements

The scripts require `pytorch` as well as `cvxpy` and `cvxpylayers` libraries available at: 

- https://www.cvxpy.org/

- [GitHub - cvxgrp/cvxpylayers: Differentiable convex optimization layers](https://github.com/cvxgrp/cvxpylayers)

## Contents

- `implicit_learning.py` implements the implicit learning architecture

- `convex_sets.py` implements different convex set parametrization including `Polyhedron, Ellipsoids, ConvexHullEllipsoids` and `Spectrahedron`

- `utils.py` contains various utility functions

- The `examples` folder contains the scripts corresponding to each folders of the paper, including data sets or scripts for generating such data sets


