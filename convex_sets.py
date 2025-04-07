import cvxpy as cp
import numpy as np
from utils import to_sym_matrix, to_triu_matrix, rand_uniform


class Polyhedron:
    def __init__(self, n, m):
        """A polyhedron in R^n defined by m hyperplanes"""
        self.n = n
        self.m = m
        self.additional_vars = []
        self.vars = self.additional_vars
        self.params = [cp.Parameter((n, m))]

    def initialize_params(self):
        return [
            rand_uniform((self.n, self.m)),
        ]

    def constraints(self, x):
        A = self.params[0]
        cons = [x @ A <= 1]
        return cons


class Ellipsoid:
    def __init__(self, n):
        """An ellipsoid in R^n"""
        self.n = n
        Q = cp.Parameter(n * (n + 1) // 2)
        c_ = cp.Parameter(n)
        # here c_ denotes Q @ c with notations from the paper
        self.params = [Q, c_]
        self.vars = []

    def initialize_params(self):
        return [
            rand_uniform((self.n * (self.n + 1) // 2,)),
            rand_uniform((self.n,)),
        ]

    def constraints(self, x):
        Q_, c_ = self.params
        Q = to_triu_matrix(Q_, self.n)
        cons = [cp.norm(Q @ x - c_) <= 1]
        return cons

    def plot(self, Q, c, radius, m=200):
        """Only used for postprocessing"""
        c_ = np.linalg.inv(Q) @ c
        n = len(c)
        t = np.linspace(0, 2 * np.pi, m)
        xt = np.vstack((np.cos(t), np.sin(t))).T  # xt = (x-c_)
        lamb = np.linalg.norm(xt @ Q.T, axis=1)
        Lamb = np.repeat(lamb[:, np.newaxis], n, axis=1)
        return xt / Lamb * radius + np.repeat(c_[np.newaxis, :], m, axis=0)


class ConvexHullEllipsoids:
    def __init__(self, n, ellipsoids):
        """A convex hull of m ellipsoids in R^n"""
        self.n = n
        self.m = len(ellipsoids)
        self.ellipsoids = ellipsoids
        self.additional_vars = [
            cp.Variable(self.m, nonneg=True),
            cp.Variable((self.m, n)),
        ]
        self.vars = [v for dom in ellipsoids for v in dom.vars] + self.additional_vars
        self.params = [p for dom in ellipsoids for p in dom.params]

    def initialize_params(self):
        return [p for dom in self.ellipsoids for p in dom.initialize_params()]

    def constraints(self, x):
        cons = []
        t, X = self.additional_vars
        for i, dom in enumerate(self.ellipsoids):
            Q_, c_ = dom.params
            Q = to_triu_matrix(Q_, dom.n)
            cons += [cp.norm(Q @ X[i] - c_ * t[i]) <= t[i]]
        cons += [cp.sum(t) == 1, x == cp.sum(X, axis=0)]

        return cons

    def plot(self, Q_, c, radius):
        """Only used for postprocessing"""
        out = []
        for i, dom in enumerate(self.ellipsoids):
            Q = to_triu_matrix(Q_[i], dom.nx).value
            x = dom.plot(Q, c[i], radius)
            out.append(x)
        return out


class Spectrahedron:
    """A m-spectrahedron in R^n"""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.additional_vars = []
        self.vars = self.additional_vars
        self.params = [cp.Parameter((n, m * (m + 1) // 2))]

    def initialize_params(self):
        return [
            rand_uniform((self.n, self.m * (self.m + 1) // 2)),
        ]

    def constraints(self, x):
        A = self.params[0]
        X = to_sym_matrix(x @ A, self.m)
        cons = [np.eye(self.m) + X >> 0]
        return cons
