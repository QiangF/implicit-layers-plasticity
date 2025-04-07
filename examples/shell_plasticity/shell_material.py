import numpy as np
import cvxpy as cp


dim = 6  # Nxx, Nyy, Nxy, Mxx, Myy, Mxy


def gauss_quadrature_1d(n):
    """
    Generates points and weights for the Gauss quadrature rule of order n.

    Parameters:
    n (int): The order of the Gauss quadrature.

    Returns:
    points (numpy.ndarray): Quadrature points (points) in the interval [-1, 1].
    weights (numpy.ndarray): Quadrature weights.
    """
    # Use numpy to get roots (points) and weights for the Legendre polynomial
    points, weights = np.polynomial.legendre.leggauss(n)
    return points, weights


class PlaneStressvonMisesShell:
    def __init__(self, E, nu, thick, sig0, ngauss):
        self.ngauss = ngauss
        self.points, self.weights = gauss_quadrature_1d(ngauss)
        self.thick = thick
        self.sig0 = sig0
        self.S = np.array(
            [[1.0 / E, -nu / E, 0], [-nu / E, 1.0 / E, 0], [0, 0, (1 + nu) / E]]
        )
        self.set_cvxpy_model()

    def set_cvxpy_model(self):
        self.sig = cp.Variable((3, self.ngauss))
        self.Eps = cp.Parameter((dim,))
        obj = 0
        cons = []
        for i in range(self.ngauss):
            zi = self.points[i] * self.thick / 2
            wi = self.weights[i] * self.thick / 2
            epsi = self.Eps[:3] - zi * self.Eps[3:]
            sigi = self.sig[:, i]

            obj += wi * (0.5 * cp.quad_form(sigi, self.S) - sigi @ epsi)
            cons += self.yield_constraints(sigi)
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def yield_constraints(self, sig):
        Q = np.array([[1, -1 / 2, 0], [-1 / 2, 1, 0], [0, 0, 1]])
        sig_eq2 = cp.quad_form(sig, Q)
        return [sig_eq2 <= self.sig0**2]

    def solve(self, eps):
        self.Eps.value = eps
        self.prob.solve()
        N = self.sig.value @ self.weights * self.thick / 2
        M = -self.sig.value @ (self.points * self.weights * self.thick**2 / 4)
        return np.concatenate((N, M))
