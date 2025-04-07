import numpy as np
import cvxpy as cp
from utils import vmap


class PlaneStressHosford:
    def __init__(self, E, nu, sig0, a):
        G = E / 2 / (1 + nu)
        S = np.array(
            [
                [1.0 / E, -nu / E, 0.0],
                [-nu / E, 1.0 / E, 0.0],
                [0.0, 0.0, 1.0 / 2.0 / G],
            ]
        )
        self.C = np.linalg.inv(S)
        self.C_sqrt = np.linalg.cholesky(self.C).T
        self.sig0 = sig0
        self.a = a
        self.set_cvxpy_model()

    def yield_constraints(self, sig, R):
        Sig = cp.bmat(
            [
                [sig[0], sig[2] / np.sqrt(2)],
                [sig[2] / np.sqrt(2), sig[1]],
            ]
        )
        z = cp.Variable(3)
        return [
            cp.trace(Sig) == z[0] - z[1],
            cp.lambda_max(Sig) - cp.lambda_min(Sig) <= z[2],
            z[2] == z[0] + z[1],
            cp.norm(z, p=self.a) <= 2 ** (1 / self.a) * R,
        ]

    def set_cvxpy_model(self):
        self.sig = cp.Variable((3,))
        self.sig_el = cp.Parameter((3,))
        self.eps = cp.Parameter((3,))
        self.p_old = cp.Parameter()
        self.sig0p = cp.Parameter()

        obj = (
            0.5 * cp.quad_form(self.sig - 0 * self.sig_el, np.linalg.inv(self.C))
            - self.sig @ self.eps
        )
        self.prob = cp.Problem(
            cp.Minimize(obj), self.yield_constraints(self.sig, self.sig0p)
        )

    def constitutive_update(self, eps, state):
        eps_old = state["Strain"]
        deps = eps - eps_old
        sig_old = state["Stress"]

        self.sig_el.value = sig_old + self.C @ deps

        self.eps.value = eps

        self.sig0p.value = self.sig0

        self.prob.solve()

        state["Strain"] = eps
        state["Stress"] = self.sig.value
        return state

    def integrate(self, Eps, State):
        return vmap(self.constitutive_update)(Eps, State)

    def plot_yield_surface(self, n=100):
        """For postprocessing only"""
        sig = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, n)])
        sig_eq = (
            (
                np.abs(sig[:, 0]) ** self.a
                + np.abs(sig[:, 1]) ** self.a
                + np.abs(sig[:, 0] - sig[:, 1]) ** self.a
            )
            / 2
        ) ** (1 / self.a)
        return sig * self.sig0 / np.repeat(sig_eq[:, np.newaxis], 2, axis=1)
