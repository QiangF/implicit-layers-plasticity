from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from utils import set_global_seed
from shell_material import PlaneStressvonMisesShell

seed = 15071988
set_global_seed(seed)

# Current file's path
current_file_path = Path(__file__).resolve().parent

dim = 6  # Nxx, Nyy, Nxy, Mxx, Myy, Mxy


E, nu = 100.0e3, 0.3
thick = 0.25
sig0 = 100.0
ngauss = 7
material = PlaneStressvonMisesShell(E, nu, thick, sig0, ngauss)

Npath = 100
Nincr = 10

ground_truth_model = {
    "yield_surface": material.__class__.__name__,
    "YoungModulus": E,
    "Poisson_ratio": nu,
    "thickness": thick,
    "sig0": sig0,
    "ngauss": ngauss,
}
ground_truth_data = {
    "number_of_loading_directions": Npath,
    "number_of_increments": Nincr,
    "max_strain": 0.002,
}

with open(current_file_path / f"ground_truth_model.yaml", "w") as file:
    yaml.dump(ground_truth_model, file, default_flow_style=False)


t = np.linspace(0, ground_truth_data["max_strain"], Nincr)
scaling = np.diag([1, 1, 1, 1 / thick**2, 1 / thick**2, 1 / thick**2])
u = np.random.normal(
    0, 1, (Npath, dim)
)  # an array of d normally distributed random variables
Eps_dir = u @ scaling

Eps = np.zeros((Nincr, Npath, dim))
for i, epsi in enumerate(Eps_dir):
    for j, tj in enumerate(t):
        Eps[j, i, :] = tj * epsi

Sig = np.zeros_like(Eps)
for i in range(Npath):
    for j in range(Nincr):
        eps = Eps[j, i, :]
        Sig[j, i, :] = material.solve(eps)

np.savez(current_file_path / "stress_strain_data.npz", Strain=Eps, Stress=Sig)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(Npath):
    plt.plot(Sig[:, i, 0], Sig[:, i, 1], "-x")
plt.gca().set_aspect("equal")
plt.xlabel("$N_{11}$")
plt.ylabel("$N_{22}$")
plt.subplot(1, 2, 2)
for i in range(Npath):
    plt.plot(Sig[:, i, 3], Sig[:, i, 4], "-x")
plt.xlabel("$M_{11}$")
plt.ylabel("$M_{22}$")
plt.show()
