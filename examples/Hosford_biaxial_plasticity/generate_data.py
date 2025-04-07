from pathlib import Path
import os
import yaml
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from Hosford_material import PlaneStressHosford
from utils import set_global_seed


seed = 15071988
set_global_seed(seed)

# Current file's path
current_file_path = Path(__file__).resolve().parent


E, nu = 70.0e3, 0.3
S = np.array(
    [
        [1.0 / E, -nu / E],
        [-nu / E, 1.0 / E],
    ]
)

sig0 = 350.0
H = 0.0
a = 20
material = PlaneStressHosford(E, nu, sig0, a)


ground_truth_model = {
    "yield_surface": material.__class__.__name__,
    "YoungModulus": E,
    "Poisson_ratio": nu,
    "sig0": sig0,
    "a": a,
}

eps = 2e-2
Npath = 40
Nincr = 10

theta = np.linspace(0, 2 * np.pi, Npath + 1)[:-1]
Eps = np.vstack([np.array([eps * np.cos(t), eps * np.sin(t), 0]) for t in theta])


state = {
    "Strain": np.zeros((Npath, 3)),
    "Stress": np.zeros((Npath, 3)),
}


t_list = np.linspace(0, 1.0, Nincr)
Strain = np.zeros((Nincr, Npath, 3))
Stress = np.zeros((Nincr, Npath, 3))
fig, ax = plt.subplots()
for i, t in enumerate(t_list[1:]):
    state = material.integrate(t * Eps, state)
    Stress[i + 1, :, :] = state["Stress"]
    Strain[i + 1, :, :] = state["Strain"]

s_data = t_list
min_s = np.min(s_data)
max_s = np.max(s_data)
norm = Normalize(vmin=min_s, vmax=max_s)
cmap = plt.get_cmap("plasma")

for j in range(Npath):
    points = Stress[:, [j], :2]
    segments = np.concatenate([points[:-1, :], points[1:, :]], axis=1)

    lc = LineCollection(segments, cmap=cmap, linewidths=3, norm=norm, alpha=0.75)
    lc.set_array(s_data)
    ax.add_collection(lc)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # Normalizing to [0, 1] range
sm.set_array([])  # Required for colorbar
plt.colorbar(sm, ax=ax, label="Imposed strain")
plt.xlabel(r"Stress $\sigma_{xx}$ [MPa]")
plt.ylabel(r"Stress $\sigma_{yy}$ [MPa]")
ampl = 1.5
plt.xlim(-ampl * sig0, ampl * sig0)
plt.ylim(-ampl * sig0, ampl * sig0)
plt.gca().set_aspect("equal")
plt.savefig(current_file_path / f"stress_strain_data.pdf")
plt.show()

# For postprocessing only
sig = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 100)])
sig_eq = (
    (
        np.abs(sig[:, 0]) ** material.a
        + np.abs(sig[:, 1]) ** material.a
        + np.abs(sig[:, 0] - sig[:, 1]) ** material.a
    )
    / 2
) ** (1 / material.a)
yield_surface = sig * sig0 / np.repeat(sig_eq[:, np.newaxis], 2, axis=1)

np.savez(
    current_file_path / "stress_strain_data.npz",
    Strain=Strain[:, :, :2],
    Stress=Stress[:, :, :2],
    yield_surface=yield_surface,
)

with open(current_file_path / f"ground_truth_model.yaml", "w") as file:
    yaml.dump(ground_truth_model, file, default_flow_style=False)
