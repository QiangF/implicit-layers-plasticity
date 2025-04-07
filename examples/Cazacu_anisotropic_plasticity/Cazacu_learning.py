import os
import yaml
from pathlib import Path
import numpy as np
from utils import set_global_seed
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from convex_sets import Spectrahedron
from implicit_learning import LearningModel

current_file_path = Path(__file__).resolve().parent

seed = 15071988
set_global_seed(seed)

# Ground truth parameters
E = 150e3
nu = 0.3
sig0 = 190.0  # postprocessing only
ground_truth_model = {
    "yield_surface": "Cazacu 2001",
    "YoungModulus": E,
    "Poisson_ratio": nu,
    "sig0": sig0,
}
# elastic plane stress compliance
G = E / 2 / (1 + nu)
S = np.array(
    [
        [1.0 / E, -nu / E, 0.0],
        [-nu / E, 1.0 / E, 0.0],
        [0.0, 0.0, 1.0 / 2.0 / G],
    ]
)

with np.load(current_file_path / "Cazacu_stress_strain_data.npz") as data:
    Strain = data["Strain"][[-1], :, :]
    Stress = data["Stress"][[-1], :, :]


Nincr, Npath, dim = Strain.shape

strain_scaling = np.max(np.abs(Strain.reshape(-1, dim)), axis=0)
Strain = Strain @ np.diag(1 / strain_scaling)
D_eps = np.diag(strain_scaling)
stress_scaling = np.max(np.abs(Stress.reshape(-1, dim)), axis=0)
D_sig = np.diag(stress_scaling)
Stress = Stress @ np.diag(1 / stress_scaling)

eps = cp.Parameter(dim)

sig = cp.Variable(dim)

S_norm = D_sig @ S @ D_sig
obj = cp.Minimize(0.5 * cp.quad_form(sig, S_norm) - sig @ D_sig @ D_eps @ eps)

m = 6
surface = Spectrahedron(dim, m)

cons = surface.constraints(sig)

prob = cp.Problem(obj, cons)

potential_params = []
state_params = [eps]

primal_var = [sig]
aux_var = []


parameters = surface.params + potential_params + state_params
variables = primal_var + aux_var + surface.vars
training_layer = CvxpyLayer(prob, parameters=parameters, variables=variables)


theta_hat = surface.initialize_params()


def callback(epoch, theta_hat, data, prediction):
    sig_train = data["sig_train"]
    Sig = sig_train.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
    sig_hat_train, sig_hat_test = prediction
    Sig_hat = sig_hat_train.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
    Sig_hat_test = sig_hat_test.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
    Sig_hat_full = np.zeros((Nincr, Npath, dim))
    train_id = data["train_id"]
    test_id = data["test_id"]
    Sig_hat_full[:, train_id, :] = Sig_hat
    Sig_hat_full[:, test_id, :] = Sig_hat_test

    npath = Sig.shape[1]

    plt.figure(figsize=(6, 7.5))

    Nshear = 6
    Nrad = Npath // Nshear
    cmap_blues = plt.get_cmap("Blues_r")
    blues = cmap_blues(np.linspace(0, 1, int(1.5 * Nshear)))
    cmap_reds = plt.get_cmap("Reds_r")
    reds = cmap_reds(np.linspace(0, 1, int(1.5 * Nshear)))
    for p in range(Nshear):
        plt.plot(
            Stress[-1, Nrad * p : Nrad * (p + 1), 0] * stress_scaling[0],
            Stress[-1, Nrad * p : Nrad * (p + 1), 1] * stress_scaling[1],
            "-",
            color=blues[p],
        )
        plt.plot(
            Sig_hat_full[-1, Nrad * p : Nrad * (p + 1), 0],
            Sig_hat_full[-1, Nrad * p : Nrad * (p + 1), 1],
            "-",
            color=reds[p],
        )

    for j in range(npath):
        plt.plot(
            Sig[:, j, 0],
            Sig[:, j, 1],
            "o",
            color="darkblue",
            markerfacecolor="white",
            alpha=0.5,
            label="Training data" if j == 0 else None,
        )
        plt.plot(
            Sig_hat[:, j, 0],
            Sig_hat[:, j, 1],
            "oC3",
            alpha=0.5,
            markerfacecolor="white",
            label="Predictions" if j == 0 else None,
        )

    plt.xlabel(r"Stress $\sigma_{xx}$ [MPa]")
    plt.ylabel(r"Stress $\sigma_{yy}$ [MPa]")
    plt.title(f"Epoch {epoch:3d}")
    plt.legend(loc="upper center", ncol=2, fontsize=14, bbox_to_anchor=(0.5, 1.15))
    ampl = 1.5
    plt.xlim(-ampl * sig0, ampl * sig0)
    plt.ylim(-ampl * sig0, ampl * sig0)
    plt.gca().set_aspect("equal")
    fname = data["file_name"]
    out_name = f"{fname}/{epoch:03d}.png"
    if epoch in [0, 10, 20, 50, 100, 150, 199]:
        plt.savefig(out_name.replace(".png", ".pdf"))
    # plt.savefig(out_name)
    plt.close()


ground_truth_data = {
    "number_of_loading_directions": Npath,
    "number_of_increments": Nincr,
}
learning_hyperparameters = {
    "m": m,
    "learning_rate": 5e-2,
    "optimization_solver": "SCS",
    "test_ratio": 0.5,
    "max_epochs": 200,
}
params = {
    "seed": seed,
    "ground_truth_model": ground_truth_model,
    "ground_truth_data": ground_truth_data,
    "learning_hyperparameters": learning_hyperparameters,
}

out_fname = current_file_path / "results" / f"{surface.__class__.__name__}_{m}"

if not os.path.exists(out_fname):
    os.makedirs(out_fname)
with open(out_fname / f"parameters.yaml", "w") as file:
    yaml.dump(params, file, default_flow_style=False)

model = LearningModel(
    training_layer, theta_hat, learning_hyperparameters, out_fname, callback_frequency=2
)
model.callback = callback
model.train(Strain, Stress)
