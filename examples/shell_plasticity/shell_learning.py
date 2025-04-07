from pathlib import Path
import os
import yaml
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import torch
from cvxpylayers.torch import CvxpyLayer
from convex_sets import Spectrahedron
from implicit_learning import LearningModel

from utils import set_global_seed, to_np

current_file_path = Path(__file__).resolve().parent

seed = 15071988
set_global_seed(seed)


with np.load(current_file_path / "stress_strain_data.npz") as data:
    Strain = data["Strain"]
    Stress = data["Stress"]
Nincr, Npath, dim = Strain.shape

with open(current_file_path / f"ground_truth_model.yaml", "r") as file:
    ground_truth_model = yaml.safe_load(file)
E = ground_truth_model["Young_modulus"]
nu = ground_truth_model["Poisson_ratio"]
thick = ground_truth_model["thickness"]
sig0 = ground_truth_model["sig0"]
ngauss = ground_truth_model["ngauss"]

ground_truth_data = {
    "number_of_loading_directions": Npath,
    "number_of_increments": Nincr,
    "max_strain": 0.002,
}
learning_hyperparameters = {
    "learning_rate": 10e-2,
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

strain_scaling = np.max(np.abs(Strain.reshape(-1, dim)), axis=0)
Strain = Strain @ np.diag(1 / strain_scaling)
D_eps = np.diag(strain_scaling)
stress_scaling = np.max(np.abs(Stress.reshape(-1, dim)), axis=0)
Stress = Stress @ np.diag(1 / stress_scaling)
D_sig = np.diag(stress_scaling)


eps = cp.Parameter(dim)

sig = cp.Variable(dim)

S_ = np.array(
    [
        [1.0 / E, -nu / E, 0.0],
        [-nu / E, 1.0 / E, 0.0],
        [0.0, 0.0, (1.0 + nu) / E],
    ]
)
S = np.kron(
    np.diag([1 / thick, 12 / thick**3]),
    S_,
)
S_norm = D_sig @ S @ D_sig
obj = cp.Minimize(0.5 * cp.quad_form(sig, S_norm) - sig @ D_sig @ D_eps @ eps)


m = 6
surface = Spectrahedron(dim, m)


cons = [] + surface.constraints(sig)

prob = cp.Problem(obj, cons)

potential_params = []
state_params = [eps]

primal_var = [sig]
aux_var = []


parameters = surface.params + potential_params + state_params
variables = primal_var + aux_var + surface.vars
training_layer = CvxpyLayer(prob, parameters=parameters, variables=variables)

theta_hat = surface.initialize_params()


def plot_results(Sig, Sig_hat, label="Training"):
    npath = Sig.shape[1]
    ampl = 1.5

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    for j in range(npath):
        plt.plot(
            Sig[:, j, 0],
            Sig[:, j, 1],
            "-x",
            color="darkblue",
            markersize=5,
            alpha=0.5,
            label=f"{label} data" if j == 0 else None,
        )
        plt.plot(
            Sig_hat[:, j, 0],
            Sig_hat[:, j, 1],
            "-x",
            color="crimson",
            markersize=5,
            alpha=0.5,
            label="Predictions" if j == 0 else None,
        )
    plt.gca().set_aspect("equal")
    plt.xlabel("$N_{11}$")
    plt.ylabel("$N_{22}$")
    plt.xlim(
        -ampl * sig0 * thick,
        ampl * sig0 * thick,
    )
    plt.ylim(
        -ampl * sig0 * thick,
        ampl * sig0 * thick,
    )
    plt.gca().set_aspect("equal")

    plt.subplot(1, 3, 2)
    for j in range(npath):
        plt.plot(
            Sig[:, j, 3],
            Sig[:, j, 4],
            "-x",
            color="darkblue",
            markersize=5,
            alpha=0.5,
            label=f"{label} data" if j == 0 else None,
        )
        plt.plot(
            Sig_hat[:, j, 3],
            Sig_hat[:, j, 4],
            "-x",
            color="crimson",
            markersize=5,
            alpha=0.5,
            label="Predictions" if j == 0 else None,
        )
    plt.xlabel("$M_{11}$")
    plt.ylabel("$M_{22}$")
    plt.xlim(
        -ampl * sig0 * thick**2 / 4,
        ampl * sig0 * thick**2 / 4,
    )
    plt.ylim(
        -ampl * sig0 * thick**2 / 4,
        ampl * sig0 * thick**2 / 4,
    )
    plt.gca().set_aspect("equal")

    plt.subplot(1, 3, 3)
    for j in range(npath):
        plt.plot(
            Sig[:, j, 0],
            Sig[:, j, 3],
            "-x",
            color="darkblue",
            markersize=5,
            alpha=0.5,
            label=f"{label} data" if j == 0 else None,
        )
        plt.plot(
            Sig_hat[:, j, 0],
            Sig_hat[:, j, 3],
            "-x",
            color="crimson",
            markersize=5,
            alpha=0.5,
            label="Predictions" if j == 0 else None,
        )
    plt.xlabel("$N_{11}$")
    plt.ylabel("$M_{11}$")
    plt.xlim(
        -ampl * sig0 * thick,
        ampl * sig0 * thick,
    )
    plt.ylim(
        -ampl * sig0 * thick**2 / 4,
        ampl * sig0 * thick**2 / 4,
    )


def callback(epoch, theta_hat, data, prediction):
    sig_train = data["sig_train"]
    Sig = sig_train.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
    sig_hat_train, sig_hat_test = prediction
    Sig_hat = sig_hat_train.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)

    plot_results(Sig, Sig_hat)
    plt.suptitle(f"Epoch {epoch:3d}")
    fname = data["file_name"]
    out_name = f"{fname}/{epoch:03d}.png"
    if epoch in [0, 10, 20, 50, 100, 150, 199]:
        plt.savefig(out_name.replace(".png", ".pdf"))
    # plt.savefig(out_name)
    plt.close()

    if epoch == learning_hyperparameters["max_epochs"] - 1:
        sig_test = data["sig_test"]
        Sig_test = sig_test.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
        Sig_hat_test = sig_hat_test.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
        plot_results(Sig_test, Sig_hat_test, label="Validation")
        plt.savefig(f"{fname}/validation.pdf")
        plt.close()


out_fname = current_file_path / "results" / f"{surface.__class__.__name__}_{m}"
if not os.path.exists(out_fname):
    os.makedirs(out_fname)
with open(os.path.join(out_fname, f"parameters.yaml"), "w") as file:
    yaml.dump(params, file, default_flow_style=False)

model = LearningModel(training_layer, theta_hat, learning_hyperparameters, out_fname)
model.callback = callback
model.train(Strain, Stress)
model.train(Strain, Stress)

# Post-processing for validation in 2D planes

theta = []
with np.load(out_fname / "theta.npz") as data:
    for arr in data.values():
        theta.append(torch.asarray(arr, dtype=torch.float32))

from shell_material import PlaneStressvonMisesShell

material = PlaneStressvonMisesShell(E, nu, thick, sig0, ngauss)
load_dirs = {"N11-N22": (0, 1), "M11-M22": (3, 4), "N11-M11": (0, 3), "N11-M22": (0, 4)}
scaling = [1, 1, 1, 1 / thick**2, 1 / thick**2, 1 / thick**2]
for label, index in load_dirs.items():
    t = np.linspace(0, 1, 20)
    eps_max = 0.005
    angle = np.linspace(0, 2 * np.pi, 41)
    Eps_dir = np.zeros((len(angle), dim))
    Eps_dir[:, index[0]] = eps_max * np.cos(angle) * scaling[index[0]]
    Eps_dir[:, index[1]] = eps_max * np.sin(angle) * scaling[index[1]]

    Eps = np.zeros((len(t), len(angle), 6))
    Sig = np.zeros_like(Eps)
    for i, epsi in enumerate(Eps_dir):
        for j, tj in enumerate(t):
            eps = tj * epsi
            Eps[j, i, :] = eps
            Sig[j, i, :] = material.solve(eps)

    Strain = Eps @ np.diag(1 / strain_scaling)

    sig_hat = training_layer(
        *theta,
        torch.asarray(Strain.reshape((-1, dim)), dtype=torch.float32),
        solver_args={"solve_method": learning_hyperparameters["optimization_solver"]},
    )[0]
    Sig_hat = to_np(sig_hat).reshape((len(t), -1, 6)) @ np.diag(stress_scaling)

    plt.figure(figsize=(7, 6))
    truth = plt.plot(
        Sig[:, :, index[0]],
        Sig[:, :, index[1]],
        "-x",
        color="darkblue",
        markersize=5,
        alpha=0.5,
    )
    predict = plt.plot(
        Sig_hat[:, :, index[0]],
        Sig_hat[:, :, index[1]],
        "-x",
        color="crimson",
        markersize=5,
        alpha=0.5,
    )
    x0 = 1.5 * sig0 * thick
    if index[0] > 2:
        x0 *= thick / 4
    y0 = 1.5 * sig0 * thick
    if index[1] > 2:
        y0 *= thick / 4
    plt.xlabel(f"${label[0]}_{{{label[1:3]}}}$")
    plt.ylabel(f"${label[4]}_{{{label[5:7]}}}$")
    plt.legend(
        [truth[0], predict[0]],
        ["Ground truth", "Predictions"],
        ncol=2,
        loc="upper right",
    )
    plt.xlim(-x0, x0)
    plt.ylim(-y0, y0)
    plt.savefig(out_fname / (label + ".pdf"))
    plt.show()
