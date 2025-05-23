import os
import yaml
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from convex_sets import Spectrahedron, Polyhedron
from implicit_learning import LearningModel

from utils import set_global_seed

current_file_path = Path(__file__).resolve().parent

seed = 15071988
set_global_seed(seed)


with np.load(current_file_path / "stress_strain_data.npz") as data:
    Strain = data["Strain"]
    Stress = data["Stress"]
    yield_surface = data["yield_surface"]  # exact yield surface for plotting
Nincr, Npath, dim = Strain.shape
eps_max = np.max(np.abs(Strain))

with open(current_file_path / f"ground_truth_model.yaml", "r") as file:
    ground_truth_model = yaml.safe_load(file)
sig0 = ground_truth_model["sig0"]

ground_truth_data = {
    "number_of_loading_directions": Npath,
    "number_of_increments": Nincr,
    "max_strain": float(eps_max),
}
learning_hyperparameters = {
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

noise_level = 0.0
Stress += np.random.normal(scale=noise_level * sig0, size=Stress.shape)
ground_truth_data["noise_level"] = noise_level

Nincr = Stress.shape[0]

strain_scaling = np.max(np.abs(Strain.reshape(-1, dim)), axis=0)
Deps = np.diag(strain_scaling)
Strain = Strain @ np.diag(1 / strain_scaling)
stress_scaling = np.max(np.abs(Stress.reshape(-1, dim)), axis=0)
Dsig = np.diag(stress_scaling)
Stress = Stress @ np.diag(1 / stress_scaling)

Psig = cp.Parameter((dim, dim))

deps = cp.Parameter(dim)
sig_old = cp.Parameter(dim)
sig = cp.Variable(dim)
dsig_tilde = cp.Variable(dim)

obj = cp.Minimize(0.5 * cp.sum_squares(dsig_tilde) - sig @ deps)

# m = 20
# surface = Polyhedron(dim, m)

m = 6
surface = Spectrahedron(dim, m)

cons = [
    Psig @ dsig_tilde == sig - sig_old,
] + surface.constraints(sig)

prob = cp.Problem(obj, cons)

potential_params = [Psig]
state_params = [deps, sig_old]

primal_var = [sig]
aux_var = [dsig_tilde]


parameters = surface.params + potential_params + state_params
variables = primal_var + aux_var + surface.vars


training_layer = CvxpyLayer(prob, parameters=parameters, variables=variables)


theta_hat = surface.initialize_params() + [
    torch.asarray(
        np.eye(dim),
        requires_grad=True,
        dtype=torch.float32,
    ),
]


def plot_results(yield_surface, xx, Sig, Sig_hat, label="Training"):
    npath = Sig.shape[1]

    plt.figure(figsize=(6, 7.5))
    plt.plot(
        yield_surface[:, 0],
        yield_surface[:, 1],
        "--k",
        linewidth=1.5,
        label="Exact yield surface",
        alpha=0.5,
    )
    hull = ConvexHull(xx)
    hull_vertices = np.append(hull.vertices, hull.vertices[0])
    plt.fill(
        xx[hull_vertices, 0],
        xx[hull_vertices, 1],
        color="crimson",
        alpha=0.25,
        label="Predicted yield surface",
    )
    for j in range(npath):
        plt.plot(
            Sig[:, j, 0],
            Sig[:, j, 1],
            "-x",
            color="darkblue",
            alpha=0.5,
            markersize=5,
            label=f"{label} data" if j == 0 else None,
        )
        plt.plot(
            Sig_hat[:, j, 0],
            Sig_hat[:, j, 1],
            "-x",
            color="crimson",
            alpha=0.5,
            markersize=5,
            label="Predictions" if j == 0 else None,
        )

    plt.xlabel(r"Stress $\sigma_{xx}$ [MPa]")
    plt.ylabel(r"Stress $\sigma_{yy}$ [MPa]")
    plt.legend(loc="upper center", ncol=2, fontsize=14, bbox_to_anchor=(0.5, 1.18))
    ampl = 1.5
    plt.xlim(-ampl * sig0, ampl * sig0)
    plt.ylim(-ampl * sig0, ampl * sig0)
    plt.gca().set_aspect("equal")


from utils import to_np


def callback(epoch, theta_hat, data, prediction):
    Psig = np.sqrt(Dsig @ np.linalg.inv(Deps)) @ to_np(theta_hat[-1])
    print("Elastic stiffness:", Psig.T @ Psig)

    sig_train = data["sig_train"]
    Sig = sig_train.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
    sig_hat_train, sig_hat_test = prediction
    Sig_hat = sig_hat_train.reshape((Nincr, -1, dim)) @ np.diag(stress_scaling)
    xx = np.concatenate((sig_hat_test, sig_hat_train)) @ np.diag(stress_scaling)

    plot_results(yield_surface, xx, Sig, Sig_hat)
    # plt.title(f"Epoch {epoch:3d}")
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
        plot_results(yield_surface, xx, Sig_test, Sig_hat_test, label="Validation")
        plt.savefig(f"{fname}/validation.pdf")
        plt.close()


out_fname = (
    current_file_path
    / "results"
    / f"{surface.__class__.__name__}_{m}_noise_{noise_level}"
)
if not os.path.exists(out_fname):
    os.makedirs(out_fname)
with open(os.path.join(out_fname, f"parameters.yaml"), "w") as file:
    yaml.dump(params, file, default_flow_style=False)


model = LearningModel(
    training_layer, theta_hat, learning_hyperparameters, out_fname, callback_frequency=2
)
model.callback = callback


dStrain = np.diff(Strain, axis=0, prepend=0)
Stress_old = np.insert(Stress, 0, 0.0, axis=0)[:-1, :, :]
model.train(dStrain, Stress, additional_vars=[Stress_old])
