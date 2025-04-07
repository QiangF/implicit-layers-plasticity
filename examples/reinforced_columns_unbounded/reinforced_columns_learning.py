import os
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from convex_sets import Spectrahedron
from implicit_learning import LearningModel
from utils import set_global_seed


current_file_path = Path(__file__).resolve().parent

seed = 15071988
set_global_seed(seed)

with np.load(current_file_path / "reinforced_columns_data.npz") as data:
    Strain = data["Strain"]
    Stress = data["Stress"]


Nincr, Npath, dim = Stress.shape

strain_scaling = np.max(np.abs(Strain.reshape(-1, dim)), axis=0)
strain_scaling = np.ones((dim,))
Strain = Strain @ np.diag(1 / strain_scaling)
D_eps = np.diag(strain_scaling)
stress_scaling = np.max(np.abs(Stress.reshape(-1, dim)), axis=0)
stress_scaling = np.ones((dim,))
D_sig = np.diag(stress_scaling)
Stress = Stress @ np.diag(1 / stress_scaling)

eps = cp.Parameter(dim)

sig = cp.Variable(dim)

obj = cp.Minimize(cp.sum_squares(D_sig @ sig - D_eps @ eps))

m = 6
surface = Spectrahedron(dim, m)

cons = surface.constraints(D_sig @ sig)

prob = cp.Problem(obj, cons)

potential_params = []
state_params = [eps]

primal_var = [sig]
aux_var = []


parameters = surface.params + potential_params + state_params
variables = primal_var + aux_var + surface.vars
training_layer = CvxpyLayer(prob, parameters=parameters, variables=variables)


theta_hat = surface.initialize_params()


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
    "ground_truth_data": ground_truth_data,
    "learning_hyperparameters": learning_hyperparameters,
}

out_fname = current_file_path / "results" / f"{surface.__class__.__name__}_{m}"
if not os.path.exists(out_fname):
    os.makedirs(out_fname)
with open(os.path.join(out_fname, f"parameters.yaml"), "w") as file:
    yaml.dump(params, file, default_flow_style=False)


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

    plt.figure(figsize=(6, 7.5))

    cmap_blues = plt.get_cmap("Blues_r")
    cmap_reds = plt.get_cmap("Reds_r")
    plt.scatter(
        Sig[0, :, 0],
        Sig[0, :, 1],
        marker="x",
        color=cmap_blues(abs(Sig[0, :, 2])),
        linewidth=1.0,
    )
    plt.scatter(
        Sig_hat[0, :, 0],
        Sig_hat[0, :, 1],
        marker="x",
        color=cmap_reds(abs(Sig_hat[0, :, 2])),
        linewidth=1.0,
    )

    plt.xlabel(r"Stress $\sigma_{xx}$ [MPa]")
    plt.ylabel(r"Stress $\sigma_{yy}$ [MPa]")
    plt.title(f"Epoch {epoch:3d}")
    plt.xlim(-3, 2)
    plt.ylim(-3, 2)
    plt.gca().set_aspect("equal")
    fname = data["file_name"]
    out_name = f"{fname}/{epoch:03d}.png"
    if epoch in [0, 10, 20, 50, 100, 150, 199]:
        plt.savefig(out_name.replace(".png", ".pdf"))
    # plt.savefig(out_name)
    plt.close()


model = LearningModel(
    training_layer, theta_hat, learning_hyperparameters, out_fname, callback_frequency=2
)
model.callback = callback
model.train(Strain, Stress)
