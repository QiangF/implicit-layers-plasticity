import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils import set_global_seed
from convex_sets import (
    Ellipsoid,
    ConvexHullEllipsoids,
    Polyhedron,
    Spectrahedron,
)
from implicit_learning import LearningModel
import yaml
from scipy.spatial import ConvexHull

current_file_path = Path(__file__).resolve().parent

seed = 18052019
set_global_seed(seed)

dim = 2

with np.load(current_file_path / "2d_example_data.npz") as data:
    x = data["x"]
    y = data["y"]
    Eps = data["Eps"]  # inputs
    Sig = data["Sig"]  # outputs
N_data = len(Eps)


noise_level = 0.0
Sig += np.random.normal(scale=noise_level, size=Sig.shape)


eps = cp.Parameter(dim)
sig = cp.Variable(dim)

obj = cp.Minimize(0.5 * cp.sum_squares(sig - eps))

# m = 50
# surface = Polyhedron(dim, m)

# m = 8
# ellipsoids = [Ellipsoid(dim) for i in range(m)]
# surface = ConvexHullEllipsoids(dim, ellipsoids)

m = 4
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
    sig_hat_train, sig_hat_test = prediction
    Sig_hat = sig_hat_train
    Sig_hat_test = sig_hat_test
    train_id = data["train_id"]
    test_id = data["test_id"]
    Sig_hat_full = np.zeros_like(Sig)
    Sig_hat_full[train_id, :] = Sig_hat
    Sig_hat_full[test_id, :] = Sig_hat_test

    plt.figure()
    plt.plot(x, y, "-", color="darkblue", linewidth=2)
    plt.plot(
        sig_train[:, 0],
        sig_train[:, 1],
        "o",
        color="darkblue",
        markerfacecolor="white",
        alpha=0.5,
        label="Training data",
    )
    plt.plot(
        sig_hat_train[:, 0],
        sig_hat_train[:, 1],
        "oC3",
        alpha=0.5,
        markerfacecolor="white",
        label="Predictions",
    )

    hull = ConvexHull(Sig_hat_full)
    for simplex in hull.simplices:
        plt.plot(
            Sig_hat_full[simplex, 0],
            Sig_hat_full[simplex, 1],
            "-C3",
        )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend(ncol=2, fontsize=14)
    plt.gca().set_aspect("equal")
    fname = data["file_name"]
    # plt.savefig(f"{fname}/{epoch}.png")
    if epoch in [0, 10, 20, 50, 100, 150, 199]:
        plt.savefig(f"{fname}/{epoch}.pdf")
    plt.close()


fname = (
    current_file_path / f"results/{surface.__class__.__name__}_{m}_noise_{noise_level}"
)
learning_hyperparameters = {
    "learning_rate": 0.1,
    "optimization_solver": "SCS",
    "test_ratio": 0.5,
    "max_epochs": 200,
}
training_data = {
    "surface": surface.__class__.__name__,
    "m": m,
    "data": N_data,
    "noise_level": noise_level,
}
data = {
    "training_data": training_data,
    "learning_hyperparameters": learning_hyperparameters,
}
if not os.path.exists(fname):
    os.makedirs(fname)
with open(f"{fname}/parameters.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False)

model = LearningModel(
    training_layer, theta_hat, learning_hyperparameters, fname, callback_frequency=2
)
model.callback = callback
model.train(Eps[np.newaxis, :, :], Sig[np.newaxis, :, :], split_axis=1)
