from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxpy as cp
from utils import set_global_seed

# Current file's path
current_file_path = Path(__file__).resolve().parent


seed = 18052019
set_global_seed(seed)

dim = 2

x = np.linspace(-2, 2, 100)
y1 = 0.25 * np.abs(x - 1) ** 2 - 1.25
y2 = -0.5 * (x + 0.5) ** 2 + 1.25
idx = np.where(y1 <= y2)
y = np.concatenate((y1[idx], y2[idx][::-1], y1[idx][[0]]))
x = np.concatenate((x[idx], x[idx][::-1], x[idx][[0]]))

N = len(x)
points = np.zeros((N, dim))
points[:, 0] = x
points[:, 1] = y

hull = ConvexHull(points)
A = hull.equations[:, :-1]  # Coefficients of x and y
b = -hull.equations[:, -1]  # Right-hand side values


# Define optimization variable
z = cp.Variable((dim,))
z0 = cp.Parameter((dim,))
# Define objective: minimize squared distance to x0
objective = cp.Minimize(cp.norm2(z - z0))

# Constraints: x must satisfy Ax ≤ b
constraints = [A @ z <= b]

# Solve the quadratic program
prob = cp.Problem(objective, constraints)

N_data = 100
theta = np.random.rand(N_data) * 2 * np.pi

Eps = 2.5 * np.vstack((np.cos(theta), np.sin(theta))).T
Sig = np.zeros_like(Eps)
# print(Eps.shape)

for i, eps in enumerate(Eps):
    z0.value = eps
    prob.solve()
    Sig[i, :] = z.value

np.savez(current_file_path / "2d_example_data.npz", x=x, Eps=Eps, y=y, Sig=Sig)


plt.figure()
plt.plot(x, y, "-", color="darkblue", linewidth=2, label="Ground truth set")
plt.plot(
    Eps[:, 0],
    Eps[:, 1],
    "o",
    color="crimson",
    markerfacecolor="white",
    alpha=0.5,
    label=r"Inputs $\boldsymbol{x}$",
)
plt.plot(
    Sig[:, 0],
    Sig[:, 1],
    "o",
    color="darkblue",
    markerfacecolor="white",
    alpha=1.0,
    label=r"Outputs $\boldsymbol{y}$",
)
plt.quiver(
    Eps[:, 0],
    Eps[:, 1],
    Sig[:, 0] - Eps[:, 0],
    Sig[:, 1] - Eps[:, 1],
    scale_units="xy",
    angles="xy",
    scale=1,
    color="crimson",
    alpha=0.5,
)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend(ncol=2, bbox_to_anchor=(1.1, 1.15))
plt.gca().set_aspect("equal")
plt.show()


noise_level = 0.1
Sig += np.random.normal(scale=noise_level, size=Sig.shape)

plt.plot(x, y, "-", color="darkblue", linewidth=2, label="Ground truth data")
plt.plot(
    Sig[:, 0], Sig[:, 1], "o", color="darkblue", markerfacecolor="white", alpha=0.5
)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect("equal")
plt.show()
