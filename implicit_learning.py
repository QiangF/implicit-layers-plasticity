import torch
from torch import optim
import numpy as np
from utils import to_np, split_test_train


class LearningModel:
    def __init__(
        self,
        training_layer,
        theta_hat,
        learning_hyperparameters,
        file_name,
        callback_frequency=5,
    ):
        self.theta_hat = theta_hat
        self.training_layer = training_layer
        self.hyperparams = learning_hyperparameters
        self.file_name = file_name
        self.opt = optim.Adam(theta_hat, lr=self.hyperparams["learning_rate"])
        self.losses = []
        self.callback_frequency = callback_frequency

    def callback(self, *args):
        """Callback function the user can use to do postprocessing during training"""
        pass

    def loss_function(self, output, target):
        return torch.nn.MSELoss()(output, target)

    def train(self, Strain, Stress, additional_vars=None, split_axis=1):
        self.dim = Strain.shape[-1]
        (
            (Strain_train, Strain_test),
            (Stress_train, Stress_test),
            (train_id, test_id),
        ) = split_test_train(
            Strain, Stress, axis=split_axis, test_ratio=self.hyperparams["test_ratio"]
        )

        eps_train = torch.asarray(
            Strain_train.reshape((-1, self.dim)), dtype=torch.float32
        )
        eps_test = torch.asarray(
            Strain_test.reshape((-1, self.dim)), dtype=torch.float32
        )
        sig_train = torch.asarray(
            Stress_train.reshape((-1, self.dim)), dtype=torch.float32
        )
        sig_test = torch.asarray(
            Stress_test.reshape((-1, self.dim)), dtype=torch.float32
        )
        vars_train = []
        vars_test = []
        if additional_vars:
            for var in additional_vars:
                var_train = np.take(var, train_id, split_axis)
                vars_train.append(
                    torch.asarray(
                        var_train.reshape((-1, self.dim)), dtype=torch.float32
                    )
                )
                var_test = np.take(var, test_id, split_axis)
                vars_test.append(
                    torch.asarray(var_test.reshape((-1, self.dim)), dtype=torch.float32)
                )
        self.data = {
            "eps_train": to_np(eps_train),
            "eps_test": to_np(eps_test),
            "sig_train": to_np(sig_train),
            "sig_test": to_np(sig_test),
            "train_id": train_id,
            "test_id": test_id,
            "file_name": self.file_name,
        }
        np.savez(f"{self.file_name}/train_test_data.npz", **self.data)

        for epoch in range(self.hyperparams["max_epochs"]):
            output = self.training_layer(
                *self.theta_hat,
                eps_train,
                *vars_train,
                solver_args={
                    "solve_method": self.hyperparams["optimization_solver"],
                },
            )
            sig_hat = output[0]

            self.train_loss = self.loss_function(sig_hat, sig_train)
            output_test = self.training_layer(
                *self.theta_hat,
                eps_test,
                *vars_test,
                solver_args={"solve_method": self.hyperparams["optimization_solver"]},
            )
            sig_hat_test = output_test[0]
            test_loss = self.loss_function(sig_hat_test, sig_test)
            self.losses.append((to_np(self.train_loss), to_np(test_loss)))

            print(
                f"Epoch {epoch+1} Train loss = {self.losses[-1][0]} Test loss {self.losses[-1][1]}"
            )

            np.savetxt(
                f"{self.file_name}/losses.csv",
                np.asarray(self.losses),
                delimiter=",",
                header="Train loss, Test loss",
            )
            np.savez(f"{self.file_name}/theta.npz", *(to_np(t) for t in self.theta_hat))

            self.opt.zero_grad()
            self.train_loss.backward()
            self.opt.step()

            if (
                epoch % self.callback_frequency == 0
                or epoch == self.hyperparams["max_epochs"] - 1
            ):
                prediction = (to_np(sig_hat), to_np(sig_hat_test))
                self.callback(epoch, self.theta_hat, self.data, prediction)
