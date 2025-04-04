import os
import torch
import numpy as np
import random
import cvxpy as cp


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rand_uniform(shape, low=-1.0, high=1.0, dtype=torch.float32, **kwargs):
    return (
        (high - low) * torch.rand(*shape, dtype=dtype, **kwargs) + low
    ).requires_grad_()


def to_np(x):
    return x.detach().numpy()


def to_sym_matrix(data, n):
    X = [[0 for j in range(n)] for i in range(n)]
    row, col = np.tril_indices(n)
    for k in range(n * (n + 1) // 2):
        X[row[k]][col[k]] = data[k]
        X[col[k]][row[k]] = data[k]
    return cp.bmat(X)


def to_triu_matrix(data, n):
    X = [[0 for j in range(n)] for i in range(n)]
    row, col = np.triu_indices(n)
    for k in range(n * (n + 1) // 2):
        X[row[k]][col[k]] = data[k]
    return cp.bmat(X)


def vmap(fn, in_axes=0, out_axes=0):
    """
    Vectorizes the given function fn, applying it along the specified axes.

    Parameters:
    fn : function
        The function to be vectorized.
    in_axes : int or tuple of ints, optional
        Specifies which axis to map over for each input. If a single int is provided,
        that axis is used for all inputs. If a tuple is provided, each entry corresponds
        to a different input. Default is 0.
    out_axes : int or tuple of ints, optional
        Specifies the axis along which outputs should be stacked. If a single int is
        provided, that axis is used for all outputs. If a tuple is provided, each entry
        corresponds to a different output. Default is 0.

    Returns:
    vectorized_fn : function
        The vectorized version of the input function.
    """

    def vectorized_fn(*args):
        def moveaxis_if_array(x, axis):
            if isinstance(x, np.ndarray):
                return np.moveaxis(x, axis, 0)
            return x

        def slice_data(x, index):
            if isinstance(x, np.ndarray):
                return x[index]
            elif isinstance(x, dict):
                return {key: value[index] for key, value in x.items()}
            return x

        def stack_outputs(outputs, out_axes):
            if isinstance(outputs[0], np.ndarray) or np.isscalar(outputs[0]):
                return np.stack(outputs, axis=out_axes)
            elif isinstance(outputs[0], dict):
                return {
                    key: np.stack([output[key] for output in outputs], axis=out_axes)
                    for key in outputs[0].keys()
                }
            return outputs

        # Handle the case where in_axes is a single int or tuple of ints
        if isinstance(in_axes, int):
            in_axes_tuple = (in_axes,) * len(args)
        else:
            in_axes_tuple = in_axes

        # Move the in_axes to the first axis for each input
        moved_inputs = [
            moveaxis_if_array(arg, axis) for arg, axis in zip(args, in_axes_tuple)
        ]

        # Determine the size of the first dimension (after axis move) to loop over
        loop_size = (
            moved_inputs[0].shape[0]
            if isinstance(moved_inputs[0], np.ndarray)
            else len(next(iter(moved_inputs[0].values())))
        )

        # Apply the function to each slice of the inputs along the first axis
        results = [
            fn(*[slice_data(arg, i) for arg in moved_inputs]) for i in range(loop_size)
        ]

        # Handle the case where out_axes is a single int or tuple of ints
        if isinstance(out_axes, int):
            out_axes_tuple = (out_axes,)
        else:
            out_axes_tuple = out_axes

        # Stack the results along the specified out_axes
        if isinstance(results[0], tuple):
            # If the function returns multiple outputs, handle each separately
            stacked_result = tuple(
                stack_outputs([res[i] for res in results], out_axes_tuple[i])
                for i in range(len(results[0]))
            )
        else:
            # If the function returns a single output
            stacked_result = stack_outputs(results, out_axes)

        return stacked_result

    return vectorized_fn


# Function to sample a random batch from the dataset
def split_test_train(dataset, outputs, test_ratio=0.1, axis=0):
    # Randomly permute the indices of the columns (samples)
    n = dataset.shape[axis]
    perm = np.random.permutation(
        n
    )  # get random permutation of nsize (number of columns)
    n_train = int((1 - test_ratio) * n)
    train_indices = perm[:n_train]
    test_indices = perm[n_train:]
    data_train = np.take(dataset, train_indices, axis)
    data_test = np.take(dataset, test_indices, axis)
    output_train = np.take(outputs, train_indices, axis)
    output_test = np.take(outputs, test_indices, axis)
    return (
        (data_train, data_test),
        (output_train, output_test),
        (train_indices, test_indices),
    )
