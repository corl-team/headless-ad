import os
import random
import time
from itertools import product

import numpy as np
import torch
from scipy.interpolate import interp1d


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)


class Timeit:
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.Event(enable_timing=True)
            self.end_gpu = torch.cuda.Event(enable_timing=True)
            self.start_gpu.record()
        self.start_cpu = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            self.end_gpu.record()
            torch.cuda.synchronize()
            self.elapsed_time_gpu = self.start_gpu.elapsed_time(self.end_gpu) / 1000
        else:
            self.elapsed_time_gpu = -1.0
        self.elapsed_time_cpu = time.time() - self.start_cpu


def train_test_goals(grid_size, num_train_goals, seed):
    set_seed(seed)
    assert num_train_goals <= grid_size**2

    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    goals = np.random.permutation(goals)

    train_goals = goals[:num_train_goals]
    test_goals = goals[num_train_goals:]
    return train_goals, test_goals


def all_goals(grid_size):
    all_goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    all_goals = np.random.permutation(all_goals)
    return all_goals


def orthogonal_(gen, tensor, gain=1):
    r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if tensor.numel() == 0:
        # no-op
        return tensor
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1, generator=gen)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


@torch.no_grad()
def index_mask(num_actions: torch.Tensor, num_total_actions: int):
    batch_size = num_actions.shape[0]
    mask = torch.zeros(
        batch_size,
        num_total_actions + 1,
        dtype=torch.float32,
        device=num_actions.device,
    )
    mask[(torch.arange(batch_size), num_actions)] = 1
    mask = mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column
    mask = 1.0 - mask

    return mask


def norm_regret(regrets, lower_regrets, upper_regrets):
    normalized_regret = (regrets.mean(0)[-1] - lower_regrets.mean(0)[-1]) / (
        upper_regrets.mean(0)[-1] - lower_regrets.mean(0)[-1]
    )

    return normalized_regret


def get_action_sequences(num_actions: int, seq_len: int):
    seqs = list(product(np.arange(num_actions), repeat=seq_len))
    seqs = np.vstack(seqs)

    return seqs


def rew_to_ret(rewards: np.ndarray):
    returns = rewards.reshape(rewards.shape[0], rewards.shape[1] // 20, 20).sum(2)

    return returns


def resize_returns(returns, target_shape):
    x_original = np.linspace(0, 1, len(returns))
    interpolation_function = interp1d(x_original, returns, kind="linear")

    x_new = np.linspace(0, 1, target_shape)

    interpolated_array = interpolation_function(x_new)

    return interpolated_array
