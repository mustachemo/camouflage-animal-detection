import torch
from torch import Tensor


def tversky_index(input: Tensor, target: Tensor, alpha=0.75, beta=0.25, epsilon=1e-6):
    assert input.size() == target.size(), "Input and target must have the same shape"
    assert input.dim() == 3
    assert target.max() <= 1.0, "Target values must be in [0, 1]"

    sum_dim = (-1, -2) if input.dim() == 3 else (-1, -2, -3)

    true_pos = (input * target).sum(dim=sum_dim)
    false_pos = (input * (1 - target)).sum(dim=sum_dim)
    false_neg = ((1 - input) * target).sum(dim=sum_dim)

    # calculate the Tversky index
    numerator = true_pos
    denominator = true_pos + (alpha * false_pos) + (beta * false_neg)
    tversky = numerator / (denominator + epsilon)

    return tversky.mean()


def tversky_loss(input: Tensor, target: Tensor, alpha=0.75, beta=0.25, epsilon=1e-6):
    return 1 - tversky_index(input, target, alpha, beta, epsilon)
