import torch
import networks

def max_update(W, W_activation, lambda_, mode):
    if mode == "soft_weight_recovery":
        return max_update_soft_weight_recovery(W, W_activation, lambda_)
    elif mode == "multi_soft_weight_recovery":
        return max_update_multi_soft_weight_recovery(W, W_activation, lambda_)
    elif mode == "unnormalized":
        return max_unnormalized_update(W, W_activation, lambda_)
    elif mode == "normalized":
        return max_update_normalized(W, W_activation, lambda_)
    else:
        raise ValueError("Unknown update mode")


def min_update(W, W_activation, lambda_, mode):
    if mode == "soft_weight_recovery":
        return min_update_soft_weight_recovery(W, W_activation, lambda_)
    elif mode == "multi_soft_weight_recovery":
        return min_update_multi_soft_weight_recovery(W, W_activation, lambda_)
    elif mode == "unnormalized":
        return min_unnormalized_update(W, W_activation, lambda_)
    elif mode == "normalized":
        return min_update_normalized(W, W_activation, lambda_)
    else:
        raise ValueError("Unknown update mode")


def min_unnormalized_update(W, W_activation, lambda_):
    return W - lambda_ * (1 - W_activation / torch.sum(W_activation))


def max_unnormalized_update(W, W_activation, lambda_):
    return W + lambda_ * (1 - W_activation / torch.sum(W_activation))


def max_update_normalized(W, W_activation, lambda_):
    return W + lambda_ * (1 - W_activation / torch.sum(W_activation, dim=0))


def min_update_normalized(W, W_activation, lambda_):
    return W - lambda_ * (1 - W_activation / torch.sum(W_activation, dim=0))


def max_update_soft_weight_recovery(W, W_activation, lambda_):
    minimums = torch.min(W_activation, dim=1).values
    maximums = torch.max(W_activation, dim=1).values
    ratio = (W_activation.T - minimums) / (maximums - minimums)
    ratio = ratio.T
    return W + lambda_ * (1 - ratio)


def max_update_multi_soft_weight_recovery(W, W_activation, lambda_):
    minimums = torch.topk(W_activation, 3, dim=1, largest=False).values[:, 2]
    maximums = torch.topk(W_activation, 3, dim=1, largest=True).values[:, 2]
    ratio = (W_activation.T - minimums) / (maximums - minimums)
    ratio = ratio.T
    return W + lambda_ * (1 - ratio)


# add comment (inspiriert und anderst im report --> wie positioniert)
def min_update_soft_weight_recovery(W, W_activation, lambda_):
    minimums = torch.min(W_activation, dim=1).values
    maximums = torch.max(W_activation, dim=1).values
    ratio = (W_activation.T - minimums) / (maximums - minimums)
    ratio = ratio.T
    return W - lambda_ * (1 - ratio)


def min_update_multi_soft_weight_recovery(W, W_activation, lambda_):
    minimums = torch.topk(W_activation, 3, dim=1, largest=False).values[:, 2]
    maximums = torch.topk(W_activation, 3, dim=1, largest=True).values[:, 2]
    ratio = (W_activation.T - minimums) / (maximums - minimums)
    ratio = ratio.T
    return W - lambda_ * (1 - ratio)
