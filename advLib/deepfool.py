"""
    This module provides tools for generating adversaries
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torch
from . import genGrad


@torch.no_grad()
def _stepcore(g, outs, targets, q, device, overshoot):
    _, pred = torch.max(outs.detach(), 1)
    mask = targets == pred  # Use to mask out already wrong samples

    indx = torch.arange(g.shape[1], device=device)
    outs.add_(-outs.gather(1, targets.unsqueeze(1))).abs_()
    g.add_(-g[targets, indx])
    g.transpose_(0, 1)
    gn = g.view(g.shape[0:2] + (-1,)).norm(p=q, dim=2)
    outs.div_(gn)
    outs.scatter_(1, targets.unsqueeze(1), float('inf'))
    dis, sel = outs.min(1)  # select minimum perturbation
    # Cal step size and mask out already wrong ones
    dis.div_(gn[indx, sel].pow_(q-1)).add_(overshoot).mul_(mask)
    g = g[indx, sel]
    g = g.abs().pow_(q-1).mul_(g.sign())
    g = g.transpose_(0, -1).mul_(dis).transpose_(0, -1)

    return g, sel, mask.sum()


def deepfoolL(data, clf, q=2, device=None, overshoot=0.0):
    r"""Generate the minimum perturbation to fool a linear classifer
        minimize l_p distance.

    Args:
        data (tuple): [images, labels].
        clf (callable obj): the classifier you want to test.
        q (num, optional): q = p/(p-1)
        device (torch.device, optional): test on which device. This should
            be on the same device as the 'clf' does.
        overshoot (float, optional): without overshoot, the perturbation
            will bring you right on the margin.

    Returns:
        (tensor, 1D tensor): the perturbation matrix and targeted label.
    """
    x = data[0].to(device)
    targets = data[1].to(device)
    g, outs = genGrad(x, clf, device=device)
    delta, sel, _ = _stepcore(g, outs, targets, q, device, overshoot)
    return delta, sel


def deepfoolGD(data, clf, q=2, device=None, rate=0.5, overshoot=0.000001):
    r"""Generate the minimum perturbation to fool a classifer by
        gradient decent. Minimize l_p distance.

    Args:
        data (tuple): [images, labels].
        clf (callable obj): the classifier you want to test.
        q (num, optional): q = p/(p-1)
        device (torch.device, optional): test on which device. This should
            be on the same device as the 'clf' does.
        rate (float, optional): descending rate. Defaults to 0.5.
        overshoot (float, optional): without overshoot, the perturbation
            will bring you right on the margin.

    Returns:
        (tensor, 1D tensor): the perturbation matrix and targeted label.
    """
    x = data[0].to(device)
    targets = data[1].to(device)
    nx = torch.zeros(x.shape)
    nx.add_(x)

    while True:
        g, outs = genGrad(nx, clf, device=device)
        delta, sel, corr = _stepcore(g, outs, targets, q, device, overshoot)
        if corr.eq(0):
            break
        nx.add_(delta, alpha=rate)

    return nx.add_(-x), sel
