"""
    General tools and helper functions
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torch


@torch.enable_grad()
def genGrad(x, clf, criterion=None, device=None):
    r"""calculate gradient for each class w.r.t input data

    Args:
        x (tensor): input data
        clf (callable obj): the classifier you want to test
        criterion (nn.loss, optional): loss function. Defaults to None.
            !! Be careful with the reduction setting in loss function.
            !! Reduction setting will affect gradients accordingly.
        device (torch.device, optional): test on which device. This should
            be on the same device as the 'clf' does.

    Returns:
        tensor: gradient for each class
            0-dim represents each class accordingly
    """
    x = x.detach().to(device).requires_grad_(True)
    output = clf(x)

    c_num = output.shape[1]
    p_target = torch.arange(0, c_num, device=device).unsqueeze_(1)

    grads = torch.zeros((c_num,) + x.shape, device=device)
    for target, slot in zip(p_target, grads):
        if criterion:
            loss = criterion(output, target.repeat(output.shape[0]))
        else:
            # Cal gradient of direct output
            loss = output[:, target].sum()

        loss.backward(retain_graph=True)
        slot.add_(x.grad)
        x.grad.zero_()

    return grads, output.detach()
