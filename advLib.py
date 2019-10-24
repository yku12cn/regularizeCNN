"""
    This module provides tools for generating adversaries
    Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""
import torch.nn.functional as F
import torch


def genGrad(image, net):
    r"""calculate gradient for each class w.r.t input image.

    Arguments:
        genGrad(image, net):
        image : input image
        net : the NN you want to test

    .. note::
        this function will automatically transfer data
        to the device where your NN is located
    """
    device = next(net.parameters()).device  # fetch where is the model
    isTrain = net.training  # mark model's original state
    net.eval()  # set net into eval mode

    net.zero_grad()
    inputs = image.to(device)
    if len(inputs.shape) == 3:  # fix single image error
        inputs = inputs.unsqueeze(0)
    inputs.requires_grad = True
    output = net(inputs)

    c_num = output.shape[1]
    masks = F.one_hot(torch.arange(0, c_num).view(c_num, 1) % c_num)
    masks = masks.float().to(device)
    grads = torch.zeros((10, 3, 32, 32), device=device)
    for mask, slot in zip(masks, grads):
        output.backward(mask, retain_graph=True)
        slot += inputs.grad[0]
        inputs.grad.zero_()

    inputs.requires_grad = False
    if isTrain:
        net.train()

    return grads, output.detach()


def genAdv(data, net, rate=1.01, maxtry=100):
    r"""generate closest adversary.

    Arguments:
        genAdv(data, net, rate):
        data : input data [image, label]
        net : the NN you want to test
        rate : overshoot rate

    .. note::
        this function will automatically transfer data
        to the device where your NN is located
    """
    device = next(net.parameters()).device  # fetch where is the model
    image, label = data[0].to(device), data[1]
    flag = False
    while True:
        maxtry -= 1
        if maxtry == 0:
            print("Max try reached")
            break
        grad, output = genGrad(image, net)
        _, predicted = torch.max(output.data, 1)
        if predicted != label:
            break

        flag = True
        output = output[0] - output[0][label]
        output.abs_()
        output[label] = float("Inf")

        grad = grad - grad[label]
        gnorm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1)
        gnorm[label] = 1

        _, target = torch.min(torch.div(output, gnorm), 0)
        delta = grad[target] * (output[target] / gnorm[target] ** 2) * rate

        image = torch.clamp(image + delta, -1, 1)  # clamp color range

    return image, flag
