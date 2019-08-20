"""
    This module provides some general ways to evaluate NN
    Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


def showImg(img, label=None, unnormalize=True):
    r"""Display a image tensor

    Arguments:
        imshow(img, unnormalize):
        img : tensor that represent a image
        unnormalize : if the image is normalized

    .. note::
        by default, it will try to undo
        the (0.5, 0.5, 0.5) normalization
    """
    if len(img.shape) == 4:
        print("showImg: Warning! img dimension equals 4:", img.shape)
        print("showImg: Try to make grid")
        img = make_grid(img)

    if unnormalize:
        img = img / 2 + 0.5     # unnormalize

    if img.device != torch.device('cpu'):
        img = img.cpu()  # if img is not on cpu

    if label:
        plt.title(label)

    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def showDp(data, dic=None, unnormalize=True):
    r"""Display one data point. aka "an image with label"

    Arguments:
        imshow(data, dic, unnormalize):
        data : a data point, image & label
        dic : dictionary for label, should be a list
        unnormalize : if the image is normalized

    .. note::
        by default, it will try to undo
        the (0.5, 0.5, 0.5) normalization
        dic is optional
    """
    label = dic[data[1]] if dic else data[1]
    showImg(data[0], label, unnormalize)


def evalAllCLF(inputdata, net, batchsize=1):
    r"""evaluate model with whole dataset.
        the dataset should follow "torchvision.datasets"

    Arguments:
        evalAllCLF(inputdata, net, batchsize):
        inputdata : <class 'torchvision.datasets'>
        net : the NN you want to test
        batchsize : related to your RAM

    .. note::
        by default, batchsize=1
        this function will automatically transfer data
        to the device where your NN is located
    """
    device = next(net.parameters()).device  # fetch where is the model
    isTrain = net.training  # mark model's original state
    net.eval()  # set net into eval mode
    # Build data loader
    loader = torch.utils.data.DataLoader(inputdata, batch_size=batchsize,
                                         shuffle=False, num_workers=2)
    correct = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    if isTrain:
        net.train()
    return correct, len(loader.dataset)


def evalCLF(inputdata, net):
    r"""evaluate model with some data.
        input data shall be a list of two tensors:
        "[tensor(index,C-Channel,Height,Width),
        tensor(labels)]"

    Arguments:
        evalCLF(inputdata, net):
        inputdata : a list of two tensors [images, labels]
        net : the NN you want to test

    .. note::
        this function will automatically transfer data
        to the device where your NN is located
    """
    device = next(net.parameters()).device  # fetch where is the model
    isTrain = net.training  # mark model's original state
    net.eval()  # set net into eval mode

    # pre-process image
    inputs = inputdata[0].to(device)
    if len(inputs.shape) == 3:  # fix single image error
        inputs = inputs.unsqueeze(0)

    # pre-process labels
    if isinstance(inputdata[1], torch.Tensor):
        labels = inputdata[1].to(device)
    else:
        labels = torch.tensor(inputdata[1], device=device)

    if len(labels.shape) == 0:  # fix single label error
        labels = labels.unsqueeze(0)

    if inputs.shape[0] != labels.shape[0]:
        raise ValueError("length of labels should match number of images")

    # eval NN
    with torch.no_grad():
        _, predicted = torch.max(net(inputs).data, 1)
        correct = predicted == labels

    if isTrain:
        net.train()

    return predicted, correct
