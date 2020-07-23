"""
    This module provides some general ways to evaluate NN and analyse results.
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.hub import tqdm
from torchvision.utils import make_grid


def showImg(img, label=None, unnormalize=True, trans=True):
    r"""Display a image tensor

    Args:
        img (tensor): tensor that represents a image
        label (str, optional): title for the image. Defaults to None.
        unnormalize (bool, optional): if the image is normalized
                                by default, it will try to undo
                                the (0.5, 0.5, 0.5) normalization.
        trans (bool, optional): re-arrange data sequence. Defaults to True.
                pyTorch: [channel, height, weight]
                pyplot.imshow: [height, weight, channel] or [height, weight]
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

    np_img = img.numpy()
    if trans:  # re-arrange data shape
        np_img = np.transpose(np_img, (1, 2, 0))

    if len(np_img.shape) == 3:  # handle gray image
        np_img = np_img.squeeze()

    plt.imshow(np_img)
    plt.show()


def showDp(data, dic=None, unnormalize=True, trans=True):
    r"""Display one data point. aka "an image with label"

    Args:
        data ((tensor, int)): a data point, image & label
        dic (list of str, optional): dictionary for label. Defaults to None.
        unnormalize (bool, optional): if the image is normalized
                                by default, it will try to undo
                                the (0.5, 0.5, 0.5) normalization.
        trans (bool, optional): re-arrange data sequence. Defaults to True.
                pyTorch: [channel, height, weight]
                pyplot.imshow: [height, weight, channel] or [height, weight]
    """
    label = dic[data[1]] if dic else data[1]
    showImg(data[0], label, unnormalize, trans)


@torch.no_grad()
def calCM(pred, truth, c_num, device=None):
    r"""Generate confusion matrix from prediction and ground truth

    Args:
        pred (1D tensors/lists): prediction
        truth (1D tensors/lists): ground truth
        c_num (int): num of classes, will always assume
                     lables are [0, 1, 2 ... c_num - 1]
        device (torch.device, optional):  GPU or CPU? Defaults to None.

    Returns:
        2D tensor : the output will be like:
                         prediction
                      |* . . . . . .|
                    t |. * . . . . .|
                    r |. . * . . . .|
                    u |. . . * . . .|
                    t |. . . . * . .|
                    h |. . . . . * .|
                      |. . . . . . *|
    """
    if len(pred) != len(truth):
        raise ValueError("data lenth mismatch")
    length = len(pred)

    # Prepare pred
    if isinstance(pred, list):
        pred = torch.tensor(pred, device=device, requires_grad=False)
    elif isinstance(pred, torch.Tensor):
        if device:
            pred = pred.detach().to(device)
        else:
            pred = pred.detach()
            device = pred.device
    else:
        raise TypeError("pred is neither tensor nor list")

    # Prepare truth
    if isinstance(truth, list):
        truth = torch.tensor(truth, device=device, requires_grad=False)
    elif isinstance(truth, torch.Tensor):
        truth = truth.detach().to(device)
    else:
        raise TypeError("truth is neither tensor nor list")

    # Generate one hot version of the prediction
    pred_o = torch.zeros(length, c_num, dtype=torch.float, device=device)
    pred_o.scatter_(1, pred.unsqueeze(1), 1)

    truth_o = torch.zeros(c_num, length, dtype=torch.float, device=device)
    truth_o.scatter_(0, truth.unsqueeze(0), 1)

    # Generate confusion matrix
    return truth_o @ pred_o


@torch.no_grad()
def statsOfCM(mat):
    r"""Analyze the input confusion matrix
        return a dictionary of its statistics

    Args:
        mat (2D tensor): confusion matrix

    Returns:
        dict : a dictionary of statistics
            "total": Total cases
            "acc" : Accuracy
            "tp" : list of True positives
            "gt" : Num of each class
            "precision" : precision of each class
            "recall" : recall of each class
            "f1" : f1 score of each class
    """
    # True positives
    tp = mat.diagonal()
    # Num of each class
    gt = mat.sum(1)
    # Precision (PPV): TP/(TP+FP)
    precision = tp / mat.sum(0)
    # Recall (TPR): TP/(TP+FN)
    recall = tp / gt
    # F1 score
    f1 = 2 * precision * recall / (precision + recall)
    precision.mul_(100)
    recall.mul_(100)

    # Assemble dictionary
    stats = {
        "total": mat.sum().int().item(),
        "tp": tp.int().tolist(),
        "gt": gt.int().tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist()
    }
    stats["acc"] = sum(stats["tp"]) / stats["total"] * 100
    return stats


def showCM(mat, tags=None, printer=print):
    r"""Analyze the input confusion matrix
        print Accuracy/Precision/Recall/F1 score

    Args:
        mat (2D tensor): confusion matrix
        tags (list of str, optional): the name of each class. Defaults to None.
        printer (callable, optional): you can switch to other print function.
                                      Defaults to print.
    """
    if tags is None:
        # Generate default tag
        tags = list(range(len(mat)))

    stats = statsOfCM(mat)  # Cal statistics
    mat = mat.int().tolist()
    gt = stats["gt"]
    precision = [round(x, 2) for x in stats["precision"]]
    recall = [round(x, 2) for x in stats["recall"]]
    f1 = [round(x, 2) for x in stats["f1"]]

    # Assemble the table
    table = [["T\\P"] + tags + ["SUM", "TPR %", "f1"]]
    for i, row in enumerate(mat):
        table += [[tags[i]] + row + [gt[i], recall[i], f1[i]]]
    table += [["PPV %"] + precision + [stats["total"], "", ""]]

    # Calculate display space for each column
    lens = [max([len(str(item)) for item in col]) for col in zip(*table)]
    # Generate format scheme for each column
    scheme = '| '.join(f'{{:{item}}}' for item in lens)
    # Print table
    for row in table:
        printer(scheme.format(*row))
    printer(f"Accuracy: {stats['acc']:.2f}%",)


@torch.no_grad()
def evalAllCLF(inputdata, clf, batchsize=1, device=None):
    r"""evaluate a classifier with the whole dataset

    Args:
        inputdata (torchvision.datasets): dataset to be evaluated
        clf (callable obj): the classifier you want to test
        batchsize (int, optional): related to your RAM. Defaults to 1.
        device (torch.device, optional): test on which device. This should
            be on the same device as the 'clf' does.
            If left undefined, it will try to match with the 'clf' or current
            default device

    Returns:
        (1D tensor, 1D tensor): prediction & ground truth

    .. note::
        If clf is a torch.nn.Module, 'device' will be disregarded
        And data will always be pass to where clf is located
    """
    isTrain = False
    if isinstance(clf, torch.nn.Module):  # Special treatment for NN
        isTrain = clf.training  # mark model's original state
        clf.eval()  # set net into eval mode
        device = next(clf.parameters()).device  # fetch where is the model

    # Build data loader
    loader = torch.utils.data.DataLoader(inputdata, batch_size=batchsize,
                                         shuffle=False, num_workers=2)
    # Build progress bar
    pb = tqdm(total=len(inputdata), desc=f"Evaluating NN on {device}: ",
              leave=True, ascii=(os.name == "nt"))
    ind = 0
    pred, gt = None, None
    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        _, predicted = torch.max(clf(inputs).detach(), 1)
        if pred is None:
            # pre-allocate memory for storing results
            pred = torch.empty(len(inputdata), dtype=predicted.dtype,
                               device=device)
            gt = torch.empty(len(inputdata), dtype=labels.dtype,
                             device=device)
        pred[ind:ind + len(labels)] = predicted
        gt[ind:ind + len(labels)] = labels
        ind += len(labels)
        pb.update(len(labels))
    pb.close()

    if isTrain:
        clf.train()

    return pred, gt


@torch.no_grad()
def evalCLF(inputdata, clf, device=None):
    r"""evaluate a classifier with some data

    Args:
        inputdata ([tensor, tensor]): dataset to be evaluated
                   [images, labels]
                   [tensor(index,C-Channel,Height,Width), tensor(labels)]
        clf (callable obj): the classifier you want to test
        device (torch.device, optional): test on which device. This should
            be on the same device as the 'clf' does.
            If left undefined, it will try to match with the 'clf' or current
            default device

    Returns:
        (1D tensor, 1D tensor) : prediction & ground truth

    .. note::
        If clf is a torch.nn.Module, 'device' will be disregarded
        And data will always be pass to where clf is located
    """
    isTrain = False
    if isinstance(clf, torch.nn.Module):  # Special treatment for NN
        isTrain = clf.training  # mark model's original state
        clf.eval()  # set net into eval mode
        device = next(clf.parameters()).device  # fetch where is the model

    # pre-process image
    inputs = inputdata[0].detach().to(device)

    # pre-process labels
    if isinstance(inputdata[1], torch.Tensor):
        labels = inputdata[1].detach().to(device)
    else:
        labels = torch.tensor(inputdata[1], device=device, requires_grad=False)

    if len(labels.shape) == 0:  # fix single label error
        labels = labels.unsqueeze(0)

    if inputs.shape[0] != labels.shape[0]:
        raise ValueError("length of labels should match number of images")

    # eval NN
    _, pred = torch.max(clf(inputs).detach(), 1)

    if isTrain:
        clf.train()

    return pred, labels
