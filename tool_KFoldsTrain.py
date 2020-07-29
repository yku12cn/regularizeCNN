"""
    Do K-Folds cross-validation
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torch

from ykuTorch import evalNN
from ykuUtils import printLog, tStamp
from dataProcess import setKFold


def doKFolds(dataset, net, k, epoch=1, train_batch=10, test_batch=512,
             optimizer=None, criterion=None, workers=4, device=None,
             logfile=None, rational=False):
    r"""Do K-Folds cross-validation and generates confusion matrixes

    Args:
        dataset (VDPlus set): Data for training.
        net (simpleNN module): the net to be evaluated
        k (int): how many folds
        epoch (int, optional): Defaults to 1.
        train_batch (int, optional): batch size for training. Defaults to 10.
        test_batch (int, optional): batch size for testing. Defaults to 512.
        optimizer (torch.optim.optimizer): Defaults to default Adam.
        criterion (torch.nn.lose): Defaults to nn.CrossEntropyLoss().
        workers (int, optional): Multithread loader. Defaults to 4.
        device (torch.device, optional): running on which device.
        logfile (str, optional): Log file location
        rational (bool, optional): abort if accuracy is too low

    Returns:
        (list, list): list of train results, list of test results
    """
    kf = setKFold(dataset, folds=k)
    timestamp = tStamp()
    if logfile:
        printL = printLog(logfile)
    else:
        printL = printLog(f"./_outputs/{timestamp}_{k}-folds.log")

    test_cm = []
    train_cm = []
    for i in range(k):
        printL(f"Start Fold-{i+1}...", t=True)

        # Init net
        net.__init__(device=device)
        # Init optimizer
        optimizer.__init__(net.parameters(), **optimizer.defaults)
        # set folds
        kf.select(i, train=True)
        net.trainNet(kf, epoch, train_batch, workers, 0,
                     optimizer, criterion, printL)

        printL("Do train evaluation...")
        _cm = evalNN.calCM(
            *evalNN.evalAllCLF(kf, net, test_batch, device=device),
            len(kf.classes), device)
        train_cm.append(_cm)
        evalNN.showCM(_cm, tags=kf.classes, printer=printL)

        printL("Do test evaluation...")
        kf.select(i, train=False)
        _cm = evalNN.calCM(
            *evalNN.evalAllCLF(kf, net, test_batch, device=device),
            len(kf.classes), device)
        test_cm.append(_cm)
        evalNN.showCM(_cm, tags=kf.classes, printer=printL)

        if rational:
            if evalNN.statsOfCM(_cm)["acc"] < (100/len(dataset.classes) + 10):
                break

    return train_cm, test_cm


def aggregateMatrix(matlist):
    r"""aggregate confusion matrix

    Args:
        matlist (list): list of confusion matrix

    Returns:
        tensor: aggregated confusion matrix
    """
    return torch.stack(matlist).sum(0)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from ykuTorch import fetch_device
    from dataProcess import setfromPack
    from myNorm import matrixNorm
    from myAdam import Adam
    from demoCNN import Net

    dev = fetch_device("gpu", set_default=True)

    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    trainset = setfromPack("./_data/FashionMNIST_Unpack/train/",
                           transform=transform)

    _net = Net(device=dev)
    norm = matrixNorm(2, 2, row=True)
    op = Adam(_net.parameters(), lr=0.001, weight_decay=0.01,
              reg=norm, reglayer=4)

    trainCM, testCM = doKFolds(trainset, _net, 5, 5, 64, 512, op,
                               workers=0, device=dev)

    print("Training results:")
    evalNN.showCM(aggregateMatrix(trainCM), tags=trainset.classes)
    print("Testing results:")
    evalNN.showCM(aggregateMatrix(testCM), tags=trainset.classes)
