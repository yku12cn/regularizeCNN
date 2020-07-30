"""
    Optimize the weight of a given regularizer by cross-validation
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from pathlib import Path

from ykuTorch import evalNN
from ykuUtils import printLog, tStamp

from myAdam import Adam
from tool_KFoldsTrain import doKFolds, aggregateMatrix


def searchRW(net, dataset, w_range, norm, reglayer, depth=9,
             epoch=5, train_batch=64, test_batch=512,
             title="./_outputs/weightsearch", device=None):
    r"""Optimize the weight of a given regularizer by cross-validation

    Args:
        net (simpleNN module): the net to be evaluated
        dataset (VDPlus set): Data for training.
        w_range (tuple): (start, end) searching range
        norm (youTorch.cNorm): the norm regularizer
        reglayer (int): apply regularizer starting from which layer
        depth (int, optional): search depth. Defaults to 9.
        title (str, optional): the folder for storing reports.
        device (torch.device, optional): running on which device.
    """
    folder = Path(title)
    printL = printLog(folder / f"{tStamp()}_tuneRegWeight.log")

    start_w = w_range[0]
    end_w = w_range[1]

    step = (end_w - start_w) / 2

    c_w = start_w

    last_acc = 0
    _dic = {}

    printL(f"""Start Searching...
==========Settings Brief===========
Search Range: {w_range}
Search depth: {depth}
Regularizer: {norm}
Regularize starts from layer #{reglayer}
logfile: {printL.logfile()}
===========Dataset Brief===========
{dataset}
=========Model definition=========
{net.printFUN()}
==================================""")

    while depth > 0:
        if c_w not in _dic:
            printL(f"Current weight is: {c_w}", t=True)
            op = Adam(net.parameters(), lr=0.001, weight_decay=c_w,
                      reg=norm, reglayer=reglayer)
            trainCM, testCM = doKFolds(
                dataset, net, 5, epoch, train_batch, test_batch, op,
                workers=0, device=device, rational=True,
                logfile=folder / f"{tStamp()}_{c_w:.6f}_k-folds.log")

            printL("Training results:", t=True)
            evalNN.showCM(aggregateMatrix(trainCM), dataset.classes, printL)
            printL("Testing results:", t=True)
            evalNN.showCM(aggregateMatrix(testCM), dataset.classes, printL)

            _acc = evalNN.statsOfCM(aggregateMatrix(testCM))["acc"]
            _acct = evalNN.statsOfCM(aggregateMatrix(trainCM))["acc"]

            if _acc < (100/len(dataset.classes) + 10):
                _acc = 100/len(dataset.classes)
                _acct = _acc
                printL("Accuracy too low, abort k-fold")
            else:
                printL(f"Accuracy is {_acc} at weight[{c_w}]", t=True)
            _dic[c_w] = [
                round(_acc, 6),
                round(_acct, 6)]
        else:
            _acc = _dic[c_w][0]

        if _acc >= last_acc:
            c_w += step
        else:
            print("inv", c_w)
            step /= -2
            depth -= 1
            c_w += step

        if c_w > max(start_w, end_w):
            c_w = max(start_w, end_w)
            step = -abs(step)
        elif c_w < min(start_w, end_w):
            c_w = min(start_w, end_w)
            step = abs(step)

        last_acc = _acc

    sortlist = [[k] + v for k, v in _dic.items()]
    sortlist.sort(key=lambda x: x[0])
    printL(f"Done!, Optimal weight is {max(sortlist, key=lambda x: x[1])[0]}")
    sortlist = [["Weight", "Test Accuracy", "Train Accuracy"]] + sortlist
    # Calculate display space for each column
    lens = [max([len(str(item)) for item in col]) for col in zip(*sortlist)]
    # Generate format scheme for each column
    scheme = '| '.join(f'{{:{item}}}' for item in lens)
    for item in sortlist:
        printL(scheme.format(*item))


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from ykuTorch import fetch_device
    from dataProcess import setfromPack
    from myNorm import matrixNorm
    from _foo_Linear import Net as Linear

    dev = fetch_device("gpu", set_default=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    searchRW(
        title="./_outputs/testrun",
        dataset=setfromPack(
            "./_data/MNIST_Unpack/train/",
            transform=transform),
        net=Linear(device=dev),
        reglayer=4,
        norm=matrixNorm(r_col=2, r_row=2, row=True),
        w_range=(1, 0),
        )
