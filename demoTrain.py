"""
    Run whole demo
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torchvision.transforms as transforms
import torchvision

from ykuTorch import fetch_device, evalNN
from ykuUtils import printLog, tStamp

from demoCNN import Net
from myAdam import Adam
from myNorm import matrixNorm


if __name__ == "__main__":
    # Check processor preference
    device = fetch_device("gpu", set_default=True)
    # Configure log file
    timestamp = tStamp()
    printL = printLog(f"./_outputs/{timestamp}.log")

    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    trainset = torchvision.datasets.MNIST(root='./_data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./_data', train=False,
                                         download=True, transform=transform)

    # Init NN
    net = Net(device=device)

    # Train the net
    norm = matrixNorm(2, 2, row=True)
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.01,
                     reg=norm, reglayer=4)
    net.trainNet(trainset, epoch=5, batchsize=64,
                 optimizer=optimizer, workers=0, loger=printL)
    net.saveModel(f"./_outputs/{timestamp}.model")

    # Evaluate model
    pred, truth = evalNN.evalAllCLF(trainset, net, 512)
    evalNN.showCM(
        evalNN.calCM(pred, truth, 10), tags=trainset.classes,
        printer=printL)

    pred, truth = evalNN.evalAllCLF(testset, net, 512)
    evalNN.showCM(
        evalNN.calCM(pred, truth, 10), tags=testset.classes,
        printer=printL)
