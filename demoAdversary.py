"""
    A demo showcase adversary attack
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import random

import torchvision.transforms as transforms
import torchvision
import torch

from ykuTorch import fetch_device, evalNN
from ykuUtils import printLog, tStamp
from demoCNN import Net
from advLib import deepfoolGD


if __name__ == "__main__":
    device = fetch_device("cpu")  # Check processor preference
    # Configure log file

    printL = printLog("./_outputs/%s.log" % tStamp())

    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    testset = torchvision.datasets.MNIST(root='./_data', train=False,
                                         download=True, transform=transform)

    # Init NN
    net = Net(device=device, filename="./demoCNN.dmodel")
    tagdic = testset.classes

    while True:
        n = random.randint(0, 9999)
        printL(f"Image number {n}")
        img, tag = testset[n]
        img = img.unsqueeze(0)
        tag = torch.tensor([tag])

        pred, _ = evalNN.evalCLF((img, tag), net, device=device)
        printL(f"Image is '{tagdic[tag]}', NN predicts '{tagdic[pred]}'")

        printL("Generating adversary")
        delta, _ = deepfoolGD((img, tag), net, device=None,
                              rate=0.2, overshoot=0.0001)
        n_img = img.add(delta)

        pred, _ = evalNN.evalCLF((n_img, tag), net, device=device)
        printL(f"Tag is '{tagdic[tag]}', NN predicts '{tagdic[pred]}'")

        evalNN.showImg(
            img=torch.cat([img, delta, n_img], dim=0),
            label=f"'{tagdic[tag]}' + perturbation = '{tagdic[pred]}'",
        )
        if input("next? (y/n): ") == "n":
            break
