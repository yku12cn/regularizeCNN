"""
    find adversary_demo
    Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""
import ykuTorch.myDevice as myDevice
import ykuTorch.evalNN as evalNN
from cnnNet import Net
import advLib


import torchvision.transforms as transforms
import torchvision
import torch

import random


if __name__ == "__main__":
    device = myDevice.fetch_device("gpu")  # Check processor preference

    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Init NN
    net = Net(device=device, filename="test.model")

    while True:
        n = random.randint(0, 9999)
        # n = 4586
        print(n)
        data = testset[n]  # 2231 is wrong cat

        # test with NN
        prid1, co = evalNN.evalCLF(data, net)
        print("NN think it is: %s. It is %s." %
              (classes[prid1[0]], "correct" if co[0] == 1 else "wrong"))

        # generate adversary
        adversary = advLib.genAdv(data, net)

        # test with NN
        prid2, co = evalNN.evalCLF([adversary, data[1]], net)
        print("NN think it is: %s. It is %s." %
              (classes[prid2[0]], "correct" if co[0] == 1 else "wrong"))

        adversary = adversary.to('cpu')
        diff = (adversary - data[0])

        bb = torch.cat(
            [data[0].unsqueeze(0), diff.unsqueeze(0), adversary.unsqueeze(0)])
        evalNN.showImg(bb, "pre %s        after %s" %
                       (classes[prid1[0]], classes[prid2[0]]))
