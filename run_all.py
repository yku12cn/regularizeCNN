"""
    Run whole demo
    Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""
import ykuTorch.myDevice as myDevice
import ykuTorch.evalNN as evalNN
from cnnNet import Net

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
from myAdam import Adam


if __name__ == "__main__":
    device = myDevice.fetch_device("gpu")  # Check processor preference

    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Init NN
    # net = Net(device=device, filename="aa.model")
    net = Net(device=device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=400,
                                              pin_memory=True, shuffle=True,
                                              num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.0005, weight_decay=0.005)
    net.train()
    for epoch in range(60):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')
    net.saveModel("test.model")

    # Evaluate model
    correct, total = evalNN.evalAllCLF(trainset, net, 1000)
    print('Train accuracy: %.2f%%' % (100 * correct / total))

    correct, total = evalNN.evalAllCLF(testset, net, 1000)
    print('Test accuracy: %.2f%%' % (100 * correct / total))
