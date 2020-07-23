"""
    This module is a Demo of how to define a neural net
    by using simpleNN class
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""

import torch.nn as nn
import torch.nn.functional as F
from ykuTorch import simpleNN


class Net(simpleNN):
    """A Demo CNN
    """
    def NNstruct(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view((-1,) + x.shape[-3:])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
