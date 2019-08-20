"""
    This module is a wrap of nn.Module
    including save and load feature
    you should only inherit this class and define:
        NNstruct(self) : your NN structure and how to initialize your weights
        forward(self, x) : forward function for your NN
    Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""
import torch
import torch.nn as nn
from pathlib import Path


class simpleNN(nn.Module):
    def NNstruct(self):
        r"""You should always override this function!
            Define only each layer of your NN with
            torch.nn classes
            You can also define how to initialize your weights by default
            if a filename is presented in the constructor, weights will
            be loaded from that file
        """
        raise NotImplementedError("You should override NNstruct(self)!")

    def __init__(self, device=torch.device("cpu"), filename=None):
        r"""A wrap version of nn.Module __init__
            you can set device and filename if you want to load existed model

        Arguments:
            __init__(device, filename):
            device = <class 'torch.device'>
            filename = "path to model"

        .. note::
            by default, device point to cpu
            if filename is presented, saved weights will be loaded from file
        """
        super(simpleNN, self).__init__()
        self.NNstruct()
        self.to(device)
        if filename:  # load model from file if filename presented
            self.loadModel(filename)

    def saveModel(self, filename):
        r"""save current model to file

        Arguments:
            saveModel(filename):
            filename = "path to model"
        """
        print("Saving model to file: \"%s\"..." % (filename))
        filename = Path(filename)
        torch.save(self.state_dict(), filename)
        print("Done")

    def loadModel(self, filename):
        r"""reload saved model from file

        Arguments:
            saveModel(filename):
            filename = "path to model"
        """
        device = next(self.parameters()).device
        print("Loading model from file: \"%s\"..." % (filename))
        filename = Path(filename)
        self.load_state_dict(torch.load(filename, map_location=device))
        print("Done")
