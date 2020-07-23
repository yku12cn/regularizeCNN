"""
    This module is a wrap of nn.Module
    including save and load feature
    you should only inherit this class and define:
        NNstruct(self) : your NN structure and how to initialize your weights
        forward(self, x) : forward function for your NN
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import os
from pathlib import Path
import inspect

import torch
import torch.nn as nn
from torch.hub import tqdm
from torch.optim import Adam

from ykuUtils import stripCode, printLog


class simpleNN(nn.Module):
    r"""A Refactoring of nn.Module
        you can set device and filename if you want to load existed model

    Args:
        device ([type], optional): "gpu" or "cpu". Defaults to None.
            by default, device point to torch._C._get_default_device()
        filename ([type], optional): path to existing model. Defaults to None.
            if filename is presented, saved weights will be loaded from file
    """
    def NNstruct(self):
        r"""You should always override this function!
            Define only each layer of your NN with
            torch.nn classes
            You can also define how to initialize your weights by default
            if a filename is presented in the constructor, weights will
            be loaded from that file
        """
        raise NotImplementedError("You should override NNstruct(self)!")

    def __init__(self, device=None, filename=None):
        super(simpleNN, self).__init__()
        self.NNstruct()
        if device and isinstance(device, torch.device):
            self.to(device)
        if filename:  # load model from file if filename presented
            self.loadModel(filename)

    def saveModel(self, filename):
        r"""save current model to file

        Args:
            filename (str/Path): path to model
        """
        filename = Path(filename)
        print("Saving model to file: \"%s\"..." % (filename))
        if not filename.parent.exists():
            Path.mkdir(filename.parent, parents=True)
        torch.save(self.state_dict(), filename)

    def loadModel(self, filename):
        r"""reload saved model from file

        Args:
            filename (str/Path): path to model
        """
        device = next(self.parameters()).device
        print("Loading model from file: \"%s\"..." % (filename))
        filename = Path(filename)
        try:
            self.load_state_dict(torch.load(filename, map_location=device))
        except RuntimeError as error:
            print("""!!!!!!!!!!!!!!
Loading failed, the model you are trying to\
load doesn't match your current definition.
To proceed, you may want to check the \'Model definition\' \
section in the .log file related to %s
!!!!!!!!!!!!!!
                  """ % (filename))
            raise SystemExit(error)

    @torch.enable_grad()
    def trainNet(self, dataset, epoch=1, batchsize=10, workers=4, logPE=10,
                 optimizer=None, criterion=None, loger=print):
        r"""Train your net with decency

        Args:
            dataset (torch dataset or dataloader): Data for training.
            epoch (int, optional): Defaults to 1.
            batchsize (int, optional): Defaults to 10.
            workers (int, optional): Multithread loader. Defaults to 4.
            logPE (int, optional): Number of logs per epoch. Defaults to 10.
            optimizer (torch.optim.optimizer): Defaults to default Adam.
            criterion (torch.nn.lose): Defaults to nn.CrossEntropyLoss().
            loger (print like function): Could be a custom print function.

        .. note::
            If a dataloader is set, batchsize and workers will be neglected.
        """
        device = next(self.parameters()).device  # fetch where is the model
        # Prepare dataset
        if device == torch.device("cpu"):
            pind = True
        else:
            pind = bool(workers)

        if isinstance(dataset, torch.utils.data.Dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                                 pin_memory=pind, shuffle=True,
                                                 num_workers=workers)
        elif isinstance(dataset, torch.utils.data.DataLoader):
            loader = dataset
        else:
            raise ValueError("Invalid training data. Should be either \
a torch dataset or a torch dataloader")

        # Assign default optimizer
        if not optimizer:
            optimizer = Adam(self.parameters())

        # Assign default criterion
        if not criterion:
            criterion = nn.CrossEntropyLoss()

        isTrain = self.training  # mark model's original state
        self.train()  # set model into training mode

        # Log info
        brief = "\n".join(f"{k}: {v}" for k, v in optimizer.defaults.items())
        logstr = f"""Start training...
==========Training Brief===========
Epoch: {epoch}
Batch size: {loader.batch_size}
Num of loaders: {workers}
Loss func: {type(criterion).__name__}
Optimizer: {type(optimizer).__name__}
==========Optimizer Brief==========
{brief}
===========Dataset Brief===========
{loader.dataset}
=========Model definition=========
{self.printFUN()}
=================================="""
        if isinstance(loger, printLog):
            loger(logstr, t=True)
        else:
            loger(logstr)

        # Start training
        # Build progress bar
        pb = tqdm(total=epoch*len(dataset), desc=f"Training NN on {device}: ",
                  leave=True, ascii=(os.name == "nt"), mininterval=0.3)
        logPE = int(max(len(loader)/logPE, 1))
        for ep in range(epoch):
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                self.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (logPE == 1) or (i % logPE == logPE - 1):
                    logstr = f"\
Ep {ep+1}/{epoch} - trained {(i+1)*loader.batch_size}:  \
loss_{round(running_loss / logPE, 3)}"
                    if isinstance(loger, printLog):
                        loger(logstr, redirect=True, t=True)
                    if getattr(pb, "write", None):
                        pb.write(logstr)
                    running_loss = 0.0
                pb.update(len(labels))
        pb.close()
        if getattr(pb, "format_interval", None):
            logstr = pb.format_interval(pb.format_dict["elapsed"])
            logstr = f"Done training in {logstr}."
        else:
            logstr = "Done training."

        if isinstance(loger, printLog):
            loger(logstr, t=True)
        else:
            loger(logstr)

        # reset model to its original state
        if not isTrain:
            self.eval()

    def printFUN(self):
        r"""return the source code of current NN

        Returns:
            str : the code of current NN
        """
        structure = stripCode(inspect.getsource(self.NNstruct))
        ffunction = stripCode(inspect.getsource(self.forward))
        return structure + "\n" + ffunction
