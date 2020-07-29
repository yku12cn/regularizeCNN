"""
    General tools and helper functions
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torch


class costTracker():
    r"""Track the normalized perturbation cost.
        Created by specifing what L-p distance you want
        add perturbation by either 'call(p)' or '+= p'

    Args:
        p (int, optional): L-p distance setting. Defaults to 2.
        device (torch.device, optional): device to store
    """
    def __init__(self, p=2, device=None):
        self.p = torch.tensor(p, device=device)
        self.cost = torch.tensor(0., device=device)
        self.num = torch.tensor(0, device=device)
        self.device = self.cost.device

    @torch.no_grad()
    def __call__(self, x):
        r"""Log one perturbation in

        Args:
            x (tensor): make sure each data point align alone Dim-0.
                Even if you have only one data point, say shape(3, 28, 28),
                you should still unsqueeze it into shape(1, 3, 28, 28)

        .. note::
            This tracker will move itself to the same device as
            your input tensor.
        """
        if not self.device == x.device:
            self.to(x.device)
        x = x.detach()
        self.num.add_(x.shape[0])
        self.cost.add_(
            x.view(x.shape[0], -1).norm(p=self.p, dim=1).sum())

    def __iadd__(self, x):
        self.__call__(x)
        return self

    def __repr__(self):
        return f"[{self.cost.item():.3f} / {self.num.item()}\
 = {self.avg().item():.3f}]"

    def to(self, device):
        r"""Move to device

        Args:
            device (torch.device): move to which device
        """
        self.p = self.p.to(device)
        self.cost = self.cost.to(device)
        self.num = self.num.to(device)
        self.device = self.cost.device

    def avg(self):
        """return the average costs

        Returns:
            1-D tensor
        """
        return self.cost / self.num


@torch.enable_grad()
def genGrad(x, clf, criterion=None, device=None):
    r"""calculate gradient for each class w.r.t input data

    Args:
        x (tensor): input data
        clf (callable obj): the classifier you want to test
        criterion (nn.loss, optional): loss function. Defaults to None.
            !! Be careful with the reduction setting in loss function.
            !! Reduction setting will affect gradients accordingly.
        device (torch.device, optional): test on which device. This should
            be on the same device as the 'clf' does.

    Returns:
        tensor: gradient for each class
            0-dim represents each class accordingly
    """
    x = x.detach().to(device).requires_grad_(True)
    output = clf(x)

    c_num = output.shape[1]
    p_target = torch.arange(0, c_num, device=device).unsqueeze_(1)

    grads = torch.zeros((c_num,) + x.shape, device=device)
    for target, slot in zip(p_target, grads):
        if criterion:
            loss = criterion(output, target.repeat(output.shape[0]))
        else:
            # Cal gradient of direct output
            loss = output[:, target].sum()

        loss.backward(retain_graph=True)
        slot.add_(x.grad)
        x.grad.zero_()

    return grads, output.detach()
