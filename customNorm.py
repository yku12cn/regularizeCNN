"""
    This where you want to implement the deritive of your Norm.
"""
import torch


def cNorm(p):
    p = torch.div(p, (torch.norm(p, p=2) + 0.01))
    return p
