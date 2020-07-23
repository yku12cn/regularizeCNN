"""
    A demo of customized norm
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torch

from ykuTorch import cNorm


class lnNorm(cNorm):
    r""" Derivative of ln norm

        lnNorm(rank=2)

        rank (int, optional): l1 ~ ln norm. Defaults to 2.
            rank=float('inf') will be infinity norm

        .. note::
            infinity norm is not correct while there is non-unique
            max values in the input tensor.

            #Bug: https://github.com/pytorch/pytorch/issues/41779
    """
    def __init__(self, rank=2, **kwargs):
        super(lnNorm, self).__init__(**kwargs)
        self.rank = rank

    def __call__(self, x):
        r"""Derivative of ln norm, given x

        Args:
            x (tensor): input vector

        Returns:
            tensor: Derivative of ln norm
        """
        # Isolate tensor
        x = x.detach().requires_grad_(True)

        # Run autograd
        with torch.enable_grad():
            _tmp = x.norm(p=self.rank)
            _tmp.backward()

        return x.grad


class matrixNorm(cNorm):
    r""" Derivative of matrix norm

        matrixNorm(rank_col=2, rank_row=2, row=True)

        rank_col/row (int): l1 ~ ln norm. Defaults to 2.
            rank=float('inf') will be infinity norm
        row (bool): calculate row or column firstly
    """
    def __init__(self, r_col=2, r_row=2, row=True, **kwargs):
        super(matrixNorm, self).__init__(**kwargs)
        self.row = int(row)
        self.r1 = r_row if self.row else r_col
        self.r2 = r_col if self.row else r_row
        self.r_col = r_col

    def __call__(self, x):
        r"""Derivative of matrix norm, given x

        Args:
            x (tensor): input vector

        Returns:
            tensor: Derivative of ln norm

        .. note::
            infinity norm is not correct while there is non-unique
            max values in the input tensor.

            #Bug: https://github.com/pytorch/pytorch/issues/41779
        """
        # Isolate tensor
        x = x.detach().requires_grad_(True)

        # Run autograd
        with torch.enable_grad():
            if len(x.shape) == 2:
                _tmp = torch.norm(x.norm(p=self.r1, dim=self.row), p=self.r2)
            else:
                # Handle 1-D bias vector
                _tmp = x.norm(p=self.r_col)

            _tmp.backward()

        return x.grad
