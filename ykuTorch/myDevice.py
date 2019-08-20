"""
    This module is used to check available device
    Only support CUDA
    Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""
import torch


def fetch_device(prefer="gpu", which=0):
    r"""Return available device as <class 'torch.device'>

    Arguments:
        fetch_device(prefer="gpu" or "cpu", which=<int>):
        set device preference and index if gpu is select

    .. note::
            It will return you prefer device if available
            by default, it prefer gpu of index 0.
    """
    if prefer == "cpu":
        print("Set device as CPU")
        return torch.device("cpu")
    elif prefer == "gpu":
        if torch.cuda.is_available():
            if torch._C._cuda_getDeviceCount() > which:
                print("Set device as", torch.cuda.get_device_name(which))
                return torch.device("cuda", which)
            else:
                which = torch._C._cuda_getDeviceCount() - 1
                print("Set device as", torch.cuda.get_device_name(which))
                return torch.device("cuda", which)
        else:
            print("System doesn't support CUDA, use CPU instead")
            return torch.device("cpu")
    else:
        raise ValueError("use \'gpu\' or \'cpu\' as argument")


if __name__ == "__main__":
    device = fetch_device("gpu")
    print(device)
