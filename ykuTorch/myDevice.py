"""
    This module is used to check available device
    Only support CUDA
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import torch


def _setCPU(set_default=False):
    device = torch.device("cpu")
    print("\"CPU\" is selected")
    if set_default:
        torch.set_default_tensor_type(torch.FloatTensor)
        print("Set default device as CPU")
    return device


def _setGPU(which, set_default=False):
    if torch._C._cuda_getDeviceCount() <= which:
        which = torch._C._cuda_getDeviceCount() - 1
    device = torch.device("cuda", which)
    print(f"\"{torch.cuda.get_device_name(which)}\" is selected")
    if set_default:
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print(f"Set default device as \"{device}\"")

    return device


def fetch_device(prefer="gpu", which=0, set_default=False):
    r"""Return available device as <class 'torch.device'>

    Args:
        prefer (str, optional): prefer "gpu" or "cpu". Defaults to "gpu".
        which (int, optional): index of gpu. Defaults to 0.
        set_default (bool, optional): True if you prefer not to do
                                      it everytime. Defaults to False.

    Returns:
        torch.device: available device

    .. note::
            It will return you preferred device if available
    """
    if prefer == "cpu":
        return _setCPU(set_default)

    if prefer == "gpu":
        if torch.cuda.is_available():
            return _setGPU(which, set_default)
        print("System doesn't support CUDA, use CPU instead")
        return _setCPU(set_default)

    raise ValueError("use \'gpu\' or \'cpu\' as argument")


if __name__ == "__main__":
    dev = fetch_device("gpu")
    print(dev)
