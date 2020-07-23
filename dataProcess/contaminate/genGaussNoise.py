"""
    Generate a single dirty dataset contaminated by Gaussian noise
    Should only be included by contaminate.py
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""

import os
from pathlib import Path

import torch
from torch.hub import tqdm
import torchvision.transforms.functional as F

from ..VDPlus import VDPlus


def _genGaussNoise(inset, outfolder, var=0.1, vid=0):
    r"""generate a single dirty dataset contaminated by Gaussian noise
        **Internal use only**

    Args:
        inset (VDPlus): A VDPlus object, your input set
        outfolder (Path): output directory
        var (float): 0~1, the variance of the noise
    """
    outfolder = Path(outfolder)
    # Create folder if not exist
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    # Init output set
    outset = VDPlus(str(outfolder), tags=inset.classes)
    outset.img_type = inset.img_type
    outset.classes_count = inset.classes_count
    outset.targets = inset.targets

    # Set progress bar
    pb = tqdm(total=len(inset), desc=f'Process "{outfolder.name}"', leave=True,
              position=vid, ascii=(os.name == "nt"), mininterval=0.3)

    _ram = not isinstance(inset.data[0], Path)

    # Process data
    for (_img, _), _ori in zip(inset, inset.data):
        _img = F.to_tensor(_img)
        _img.add_(torch.randn(_img.size()), alpha=var)
        _img.clamp_(0, 1)
        _img = F.to_pil_image(_img)
        if _ram:
            outset.data.append(_img)
        else:
            _img.save(outfolder / _ori)

        pb.update()
    pb.close()

    if _ram:
        # Save output set if it is on ram
        outset.makeCache(outfolder.name)
        outset.dumpMeta(outfolder.name)
