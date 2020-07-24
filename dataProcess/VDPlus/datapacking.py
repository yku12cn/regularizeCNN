"""
    utilities for data packing / loading
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""

from pathlib import Path
import os
import re

from PIL import Image, UnidentifiedImageError
from torch.hub import tqdm

from .VDPlus import VDPlus
from ..utils import detectImgMode, inspectFileList


def loadFiles(obj, filelist, ram=False, vid=0):
    r"""load all files in the specified list to a VDPlus set
        sample naming scheme "[tag]_xxxx.ext"

    Args:
        obj (VDPlus): VDPlus set for stroing the loaded data
        filelist (list of Path): images to be loaded. Relative path to obj.root
        ram (bool, optional): load to ram or not. Defaults to False.
        vid (int, optional): position of progress bar. Defaults to 0.
    """

    if not isinstance(obj, VDPlus):
        raise TypeError("'obj' has to be a VDPlus object")

    if ram and not obj.img_type:
        raise ValueError("obj.img_type must be set before loading to ram")

    obj.data = []
    obj.targets = []
    dict_target = {}  # Tracking tags
    obj.classes_count = {}  # Count the distribution of the data

    _root = Path(obj.root)

    for tag in obj.classes:
        # Associate tags with indexes if tag-list is defined
        dict_target[tag] = len(dict_target)
    obj.classes_count = dict.fromkeys(obj.classes, 0)

    # Build progress bar
    pb = tqdm(total=len(filelist), desc="Loading: ", position=vid,
              leave=True, ascii=(os.name == "nt"), mininterval=0.3)
    for file in filelist:
        # Check file's name. Looking for tag
        ftag = re.findall(r"\[(.*?)\]|$", file.name)[0]
        if not ftag:  # Jump over illegal file
            continue
        try:  # Verify data. Jump over bad files
            testimg = Image.open(_root / file)
        except UnidentifiedImageError:
            continue

        # Load to ram or not
        if ram:
            # convert always create a copy
            obj.data.append(testimg.convert(obj.img_type))
        else:
            obj.data.append(file)
        testimg.close()

        if not obj.classes:
            if ftag not in dict_target:
                dict_target[ftag] = len(dict_target)
                obj.classes_count[ftag] = 0
        obj.targets.append(dict_target[ftag])
        obj.classes_count[ftag] += 1

        pb.update()
    pb.close()

    # Generate tag list
    if not obj.classes:
        obj.classes = [None] * len(dict_target)
        for tag, value in dict_target.items():
            obj.classes[value] = tag


def setUnpack(inset, outfolder="./out", fmode="bmp", vid=0):
    r"""Unpack a VisionDataset

    Args:
        inset (trochvision Dataset): set you want to unpack
        outfolder (str, optional): output directory. Defaults to "./out".
        fmode (str, optional): output format. Defaults to "bmp".
        vid (int, optional): position of progress bar. Defaults to 0.
    """

    # Backup inset transforms and remove inset transform temporarily
    _bk_trans, _bk_t_trans, _bk_s_trans =\
        inset.transform, inset.target_transform, inset.transforms
    inset.transform = inset.target_transform = inset.transforms = None

    tags = inset.classes if hasattr(inset, 'classes') else []

    output = Path(outfolder)  # Unpack destination
    ind_len = len(str(len(inset)))  # lenth of index string

    # Create folder if not exist
    if not output.exists():
        output.mkdir(parents=True)

    # Build index container
    inds = [1] * len(tags) if tags else {}  # assign tags on the go

    # Build progress bar
    pb = tqdm(total=len(inset), desc="Unpacking: ", position=vid,
              leave=True, ascii=(os.name == "nt"), mininterval=0.3)

    # Unpacking
    for img, target in inset:
        if tags:
            fname = f"[{tags[target]}]_{inds[target]:0{ind_len}}.{fmode}"
        else:
            if target not in inds:
                inds[target] = 1
            fname = f"[{target}]_{inds[target]:0{ind_len}}.{fmode}"
        img.save(output / fname)
        inds[target] += 1
        pb.update()

    pb.close()

    # Set back inset transform
    inset.transform, inset.target_transform, inset.transforms =\
        _bk_trans, _bk_t_trans, _bk_s_trans


def setPack(root, tags=None, IMmode=None, delete=False, vid=0):
    r"""Make packed images from folder

    Args:
        root (str/Path): directory of your data
        tags (list of str, optional): define the list of your tags
            if left undefined, a list will be generated
            on-the-fly. Thus, its sequence may differ from
            what you want. Defaults to None.
        IMmode (str, optional): PIL built in modes ('L','P','RGB'...)
        delete (bool, optional): delete after packing. Defaults to False.
        vid (int, optional): position of progress bar. Defaults to 0.

    Returns:
        [bool]: Successful or not
    """
    pack = VDPlus(root=root, tags=tags)

    # list all files
    filelist = inspectFileList(root)

    # Detect image mode if not given
    pack.img_type = IMmode if IMmode else detectImgMode(filelist)

    if not pack.img_type:
        # No images found
        return False

    _root = Path(root)
    filelist = [item.relative_to(_root) for item in filelist]

    # Loading files
    loadFiles(pack, filelist, ram=True, vid=vid)

    if len(pack) == 0:
        # No valid image
        return False

    # Save package
    pack.makeCache(_root.name)
    pack.dumpMeta(_root.name)

    # remove files
    if delete:
        pb = tqdm(total=len(filelist), desc="Deleting: ", position=vid,
                  leave=True, ascii=(os.name == "nt"), mininterval=0.3)
        for file in filelist:
            file.unlink()
            pb.update()
        pb.close()

    return True
