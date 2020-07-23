"""
    An enhanced VisionDataset class and its variants
    Designed to ease the pain of loading and modifying TorchVision data
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from .VDPlus import VDPlus
from .datapacking import loadFiles, setUnpack, setPack

# Varients of VDPlus
from .setfromfolder import setfromfolder
from .setfromPack import setfromPack
