"""
    utilities for dataset processing
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""

from .utils import calMD5
from .VDPlus import VDPlus, setUnpack, setPack,\
    setfromfolder, setfromPack
from .contaminate import genDirtySet
