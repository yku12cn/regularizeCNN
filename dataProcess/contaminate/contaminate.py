"""
    utilities for generating bad samples
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""

from pathlib import Path

from ykuUtils import runParallelTqdm

from .. import VDPlus
from .genWrongLabel import _genWrongLabel
from .genGaussNoise import _genGaussNoise

_methods = {"wl": _genWrongLabel,
            "GaussNoise": _genGaussNoise
            }


def genDirtySet(inset, outfolder, method, kargs, workers=4):
    r"""generate dirty dataset(s) with given methods

    Args:
        inset (VDPlus obj): the original dataset
        outfolder (str/Path): output directory
        method (str): contamination method:
                        "wl" : wrong label
                        "GaussNoise": Gaussian noise
        kargs (dict or list of dict): arguments for specified method.
                         should be a dictionary of keyword arguments
                         or a list of such dictionaries if you want
                         to generate multiple sets.
        workers (int, optional): parallel workers. Defaults to 4.
    """
    if not isinstance(inset, VDPlus):
        raise TypeError("inset should be a VDPlus object")
    inset.inspectSet()

    # Backup inset transforms and remove inset transform temporarily
    _bk_trans, _bk_t_trans, _bk_s_trans =\
        inset.transform, inset.target_transform, inset.transforms
    inset.transform = inset.target_transform = inset.transforms = None

    contaminateFun = _methods[method]

    outfolder = Path(outfolder)  # Output destination

    # Gen all running arguments
    for karg in kargs:
        # Generate folder name accordingly
        name = [f"{k}={v}" for k, v in karg.items()]
        name.sort()
        name = "_".join(name)
        currentfolder = (outfolder.parent /
                         f"{outfolder.name}_{method}[{name}]")

        # Create folder if not exist
        if not currentfolder.exists():
            currentfolder.mkdir(parents=True)

        karg["inset"] = inset
        karg["outfolder"] = currentfolder

    # Gen in parallel
    runParallelTqdm(contaminateFun, kargs, workers)

    # Set back inset transform
    inset.transform, inset.target_transform, inset.transforms =\
        _bk_trans, _bk_t_trans, _bk_s_trans
