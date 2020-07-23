"""
    Generate a single dirty dataset contaminated by random wrong labels
    Should only be included by contaminate.py
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""

import os
from pathlib import Path
from random import choices, shuffle

from torch.hub import tqdm

from ..VDPlus import VDPlus
from .. import setUnpack


def _genWrongLabel(inset, outfolder, ratio, vid=0):
    r"""generate a single dirty dataset contaminated by wrong labels
        **Internal use only**

    Args:
        inset (VDPlus): A VDPlus with an OrderedDict attribute "classified"
                        inset can be generated by "inset.inspectSet()"
        outfolder (Path): output directory
        ratio (float): 0~1, the ratio of the contamination
    """
    outfolder = Path(outfolder)
    # Create folder if not exist
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    # Init output set
    outset = VDPlus(str(outfolder), tags=inset.classes)
    outset.img_type = inset.img_type
    outset.classes_count = dict.fromkeys(outset.classes, 0)

    # Set progress bar
    pb = tqdm(total=len(inset), desc=f'Process "{outfolder.name}"', leave=True,
              position=vid, ascii=(os.name == "nt"), mininterval=0.3)

    # Process data, iter through each class
    for target, (tag, samples) in enumerate(inset.classified.items()):
        # Randomize samples
        samples = samples.copy()
        shuffle(samples)

        # Cal the number for changed and unchanged
        _changed = round(len(samples) * ratio)
        _unchanged = len(samples) - _changed

        # Attach unchanged part
        outset.data.extend(samples)
        outset.targets.extend([target] * _unchanged)
        outset.classes_count[tag] += _unchanged
        pb.update(_unchanged)

        # Attach changed part
        w_label = []  # Generate false candidates
        for item in range(len(outset.classes)):
            if item != target:
                w_label.append(item)

        _n_targets = choices(w_label, k=_changed)
        outset.targets.extend(_n_targets)
        for item in _n_targets:
            outset.classes_count[outset.classes[item]] += 1
            pb.update()
    pb.close()

    if isinstance(inset.data[0], Path):
        # Save output set if it's not on ram
        outset.root = inset.root
        setUnpack(outset, outfolder=outfolder, vid=vid)
    else:
        # Save output set if it is on ram
        outset.root = outfolder
        outset.makeCache(outfolder.name)
        outset.dumpMeta(outfolder.name)