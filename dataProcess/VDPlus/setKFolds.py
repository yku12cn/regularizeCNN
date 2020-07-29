"""
    Generate a K-Fold enabled set out of an existing VDPlus set.
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from collections import OrderedDict

from . import VDPlus


class setKFold(VDPlus):
    r"""Generate a K-Fold enabled set out of an existing VDPlus set.

    Args:
        inset (VDPlus set): original set.
        folds (int): how many folds?
        transform (torchvision.transforms, optional): for image.
        target_transform (torchvision.transforms, optional): for lable.
    """
    def __init__(self, inset, folds, transform=None, target_transform=None):
        if not isinstance(inset, VDPlus):
            raise TypeError("Original set has to be a VDPlus set")

        if not transform:
            transform = inset.transform

        if not target_transform:
            target_transform = inset.target_transform

        super(setKFold, self).__init__(
            inset.root, tags=inset.classes, transform=transform,
            target_transform=target_transform
        )

        self.img_type = inset.img_type

        inset.inspectSet()  # Sort input set
        # Store sorted original set
        self.olib = inset.classified.copy()

        # Gen segment points
        self.segment = OrderedDict.fromkeys(self.classes)
        self.folds = folds
        for tag, samples in self.olib.items():
            seglen = len(samples) / self.folds
            ind = _high = 0
            self.segment[tag] = []
            while ind < len(samples):
                _low = _high
                _high = min(round(ind + seglen), len(samples))
                self.segment[tag].append((_low, _high))
                ind += seglen

        self.select(k=0, train=True)

    def select(self, k=0, train=False):
        r"""Select # k subset

        Args:
            k (int, optional): Which set. Defaults to 0.
            train (bool, optional): Train or Test. Defaults to False.
        """
        if k < 0 or k > self.folds:
            raise IndexError(f"k out of range (0 - {self.folds}).")

        self.data = []
        self.targets = []
        for key, (tag, seg) in enumerate(self.segment.items()):
            if train:
                seg = self.olib[tag][:seg[k][0]] + self.olib[tag][seg[k][1]:]
            else:
                seg = self.olib[tag][seg[k][0]:seg[k][1]]

            self.data.extend(seg)
            self.targets.extend([key]*len(seg))
            self.classes_count[tag] = len(seg)
            self.classified[tag] = seg
