"""
    An enhanced VisionDataset class. Shall not be used directly
    This class is designed to ease the pain of loading data from folder
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from pathlib import Path
import pickle
from collections import OrderedDict
from collections.abc import Iterable
import random

from PIL import Image
from torchvision.datasets.vision import VisionDataset

from .. import calMD5


class VDPlus(VisionDataset):
    r"""An enhanced VisionDataset class. Shall not be used directly
        This class is designed to ease the pain of loading data from folder
        sample naming scheme "[tag]_xxxx.ext"

    Args:
        root (str/Path): directory of your data
        tags (list of str, optional): define the list of your tags
            if left undefined, a list will be generated
            on-the-fly. Thus, its sequence may differ from
            what you want.
        transform (torchvision.transforms, optional): for image.
        target_transform (torchvision.transforms, optional): for lable.
    """
    def __init__(self, root, tags=None, transform=None, target_transform=None):
        if not Path(root).is_dir():
            raise ValueError("Invalid directory")

        super(VDPlus, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

        if isinstance(tags, list):
            self.classes = tags.copy()
        else:
            self.classes = []

        # Entries to be determined:
        self.img_type = None
        self.data = []
        self.targets = []
        self.classes_count = {}
        self.classified = {}  # won't be set until self.inspectSet is called

    def __getitem__(self, index):
        r"""overload __gititem__
        """
        # Handle slice style
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))

        # Handle iter
        if isinstance(index, Iterable):
            _data = []
            _targets = []
            for i in index:
                _d, _t = self[i]
                _data.append(_d)
                _targets.append(_t)
            return _data, _targets

        # Handle normal index
        if index < 0:
            index += len(self)

        img, target = self.data[index], int(self.targets[index])

        if isinstance(img, Path):
            # Load from disk
            imgL = Image.open(Path(self.root) / img)
            if imgL.mode != self.img_type:
                img = imgL.convert(self.img_type)
            else:
                img = imgL.copy()
            imgL.close()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        data_dist = []
        i = 0
        for key in self.classes:
            data_dist.append(f"|   |#{i} - [{key}]: {self.classes_count[key]}")
            i += 1
        data_dist = '\n'.join(data_dist)

        trans_repr = repr(self.transforms)
        trans_repr = trans_repr.replace("\n", "\n|   |   ")

        return f"""Dataset from "{self.root}":
|   Image type: {self.img_type}
|   Number of datapoints: {self.__len__()}
|   Data distribution:
{data_dist}
|
|   Transforms: {trans_repr}
|"""

    def _prePath(self, c_name, ext):
        c_name = Path(self.root) / c_name
        if c_name.suffix != ext:
            c_name = c_name.with_suffix(c_name.suffix + ext)
        return c_name

    def makeCache(self, c_name, snapshot=None):
        r"""Save "self.data" and "self.target" to cache file.
            The file name will be "c_name.vdcache" and
            stored under the root folder

        Args:
            c_name (str/Path): name of the cache file.
                               example: "_cache_/data.vdcache"
            snapshot (list, optional): After dump, a MD5 thumbnail will be
                created out of this info. File list or metadatas are
                recommended. If not specified, snapshot won't be created.

        .. note::
            A good snapshot may come handy when you reload from
            cache and check if the cache is obsolete
        """
        c_name = self._prePath(c_name, ".vdcache")
        # Create folder if not exists
        if not c_name.parent.exists():
            c_name.parent.mkdir(parents=True)

        # Dump data
        pickle.dump(
            (self.data, self.targets),
            open(c_name, "wb")
        )

        # Create snapshot
        if snapshot:
            c_name = c_name.with_suffix(".vdsnap")
            pickle.dump(calMD5(snapshot), open(c_name, "wb"))

    def loadCache(self, c_name, snapshot=None):
        r"""Load "self.data" and "self.target" from cache.
            The cache file should be stored under the root folder,
            and its extension should be ".vdcache"

        Args:
            c_name (str/Path): name of the cache file
                               example: "_cache_/data.vdcache"
            snapshot (list, optional): Additional info used to determine
                whether the cache is obsolete. If not specified, cache
                will be loaded without check.

        Returns:
            bool: whether loading is successful

        .. note::
            Please refer makeCache() for how to use snapshot
        """
        c_name = self._prePath(c_name, ".vdcache")
        if not c_name.is_file():
            return False

        if snapshot:  # Check snapshot
            s_name = c_name.with_suffix(".vdsnap")
            if not s_name.is_file():
                return False

            if calMD5(snapshot) != pickle.load(open(s_name, "rb")):
                return False

        self.data, self.targets = pickle.load(open(c_name, "rb"))
        return True

    def dumpMeta(self, c_name):
        r"""Save "self.img_type", "self.classes" and "self.classes_count"
            The file name will be "c_name.vdmeta" and
            stored under the root folder

        Args:
            c_name (str/Path): name of the dump file
                               example: "_cache_/data.vdmeta"
        """
        c_name = self._prePath(c_name, ".vdmeta")
        # Create folder if not exists
        if not c_name.parent.exists():
            c_name.parent.mkdir(parents=True)

        # Dump data
        pickle.dump(
            (self.img_type, self.classes, self.classes_count),
            open(c_name, "wb")
        )

    def loadMeta(self, c_name):
        r"""Load "self.img_type", "self.classes" and "self.classes_count"
            The file should be stored under the root folder,
            and its extension should be ".vdmeta"
            variables won't be assigned, instead, will be returned.

        Args:
            c_name (str/Path): name of the dump file
                               example: "_cache_/data.vdmeta"

        Returns:
            tuple: (self.img_type, self.classes, self.classes_count)
        """
        c_name = self._prePath(c_name, ".vdmeta")
        if not c_name.is_file():
            return False

        return pickle.load(open(c_name, "rb"))

    def inspectSet(self):
        r"""inspect all samples, assign an 'OrderedDict' of classified samples
            to self.classified. Will also update self.classes_count.
        """
        self.classified = OrderedDict.fromkeys(self.classes)
        self.classes_count = dict.fromkeys(self.classes, 0)

        for img, target in zip(self.data, self.targets):
            _sublist = self.classified[self.classes[target]]
            if _sublist:
                _sublist.append(img)
            else:
                self.classified[self.classes[target]] = [img]
            self.classes_count[self.classes[target]] += 1

        # Update self.classes_count
        for key, _sublist in self.classified.items():
            self.classes_count[key] = len(_sublist)

    def shuffle(self):
        r"""shuffle all data
        """
        _randindx = list(range(len(self)))
        random.shuffle(_randindx)
        self.data = [self.data[x] for x in _randindx]
        self.targets = [self.targets[x] for x in _randindx]
