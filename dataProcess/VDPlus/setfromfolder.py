"""
    Load data stored in a folder as individual images
    Will create cache files to speed up loading
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from . import VDPlus, loadFiles
from ..utils import detectImgMode, inspectFileList, inspectFileMeta


class setfromfolder(VDPlus):
    r"""generate dataset from folder
        sample naming scheme "[tag]_xxxx.ext"

    Args:
        root (str/Path): directory of your data
        tags (list of str, optional): define the list of your tags
            if left undefined, a list will be generated
            on-the-fly. Thus, its sequence may differ from
            what you want. Defaults to None.
        IMmode (str, optional): PIL built in modes ('L','P','RGB'...)
        ram (bool, optional): load to ram or not. Defaults to False.
        transform (torchvision.transforms, optional): for image
        target_transform (torchvision.transforms, optional): for lable
    """
    def __init__(self, root, tags=None, IMmode=None, ram=False,
                 transform=None, target_transform=None):
        super(setfromfolder, self).__init__(
            root, transform=transform, tags=tags,
            target_transform=target_transform
        )
        # Detect image mode if not given
        if IMmode:
            self.img_type = IMmode
        else:
            self.img_type = detectImgMode(inspectFileList(root))

        # list all files
        filelist = inspectFileList(root, relative=True)

        # Try loading from cache
        if not self._quickLoad(filelist, ram):
            loadFiles(self, filelist, ram)
            if ram:
                self.makeCache(
                    # Pack all data. Use meta data as snapshot
                    "_datacache_/data.vdcache", inspectFileMeta(self.root)
                )
            else:
                self.makeCache(
                    # Pack only the file list
                    "_datacache_/list.vdcache", filelist
                )
            self.dumpMeta("_datacache_/setmeta.vdmeta")

    def _quickLoad(self, filelist, ram):
        print("Checking Cache...")
        _meta = self.loadMeta("_datacache_/setmeta.vdmeta")
        if not _meta:
            print("No cache found")
            return False
        _type, _classes, _count = _meta

        if (_type != self.img_type) and ram:
            print("Color depth mismatch, abort quickload")
            return False

        if self.classes and (self.classes != _classes):
            print("Tags mismatch, abort quickload")
            return False

        # Try loading cached data
        if ram:
            print("Loading data to ram")
            _load = self.loadCache(
                "_datacache_/data.vdcache", inspectFileMeta(self.root)
            )
        else:
            print("Loading data list")
            _load = self.loadCache(
                "_datacache_/list.vdcache", filelist
            )

        if _load:
            # If loading success, assign statistics.
            self.classes_count = _count
            self.classes = _classes
        else:
            print("Cache mismatch, abort quickload")
        return _load
