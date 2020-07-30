"""
    Convert small torchvision sets to full-fledged VDPlus sets and
    generate caches for quicker load
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from pathlib import Path
import torchvision
from dataProcess import setUnpack, setfromfolder

# list sets you want to digest here
#   tvSet (torchvision.datasets): the constructor/class of that set
#   root (str): target root directory.
#   newname (str, optional): rename set. Defaults to class name.
#   tags (list of str, optional): tags for each class.
__jobs = [
    {"tvSet": torchvision.datasets.MNIST,
     "root": './_data',
     "newname": None,
     "tags": None,
     },
    {"tvSet": torchvision.datasets.FashionMNIST,
     "root": './_data',
     "newname": None,
     "tags": None,
     },
    {"tvSet": torchvision.datasets.CIFAR10,
     "root": './_data',
     "newname": None,
     "tags": None,
     },
]


def _cleanname(filename):
    for char in r'<>:"/\|?*':
        filename = filename.replace(char, '_')
    return filename


def digestSet(tvSet, root='./_data', newname=None, tags=None):
    r"""Digest a small torchvision set

    Args:
        tvSet (torchvision.datasets): the constructor/class of that set
        root (str, optional): target directory. Defaults to './_data'.
        newname (str, optional): rename set. Defaults to class name.
        tags (list of str, optional): tags for each class.
    """
    inset = tvSet(root=root, train=True, download=True)

    if not newname:
        newname = tvSet.__name__ + "_Unpack"

    if not tags:
        tags = inset.classes
        for i, tag in enumerate(tags):
            tags[i] = _cleanname(tag)

    print(newname, tags)

    _newroot = Path(root) / newname

    # Gen train set
    inset.classes = tags
    setUnpack(inset, _newroot / 'train')
    outset = setfromfolder(_newroot / 'train', tags=tags, ram=True)
    outset.makeCache('train')
    outset.dumpMeta('train')
    outset = setfromfolder(_newroot / 'train', tags=tags, ram=False)
    print(outset)

    # Gen test set
    inset = tvSet(root=root, train=False, download=True)
    inset.classes = tags
    setUnpack(inset, _newroot / 'test')
    outset = setfromfolder(_newroot / 'test', tags=tags, ram=True)
    outset.makeCache('test')
    outset.dumpMeta('test')
    outset = setfromfolder(_newroot / 'test', tags=tags, ram=False)
    print(outset)

    # Save tags
    print(tags, file=open(_newroot / 'classes.txt', 'w'))


if __name__ == '__main__':
    for job in __jobs:
        digestSet(**job)
