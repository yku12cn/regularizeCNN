"""
    Convert small torchvision sets to full-fledged VDPlus sets
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from dataProcess import setfromPack, setfromfolder, genDirtySet

# Put your settings here:
#   inset (VDPlus obj): the original dataset
#   outfolder (str/Path): output directory
#   method (str): contamination method:
#                 "wl" : wrong label
#                 "GaussNoise": Gaussian noise
#   kargs (dict or list of dict): arguments for specified method.
#                  should be a dictionary of keyword arguments
#                  or a list of such dictionaries if you want
#                  to generate multiple sets.
#   tags (list of str, optional): define the list of your tags
#       if left undefined, a list will be generated
#       on-the-fly. Thus, its sequence may differ from
#       what you want. Defaults to None.
#   ram (bool, optional): load to ram or not. Defaults to False.
__jobs = [
    {"inset": './_data/FashionMNIST_Unpack/train',
     "outfolder": './_data/FashionMNIST_Dirty/train',
     "method": 'GaussNoise',
     "kargs": [{"var": 0.05},
               {"var": 0.1},
               {"var": 0.15},
               {"var": 0.2},
               {"var": 0.25},
               {"var": 0.3},
               {"var": 0.35},
               {"var": 0.4},
               {"var": 0.45},
               {"var": 0.5},
               ],
     "tags": ['T-shirt_top', 'Trouser', 'Pullover',
              'Dress', 'Coat', 'Sandal', 'Shirt',
              'Sneaker', 'Bag', 'Ankle boot'],
     "ram": True,
     },
    {"inset": './_data/FashionMNIST_Unpack/train',
     "outfolder": './_data/FashionMNIST_Dirty/train',
     "method": 'wl',
     "kargs": [{"ratio": 0.05},
               {"ratio": 0.1},
               {"ratio": 0.15},
               {"ratio": 0.2},
               {"ratio": 0.25},
               {"ratio": 0.3},
               {"ratio": 0.35},
               {"ratio": 0.4},
               {"ratio": 0.45},
               {"ratio": 0.5},
               ],
     "tags": ['T-shirt_top', 'Trouser', 'Pullover',
              'Dress', 'Coat', 'Sandal', 'Shirt',
              'Sneaker', 'Bag', 'Ankle boot'],
     "ram": True,
     },
]


def genSet(inset, outfolder, method, kargs, tags=None, ram=False):
    r"""Generate dirty datasets based on one specified source set
        More instructions follows the head of this script
    """
    try:
        # Always search for pack version first
        inset = setfromPack(inset)
    except FileNotFoundError:
        # Try loading from folder
        inset = setfromfolder(inset, tags=tags, ram=ram)

    print(inset)
    genDirtySet(inset, outfolder, method, kargs)


if __name__ == '__main__':
    for job in __jobs:
        genSet(**job)
    print("Done")
