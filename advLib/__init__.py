"""
    Tools for generate samples that can fool the classifier
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from .utils import genGrad, costTracker
from .deepfool import deepfoolL, deepfoolGD
