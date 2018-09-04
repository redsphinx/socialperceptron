# testing out LIME explainer package on the simplified left-right question of which is more positive


from deepimpression2.model import Siamese
import chainer
import numpy as np
import h5py as h5
import deepimpression2.paths as P
# import deepimpression2.chalearn20.data_utils as D
import deepimpression2.chalearn30.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle
import deepimpression2.constants as C


model = Siamese()
# load pretrained model


if C.ON_GPU:
    model = model.to_gpu(device=C.DEVICE)