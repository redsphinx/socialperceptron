import numpy as np
import os
import deepimpression2.paths as P


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def record_loss(which, loss):
    if which == 'train':
        path = P.TRAIN_LOG
    elif which == 'val':
        path = P.VAL_LOG
    # TODO: add case for test

    with open(path, 'a') as mf:
        mf.write('%s\n' % str(loss)[0:6])
