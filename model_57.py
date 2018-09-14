import chainer
from chainer.links import Convolution2D, BatchNormalization, Linear
from chainer.initializers import HeNormal, GlorotUniform, Zero, One
from chainer.functions import relu, average_pooling_2d, max_pooling_2d, concat, tanh, flatten
import numpy as np


class LastLayers(chainer.Chain):
    def __init__(self):
        super(LastLayers, self).__init__()
        with self.init_scope():
            self.fc = Linear(in_size=512, out_size=5)

    @staticmethod
    def scale(x):
        # scale between 0-1
        ans = (x + 1) / 2
        return ans

    def __call__(self, x1, x2):
        h = concat((x1, x2), axis=1)
        h = self.fc(h)
        h = tanh(h)
        h = self.scale(h)
        return h