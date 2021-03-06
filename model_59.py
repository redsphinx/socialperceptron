import chainer
from chainer.links import Convolution2D, BatchNormalization, Linear
from chainer.initializers import HeNormal, GlorotUniform, Zero, One
from chainer.functions import relu, average_pooling_2d, max_pooling_2d, concat, tanh
import numpy as np

# model for predicting single trait

# Deepimpression: face, bg
# LastLayers: all (can only be used after face and bg have been trained)


which_initializer = 1
initial = None

if which_initializer == 1:
    initial = HeNormal()
elif which_initializer == 2:
    initial = GlorotUniform()


### BLOCK ###
class ConvolutionBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        with self.init_scope():
            self.conv = Convolution2D(in_channels, out_channels,
                                      ksize=7, stride=2, pad=3,
                                      initialW=initial)
            self.bn_conv = BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn_conv(h)
        y = relu(h)
        return y


class ResidualBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.res_branch2a = Convolution2D(in_channels, out_channels,
                                              ksize=3, pad=1,
                                              initialW=initial)
            self.bn_branch2a = BatchNormalization(out_channels)
            self.res_branch2b = Convolution2D(out_channels, out_channels,
                                              ksize=3, pad=1,
                                              initialW=initial)
            self.bn_branch2b = BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.res_branch2a(x)
        h = self.bn_branch2a(h)
        h = relu(h)
        h = self.res_branch2b(h)
        h = self.bn_branch2b(h)
        h += x
        y = relu(h)
        return y


class ResidualBlockB(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockB, self).__init__()
        with self.init_scope():
            self.res_branch1 = Convolution2D(in_channels, out_channels,
                                             ksize=1, stride=2,
                                             initialW=initial)
            self.bn_branch1 = BatchNormalization(out_channels)
            self.res_branch2a = Convolution2D(in_channels, out_channels,
                                              ksize=3, stride=2, pad=1,
                                              initialW=initial)
            self.bn_branch2a = BatchNormalization(out_channels)
            self.res_branch2b = Convolution2D(out_channels, out_channels,
                                              ksize=3, pad=1,
                                              initialW=initial)
            self.bn_branch2b = BatchNormalization(out_channels)

    def __call__(self, x):
        temp = self.res_branch1(x)
        temp = self.bn_branch1(temp)
        h = self.res_branch2a(x)
        h = self.bn_branch2a(h)
        h = chainer.functions.relu(h)
        h = self.res_branch2b(h)
        h = self.bn_branch2b(h)
        h = temp + h
        y = chainer.functions.relu(h)
        return y
### BLOCK ###


### BRANCH ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__()
        with self.init_scope():
            self.conv1_relu = ConvolutionBlock(3, 32)
            self.res2a_relu = ResidualBlock(32, 32)
            self.res2b_relu = ResidualBlock(32, 32)
            self.res3a_relu = ResidualBlockB(32, 64)
            self.res3b_relu = ResidualBlock(64, 64)
            self.res4a_relu = ResidualBlockB(64, 128)
            self.res4b_relu = ResidualBlock(128, 128)
            self.res5a_relu = ResidualBlockB(128, 256)
            self.res5b_relu = ResidualBlock(256, 256)

    def __call__(self, x):
        h = self.conv1_relu(x)
        h = max_pooling_2d(h, ksize=3, stride=2, pad=1)
        h = self.res2a_relu(h)
        h = self.res2b_relu(h)
        h = self.res3a_relu(h)
        h = self.res3b_relu(h)
        h = self.res4a_relu(h)
        h = self.res4b_relu(h)
        h = self.res5a_relu(h)
        h = self.res5b_relu(h)
        y = average_pooling_2d(h, ksize=h.data.shape[2:])
        # y.shape
        # (32, 256, 1, 1)
        return y
### BRANCH ###


### MODEL ###

class Deepimpression(chainer.Chain):
    def __init__(self):
        super(Deepimpression, self).__init__()
        with self.init_scope():
            self.b1 = ResNet18()
            self.fc = Linear(in_size=256, out_size=1)

    @staticmethod
    def scale(x):
        # scale between 0-1
        ans = (x + 1) / 2
        return ans

    def __call__(self, x):
        h = self.b1(x)  # (32, 256, 1, 1)
        list_h = h
        h = self.fc(h)
        h = tanh(h)
        h = self.scale(h)
        return h, list_h


class LastLayers(chainer.Chain):
    def __init__(self):
        super(LastLayers, self).__init__()
        with self.init_scope():
            self.fc = Linear(in_size=512, out_size=1)

    @staticmethod
    def scale(x):
        # scale between 0-1
        ans = (x + 1) / 2
        return ans

    def __call__(self, x1, x2):
        h = concat((x1, x2), axis=1)
        # h = chainer.functions.expand_dims(h, 0)
        h = self.fc(h)
        h = tanh(h)
        h = self.scale(h)
        return h