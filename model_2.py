import chainer
from chainer.links import Convolution2D, Linear
from chainer.initializers import HeNormal
from chainer.functions import average_pooling_2d, max_pooling_2d, concat, selu


# replace relu and batchnorm with selu

### BLOCK ###
class ConvolutionBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        with self.init_scope():
            self.conv = Convolution2D(in_channels, out_channels,
                                      ksize=7, stride=2, pad=3,
                                      initialW=HeNormal())

    def __call__(self, x):
        h = self.conv(x)
        y = selu(h)
        return y


class ResidualBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.res_branch2a = Convolution2D(in_channels, out_channels,
                                              ksize=3, pad=1,
                                              initialW=HeNormal())
            self.res_branch2b = Convolution2D(out_channels, out_channels,
                                              ksize=3, pad=1,
                                              initialW=HeNormal())

    def __call__(self, x):
        h = self.res_branch2a(x)
        h = selu(h)
        h = self.res_branch2b(h)
        h += x
        y = selu(h)
        return y


class ResidualBlockB(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockB, self).__init__()
        with self.init_scope():
            self.res_branch1 = Convolution2D(in_channels, out_channels,
                                             ksize=1, stride=2,
                                             initialW=HeNormal())
            self.res_branch2a = Convolution2D(in_channels, out_channels,
                                              ksize=3, stride=2, pad=1,
                                              initialW=HeNormal())
            self.res_branch2b = Convolution2D(out_channels, out_channels,
                                              ksize=3, pad=1,
                                              initialW=HeNormal())

    def __call__(self, x):
        temp = self.res_branch1(x)
        h = self.res_branch2a(x)
        h = selu(h)
        h = self.res_branch2b(h)
        h = temp + h
        y = selu(h)
        return y
### BLOCK ###


### BRANCH ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__()
        with self.init_scope():
            self.conv1_selu = ConvolutionBlock(3, 32)
            self.res2a_selu = ResidualBlock(32, 32)
            self.res2b_selu = ResidualBlock(32, 32)
            self.res3a_selu = ResidualBlockB(32, 64)
            self.res3b_selu = ResidualBlock(64, 64)
            self.res4a_selu = ResidualBlockB(64, 128)
            self.res4b_selu = ResidualBlock(128, 128)
            self.res5a_selu = ResidualBlockB(128, 256)
            self.res5b_selu = ResidualBlock(256, 256)

    def __call__(self, x):
        h = self.conv1_selu(x)
        h = max_pooling_2d(h, ksize=3, stride=2, pad=1)
        h = self.res2a_selu(h)
        h = self.res2b_selu(h)
        h = self.res3a_selu(h)
        h = self.res3b_selu(h)
        h = self.res4a_selu(h)
        h = self.res4b_selu(h)
        h = self.res5a_selu(h)
        h = self.res5b_selu(h)
        y = average_pooling_2d(h, ksize=h.data.shape[2:])
        # TODO: what is output size
        # y.shape
        # (32, 256, 1, 1)
        return y
### BRANCH ###


### MODEL ###

class Siamese(chainer.Chain):
    def __init__(self):
        super(Siamese, self).__init__()
        with self.init_scope():
            self.b1 = ResNet18()
            self.b2 = ResNet18()
            self.fc = Linear(in_size=512, out_size=10)

    def __call__(self, x1, x2):
        _1 = self.b1(x1) # (32, 256, 1, 1)
        _2 = self.b1(x2)

        h = concat((_1, _2))
        h = self.fc(h)
        h = chainer.functions.reshape(h, (h.shape[0], 5, 2))
        return h
