import chainer
from chainer.links import Linear
from chainer.functions import tanh

# model for predicting traits from luminance.
# 2 models, one for single trait and one for all traits


class SimpleAll(chainer.Chain):
    def __init__(self):
        super(SimpleAll, self).__init__()
        with self.init_scope():
            self.fc = Linear(in_size=1, out_size=5)

    @staticmethod
    def scale(x):
        # scale between 0-1
        ans = (x + 1) / 2
        return ans

    def __call__(self, x):
        h = self.fc(x)
        h = tanh(h)
        h = self.scale(h)
        return h


class SimpleOne(chainer.Chain):
    def __init__(self):
        super(SimpleOne, self).__init__()
        with self.init_scope():
            self.fc = Linear(in_size=1, out_size=1)

    @staticmethod
    def scale(x):
        # scale between 0-1
        ans = (x + 1) / 2
        return ans

    def __call__(self, x):
        h = self.fc(x)
        h = tanh(h)
        h = self.scale(h)
        return h