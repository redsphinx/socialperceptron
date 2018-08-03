import chainer
from chainer.links import Linear, BatchNormalization
from chainer.initializers import HeNormal
from chainer.functions import relu, concat, softmax


class Branch(chainer.Chain):
    def __init__(self):
        super(Branch, self).__init__()
        with self.init_scope():
            self.fc1 = Linear(in_size=1, out_size=1)
            # self.bn1 = BatchNormalization(20)
            # self.fc2 = Linear(in_size=20, out_size=1)
            # self.bn2 = BatchNormalization(1)

    def __call__(self, x):
        # h = self.fc1(x)
        # h = self.bn1(h)
        # h = relu(h)
        # h = self.fc2(h)
        # h = self.bn2(h)
        h = relu(x)
        return h


class Siamese(chainer.Chain):
    def __init__(self):
        super(Siamese, self).__init__()
        with self.init_scope():
            self.b1 = Branch()
            # self.fc1 = Linear(in_size=1, out_size=1)
            self.fc1 = Linear(in_size=2, out_size=2)
            # self.fc2= Linear(in_size=1, out_size=2)
            # self.fc3 = Linear(in_size=1, out_size=2)
            # self.fc = Linear(in_size=1, out_size=2)

    def __call__(self, x1, x2):
        _1 = self.b1(x1)
        _2 = self.b1(x2)

        # h = concat((_1, _2))
        h = _1 - _2
        h = self.fc1(h)
        h = relu(h)
        # h = self.fc2(h)
        # h = relu(h)
        # h = self.fc3(h)
        # h = softmax(h)
        # print(h)
        return h