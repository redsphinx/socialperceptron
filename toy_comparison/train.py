from deepimpression2.toy_comparison.model import Siamese
from chainer.optimizers import Adam
import numpy as np
from random import randint
import chainer
from chainer.functions import sigmoid_cross_entropy, mean_squared_error


model = Siamese()
optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
optimizer.setup(model)


epochs = 1000
batch_size = 64

data_batch = np.zeros((batch_size, 2), dtype=np.float32)

# labels = np.zeros((batch_size, 2), dtype=int)
labels = np.zeros((batch_size, 2), dtype=np.float32)

loss_list = []

for e in range(epochs):
    for b in range(batch_size):
        x1, x2 = randint(0, 9), randint(0, 9)
        data_batch[b] = [np.float32(x1), x2]
        # (1, 0) = left  (0, 1) = right
        if x1 > x2: # left
            labels[b] = [np.float32(1), np.float32(0)]
            # labels[b] = [1, 0]
        else: # right or equal
            labels[b] = [np.float32(0), np.float32(1)]
            # labels[b] = [0, 1]

    with chainer.using_config('train', True):
        model.cleargrads()
        d1 = np.expand_dims(data_batch[:, 0], -1)
        d2 = np.expand_dims(data_batch[:, 1], -1)
        # prediction = model(np.expand_dims(d1, 0), np.expand_dims(d2, 0))
        prediction = model(d1, d2)
        # loss = sigmoid_cross_entropy(prediction, labels)
        loss = mean_squared_error(prediction, labels)
        loss.backward()
        optimizer.update()

    loss_list.append(float(loss.data))

    print(e, float(loss.data))
    # print(prediction[:10], labels[:10])
