from deepimpression2.toy_comparison.model import Siamese
from chainer.optimizers import Adam
import numpy as np
from random import randint
import chainer
from chainer.functions import sigmoid_cross_entropy, mean_squared_error
# from deepimpression2 import util as U
import math


def sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s


def binarize(arr):
    assert(arr.ndim == 2)

    shapes = arr.shape
    new_arr = np.zeros(shapes, dtype=int)

    for j in range(shapes[0]):
        if sigmoid(float(arr[j].data)) > 0.5:
            new_arr[j] = 1
        else:
            new_arr[j] = 0

    return new_arr


def make_confusion_matrix(prediction, labels):
    # shape prediction and labels (32, 5, 2)
    # (1, 0) = left    (0, 1) = right
    prediction = binarize(prediction)
    shapes = prediction.shape
    tl, fl, tr, fr = 0, 0, 0, 0

    for i in range(shapes[0]):
        if labels[i] == 1:
            if prediction[i] == 1:
                tl += 1
            else:
                fl += 1
        elif labels[i] == 0:
            if prediction[i] == 0:
                tr += 1
            else:
                fr += 1

    # cm_per_trait = np.mean(cm_per_trait, axis=0)

    return [tl, fl, tr, fr]


def main():
    model = Siamese()
    print('model params: ', model.count_params())
    optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
    optimizer.setup(model)

    epochs = 1000
    batch_size = 64

    data_batch = np.zeros((batch_size, 2), dtype=np.float32)

    labels = np.zeros((batch_size, 1), dtype=int)

    loss_list = []

    for e in range(epochs):
        for b in range(batch_size):
            x1 = randint(0, 9)
            x2 = randint(0, 9)
            if x1 == x2:
                lr = randint(0, 1) # decide which will be bigger
                if x1 == 0:
                    if lr == 1: # left
                        x1 += 1
                    else:
                        x2 += 1
                elif x1 == 9:
                    if lr == 1:
                        x2 -= 1
                    else:
                        x1 -= 1
                else:
                    if lr == 1:
                        x1 += 1
                    else:
                        x2 += 1

            data_batch[b] = [np.float32(x1), x2]
            # (1, 0) = left  (0, 1) = right
            if x1 > x2: # left
                # labels[b] = [np.float32(1), np.float32(0)]
                labels[b] = 1
            else: # right or equal
                # labels[b] = [np.float32(0), np.float32(1)]
                labels[b] = 0

        with chainer.using_config('train', True):
            model.cleargrads()
            d1 = np.expand_dims(data_batch[:, 0], -1)
            d2 = np.expand_dims(data_batch[:, 1], -1)
            # prediction = model(np.expand_dims(d1, 0), np.expand_dims(d2, 0))
            prediction = model(d1, d2)
            loss = sigmoid_cross_entropy(prediction, labels)
            # loss = mean_squared_error(prediction, labels)
            loss.backward()
            optimizer.update()

        loss_list.append(float(loss.data))

        # print(e, float(loss.data))

        if (e+1) % 100 == 0:
            cm = make_confusion_matrix(prediction, labels)
            print(e, cm, loss, 'W: ', model.fc1.W, 'b: ', model.fc1.b)


for i in range(10):
    print('iteration %d' % i)
    main()