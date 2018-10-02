import numpy as np
from sklearn.linear_model import LinearRegression
import chainer
from chainer.optimizers import Adam
from chainer.functions import mean_absolute_error
import numpy as np
from deepimpression2.model_64 import SimpleAll, SimpleOne
import deepimpression2.paths as P
import os
import deepimpression2.constants as C
import h5py as h5
from random import shuffle
from time import time
import deepimpression2.chalearn30.data_utils as D
from chainer.backends.cuda import to_gpu, to_cpu
import cupy as cp
import deepimpression2.util as U


train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
id_frames = h5.File(P.NUM_FRAMES, 'r')


def linear_regression():
    labels, data, _ = D.load_data_luminance(list(train_labels), train_labels, id_frames)
    reg = LinearRegression().fit(data, labels)
    reg.score(data, labels)
    print('reg score: ', reg.score(data, labels))  # 0.011438774006662768, 0.01160391566097031, 0.011790330654130819
    labels, data, _ = D.load_data_luminance(list(test_labels), test_labels, id_frames, ordered=True)
    prediction = reg.predict(data)
    loss = np.abs(prediction - labels)
    loss = np.mean(loss, axis=1)
    loss = np.mean(loss)
    print('loss: ', loss)


linear_regression()
