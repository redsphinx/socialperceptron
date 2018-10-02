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


def linear_regression_all():
    labels, data, _ = D.load_data_luminance(list(train_labels), train_labels, id_frames)
    reg = LinearRegression().fit(data, labels)
    reg.score(data, labels)
    print('reg score: ', reg.score(data, labels))  # 0.011438774006662768, 0.01160391566097031, 0.011790330654130819
    labels, data, _ = D.load_data_luminance(list(test_labels), test_labels, id_frames, ordered=True)
    prediction = reg.predict(data)
    loss = np.abs(prediction - labels)
    loss = np.mean(loss, axis=1)
    # loss = np.mean(loss)
    print('loss: ', np.mean(loss))

    num = P.TEST_LOG.split('_')[-1].split('.')[0]
    save_path = os.path.join(P.LOG_BASE, 'testall_%s.txt' % num)
    with open(save_path, 'a') as mf:
        for i in range(len(loss)):
            line = '%s\n' % str(loss[i])[0:6]
            mf.write(line)


# linear_regression_all()


def linear_regression_single():
    labels, data, _ = D.load_data_luminance(list(train_labels), train_labels, id_frames)
    labels_test, data_test, _ = D.load_data_luminance(list(test_labels), test_labels, id_frames, ordered=True)

    traits = ['O', 'C', 'E', 'A', 'S']
    for i, t in enumerate(traits):
        print('for trait: %s' % t)
        label_trait = labels[:,  i]
        reg = LinearRegression().fit(data, label_trait)

        print('reg score: ', reg.score(data, label_trait))  # 0.011438774006662768, 0.01160391566097031, 0.011790330654130819

        prediction = reg.predict(data_test)
        loss = np.abs(prediction - labels_test[:, i])
        print('loss: ', np.mean(loss))

        num = P.TEST_LOG.split('_')[-1].split('.')[0]
        save_path = os.path.join(P.LOG_BASE, 'testall_%s_%s.txt' % (num, t))
        with open(save_path, 'a') as mf:
            for i in range(len(loss)):
                line = '%s\n' % str(loss[i])[0:6]
                mf.write(line)

# linear_regression_single()


# for trait: O
# reg score:  0.009427910109708337
# loss:  0.118614726
# for trait: C
# reg score:  0.019465699450429197
# loss:  0.12812673
# for trait: E
# reg score:  0.014062835218046077
# loss:  0.12711206
# for trait: A
# reg score:  0.005144573433965238
# loss:  0.108449906
# for trait: S
# reg score:  0.009835833135908079
# loss:  0.12744974