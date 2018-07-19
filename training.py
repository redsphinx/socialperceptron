# training on chalearn with cropped image faces
import chainer
import numpy as np
from deepimpression2.model import Siamese
import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
import deepimpression2.chalearn20.data_utils as D


model = Siamese()
optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
optimizer.setup(model)


def update_loss(total_loss, loss):
    # TODO: make for training + validation
    pass



# TODO: load data
train_data = ''
train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
train_loss = []

val_data = ''
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
val_loss = []

train_uid_keys_map = h5.File(P.TRAIN_UID_KEYS_MAPPING, 'r')
val_uid_keys_map = h5.File(P.VAL_UID_KEYS_MAPPING, 'r')

training_steps = 1 # len(train_data) / C.TRAIN_BATCH_SIZE
val_steps = len(val_data) / C.VAL_BATCH_SIZE

for e in range(C.EPOCHS):
    loss_tmp = []

    for s in range(training_steps):
        # TODO: get the batch
        labels, left_data, right_data = D.load_data('train', train_uid_keys_map)

        # training
        with chainer.using_config('train', True):
            model.cleargrads()

            prediction = model(left_data, right_data)
            loss = sigmoid_cross_entropy(prediction, labels)

            loss.backward()
            optimizer.update()

    training_loss = update_loss(train_loss, np.mean(loss_tmp))

    # implement validation

