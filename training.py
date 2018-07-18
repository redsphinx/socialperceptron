# training on chalearn with cropped image faces
import chainer
import numpy as np
from deepimpression2.model import Siamese
import deepimpression2.chalearn20.constants as C
from deepimpression2.util import get_batch, get_labels, update_loss
from chainer.functions import sigmoid_cross_entropy
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.chalearn20.paths as P


model = Siamese()
optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
optimizer.setup(model)

# TODO: load data
training_data = h5.File(P.CHALEARN_FACES_TRAIN_H5)
training_labels = []
training_loss = []
val_data = h5.File(P.CHALEARN_FACES_VAL_H5)
val_labels = []
val_loss = []

training_steps = len(training_data) / C.TRAIN_BATCH_SIZE
val_steps = len(val_data) / C.VAL_BATCH_SIZE

for e in range(C.EPOCHS):
    loss_tmp = []

    for s in range(training_steps):
        # TODO: get the batch
        batch = get_batch()
        batch_label = get_labels()

        # training
        with chainer.using_config('train', True):
            model.cleargrads()

            prediction = model(batch[0], batch[1])
            loss = sigmoid_cross_entropy(prediction, batch_label)

            loss.backward()
            optimizer.update()

    training_loss = update_loss(training_loss, np.mean(loss_tmp))

    # implement validation
    batch = get_batch()
    batch_label = get_labels()

    with chainer.using_config('train', False):
        prediction = model(batch[0], batch[1])
        loss = sigmoid_cross_entropy(prediction, batch_label)
        val_loss = update_loss(val_loss, loss)

