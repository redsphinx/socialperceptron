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
import time
from chainer.backends.cuda import to_cpu, to_gpu
import deepimpression2.util as U
import os


model = Siamese()
optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
optimizer.setup(model)

if C.ON_GPU:
    model = model.to_gpu(device=C.DEVICE)


def update_loss(total_loss, l):
    total_loss.append(l)
    return total_loss


print('Initializing')

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')

train_loss = []
val_loss = []

train_uid_keys_map = h5.File(P.TRAIN_UID_KEYS_MAPPING, 'r')
val_uid_keys_map = h5.File(P.VAL_UID_KEYS_MAPPING, 'r')

training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
val_steps = len(val_labels) // C.VAL_BATCH_SIZE

id_frames = h5.File(P.NUM_FRAMES, 'r')

print('Enter training loop with validation')
for e in range(100): # EPOCHS
    loss_tmp = []

    ts = time.time()
    for s in range(training_steps): # training_steps

        labels, left_data, right_data = D.load_data('train', train_uid_keys_map, train_labels, id_frames)

        if C.ON_GPU:
            left_data = to_gpu(left_data, device=C.DEVICE)
            right_data = to_gpu(right_data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        # training
        with chainer.using_config('train', True):
            model.cleargrads()

            prediction = model(left_data, right_data)

            loss = sigmoid_cross_entropy(prediction, labels)

            loss.backward()
            optimizer.update()

            loss_tmp.append(float(loss.data))

    loss_tmp_mean = np.mean(loss_tmp)
    train_loss.append(loss_tmp_mean)
    print('epoch %d. train loss: ' % e, loss_tmp_mean, ' time: ', time.time() - ts)
    U.record_loss('train', loss_tmp_mean)

    # validation
    loss_tmp = []
    ts = time.time()
    for vs in range(val_steps): # val_steps

        labels, left_data, right_data = D.load_data('val', val_uid_keys_map, val_labels, id_frames)

        if C.ON_GPU:
            left_data = to_gpu(left_data, device=C.DEVICE)
            right_data = to_gpu(right_data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        # training
        with chainer.using_config('train', False):
            model.cleargrads()
            prediction = model(left_data, right_data)
            loss = sigmoid_cross_entropy(prediction, labels)

        loss_tmp.append(float(loss.data))

    loss_tmp_mean = np.mean(loss_tmp)
    val_loss.append(loss_tmp_mean)
    print('epoch %d. val loss: ' % e, loss_tmp_mean, ' time: ', time.time() - ts)
    U.record_loss('val', loss_tmp_mean)

    # save model
    name = os.path.join(P.MODELS, 'epoch_%d' % e)
    chainer.serializers.save_npz(name, model)