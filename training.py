# training on chalearn with cropped image faces
import chainer
import numpy as np
from deepimpression2.model import Siamese
# from deepimpression2.model_9 import Siamese
import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_squared_error
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
import deepimpression2.chalearn20.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp


model = Siamese()
optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
# optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
optimizer.setup(model)
alpha = 1

if C.ON_GPU:
    model = model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % model.count_params())

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')

train_loss = []
confusion_matrix_train = np.zeros((C.EPOCHS, 4), dtype=int)
confusion_matrix_trait_train = np.zeros((C.EPOCHS, 5, 4), dtype=int)
batch_statistics_train = np.zeros((2, C.EPOCHS, 5))

val_loss = []
confusion_matrix_val = np.zeros((C.EPOCHS, 4), dtype=int)
confusion_matrix_trait_val = np.zeros((C.EPOCHS, 5, 4), dtype=int)
batch_statistics_val = np.zeros((2, C.EPOCHS, 5))

train_uid_keys_map = h5.File(P.TRAIN_UID_KEYS_MAPPING, 'r')
val_uid_keys_map = h5.File(P.VAL_UID_KEYS_MAPPING, 'r')

training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
val_steps = len(val_labels) // C.VAL_BATCH_SIZE

id_frames = h5.File(P.NUM_FRAMES, 'r')

print('Enter training loop with validation')
for e in range(C.EPOCHS): # C.EPOCHS
    loss_tmp = []
    cm_tmp = np.zeros((training_steps, 4), dtype=int)
    cm_trait_tmp = np.zeros((training_steps, 5, 4), dtype=int)
    bs_tmp = np.zeros((2, training_steps, 5), dtype=int)

    ts = time.time()
    for s in range(training_steps):  # training_steps

        labels, left_data, right_data = D.load_data('train', train_uid_keys_map, train_labels, id_frames)
        num_left, num_right = D.label_statistics(labels)
        bs_tmp[0][s] = num_left
        bs_tmp[1][s] = num_right

        if C.ON_GPU:
            left_data = to_gpu(left_data, device=C.DEVICE)
            right_data = to_gpu(right_data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        # training
        with cp.cuda.Device(C.DEVICE):
            with chainer.using_config('train', True):
                model.cleargrads()
                prediction = model(left_data, right_data)
                loss = sigmoid_cross_entropy(prediction, labels)
                # _1 = sigmoid_cross_entropy(prediction, labels)
                # _2 = mean_squared_error(prediction, cp.asarray(labels, dtype=cp.float32))
                # print('loss: %s   %s' % (str(_1.data)[0:5], str(_2.data)[0:5]))
                # loss = _1 + alpha * _2

                loss.backward()
                optimizer.update()

        loss_tmp.append(float(loss.data))
        cm_tmp[s], cm_trait_tmp[s] = U.make_confusion_matrix(to_cpu(prediction.data), to_cpu(labels))

    batch_statistics_train[0][e] = np.mean(bs_tmp[0], axis=0)
    batch_statistics_train[1][e] = np.mean(bs_tmp[1], axis=0)
    confusion_matrix_trait_train[e] = np.mean(cm_trait_tmp, axis=0)
    confusion_matrix_train[e] = np.mean(cm_tmp, axis=0)
    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    train_loss.append(loss_tmp_mean)
    # print('epoch %d. train loss: ' % e, loss_tmp_mean, ' time: ', time.time() - ts)
    print('E %d. train loss: ' % e, loss_tmp_mean,
          ' [tl, fl, tr, fr]: ', np.mean(cm_tmp, axis=0),
          ' left labels OCEAS: ', batch_statistics_train[0][e],
          ' right labels OCEAS: ', batch_statistics_train[1][e],
          ' row: OCEAS\n', np.mean(cm_trait_tmp, axis=0),
          ' time: ', time.time() - ts)

    U.record_loss('train', loss_tmp_mean, np.mean(cm_trait_tmp, axis=0), np.mean(bs_tmp, axis=1))

    # # validation
    loss_tmp = []
    cm_tmp = np.zeros((val_steps, 4), dtype=int)
    cm_trait_tmp = np.zeros((val_steps, 5, 4), dtype=int)
    bs_tmp = np.zeros((2, val_steps, 5), dtype=int)

    ts = time.time()
    for vs in range(val_steps):  # val_steps

        labels, left_data, right_data = D.load_data('val', val_uid_keys_map, val_labels, id_frames)
        num_left, num_right = D.label_statistics(labels)
        bs_tmp[0][vs] = num_left
        bs_tmp[1][vs] = num_right

        if C.ON_GPU:
            left_data = to_gpu(left_data, device=C.DEVICE)
            right_data = to_gpu(right_data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        # validation
        with cp.cuda.Device(C.DEVICE):
            with chainer.using_config('train', False):
                model.cleargrads()
                prediction = model(left_data, right_data)
                loss = sigmoid_cross_entropy(prediction, labels)
                # loss = sigmoid_cross_entropy(prediction, labels) + \
                #        alpha * mean_squared_error(prediction, cp.asarray(labels, dtype=cp.float32))

        loss_tmp.append(float(loss.data))
        cm_tmp[vs], cm_trait_tmp[vs] = U.make_confusion_matrix(to_cpu(prediction.data), to_cpu(labels))

    batch_statistics_val[0][e] = np.mean(bs_tmp[0], axis=0)
    batch_statistics_val[1][e] = np.mean(bs_tmp[1], axis=0)

    confusion_matrix_trait_val[e] = np.mean(cm_trait_tmp, axis=0)
    confusion_matrix_val[e] = np.mean(cm_tmp, axis=0)

    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    val_loss.append(loss_tmp_mean)
    # print('epoch %d. val loss: ' % e, loss_tmp_mean, ' time: ', time.time() - ts)
    print('E %d. val loss: ' % e, loss_tmp_mean,
          ' [tl, fl, tr, fr]: ', np.mean(cm_tmp, axis=0),
          ' left labels OCEAS: ', batch_statistics_val[0][e],
          ' right labels OCEAS: ', batch_statistics_val[1][e],
          ' row: OCEAS\n', np.mean(cm_trait_tmp, axis=0),
          ' time: ', time.time() - ts)

    U.record_loss('val', loss_tmp_mean, np.mean(cm_trait_tmp, axis=0), np.mean(bs_tmp, axis=1))

    # save model
    if ((e + 1) % 10) == 0:
        name = os.path.join(P.MODELS, 'epoch_%d_13' % e)
        chainer.serializers.save_npz(name, model)

