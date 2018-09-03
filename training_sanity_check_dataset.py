# training on chalearn with cropped image faces
import chainer
import numpy as np
from deepimpression2.model_16 import Deepimpression
import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_absolute_error, softmax_cross_entropy
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
import deepimpression2.chalearn20.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle


model = Deepimpression()
# optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
optimizer.setup(model)

if C.ON_GPU:
    model = model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % model.count_params())

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')

train_loss = []
pred_diff_train = np.zeros((C.EPOCHS, 5), float)

val_loss = []
pred_diff_val = np.zeros((C.EPOCHS, 5), float)

# train_uid_keys_map = h5.File(P.TRAIN_UID_KEYS_MAPPING, 'r')
# val_uid_keys_map = h5.File(P.VAL_UID_KEYS_MAPPING, 'r')

training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
val_steps = len(val_labels) // C.VAL_BATCH_SIZE

id_frames = h5.File(P.NUM_FRAMES, 'r')

print('Enter training loop with validation')
for e in range(C.EPOCHS): # C.EPOCHS
    loss_tmp = []
    pd_tmp = np.zeros((training_steps, 5), dtype=float)
    _tr_labs = list(train_labels)
    shuffle(_tr_labs)

    ts = time.time()
    for s in range(training_steps):  # training_steps
        train_labels_selected = _tr_labs[s*C.TRAIN_BATCH_SIZE:(s+1)*C.TRAIN_BATCH_SIZE]
        assert(len(train_labels_selected) == C.TRAIN_BATCH_SIZE)
        labels, data = D.load_data_sanity(train_labels_selected, train_labels, id_frames)

        if C.ON_GPU:
            data = to_gpu(data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        # training
        with cp.cuda.Device(C.DEVICE):
            with chainer.using_config('train', True):
                model.cleargrads()
                prediction = model(data)

                loss = mean_absolute_error(prediction, labels)

                loss.backward()
                optimizer.update()

        loss_tmp.append(float(loss.data))

        pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))


    pred_diff_train[e] = np.mean(pd_tmp, axis=0)
    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    train_loss.append(loss_tmp_mean)
    print('E %d. train loss: ' % e, loss_tmp_mean,
          ' pred diff OCEAS: ', pred_diff_train[e],
          ' time: ', time.time() - ts)

    # U.record_loss_sanity('train', loss_tmp_mean, pred_diff_train[e])

    # validation
    loss_tmp = []
    pd_tmp = np.zeros((val_steps, 5), dtype=float)
    _v_labs = list(val_labels)
    shuffle(_v_labs)

    ts = time.time()
    for vs in range(val_steps):  # val_steps
        val_labels_selected = _v_labs[vs * C.VAL_BATCH_SIZE:(vs + 1) * C.VAL_BATCH_SIZE]
        assert (len(val_labels_selected) == C.VAL_BATCH_SIZE)
        labels, data = D.load_data_sanity(val_labels_selected, val_labels, id_frames)

        # if C.ON_GPU:
        #     data = to_gpu(data, device=C.DEVICE)
        #     labels = to_gpu(labels, device=C.DEVICE)

        # training
        with cp.cuda.Device(C.DEVICE):
            with chainer.using_config('train', False):
                # model.cleargrads()
                # prediction = model(data)

                # for train_18, guess 0.5
                prediction = chainer.Variable(np.ones((C.VAL_BATCH_SIZE, 5), dtype=np.float32) * 0.5)

                loss = mean_absolute_error(prediction, labels)


        loss_tmp.append(float(loss.data))

        pd_tmp[vs] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))

    pred_diff_val[e] = np.mean(pd_tmp, axis=0)
    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    val_loss.append(loss_tmp_mean)
    print('E %d. val loss: ' % e, loss_tmp_mean,
          ' pred diff OCEAS: ', pred_diff_val[e],
          ' time: ', time.time() - ts)

    U.record_loss_sanity('val', loss_tmp_mean, pred_diff_val[e])

    # save model
    # if ((e + 1) % 10) == 0:
    #     name = os.path.join(P.MODELS, 'epoch_%d_16' % e)
    #     chainer.serializers.save_npz(name, model)

