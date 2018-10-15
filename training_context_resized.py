# Q: what is the influence of background on personality prediction?
# Resize the images to overcome the effects of a larger background

'''
Results:
__________________________________________
|_____ test  |       |       |           |
|    |_______| face  | BG    | BG + face |
| train      |       |       |           |
|------------|-------|-------|-----------|
| face       | 0.114 | 0.16  |   0.159   | 0.046, 0.045
|------------|-------|-------|-----------|
| BG         | 0.137 | 0.126 |   0.128   | 0.011, 0.009
|------------|-------|-------|-----------|
| BG + face  | 0.147 | 0.135 |   0.123   | 0.024, 0.012
------------------------------------------
Conclusion: it seems that the network is learning that face has a structure
'''

import chainer
import numpy as np
from deepimpression2.model_16 import Deepimpression
import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_absolute_error, softmax_cross_entropy
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
# import deepimpression2.chalearn20.data_utils as D
import deepimpression2.chalearn30.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle


my_model = Deepimpression()

load_model = True
if load_model:
    p = os.path.join(P.MODELS, 'epoch_89_33')
    chainer.serializers.load_npz(p, my_model)
    print('model loaded')
    continuefrom = 0
else:
    continuefrom = 0

# optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

# epochs = C.EPOCHS
epochs = 1

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')

train_loss = []
pred_diff_train = np.zeros((epochs, 5), float)

val_loss = []
pred_diff_val = np.zeros((epochs, 5), float)

test_loss = []
pred_diff_test = np.zeros((epochs, 5), float)

training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
val_steps = len(val_labels) // C.VAL_BATCH_SIZE
test_steps = len(test_labels) // C.TEST_BATCH_SIZE
# training_steps = 10
# val_steps = 10

id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, which_data, ordered=False,
        save_all_results=False, record_predictions=False, record_loss=True):
    print('steps: ', steps)
    assert(which in ['train', 'test', 'val'])
    assert(which_data in ['all', 'bg', 'face'])

    if which == 'train':
        which_batch_size = C.TRAIN_BATCH_SIZE
    elif which == 'val':
        which_batch_size = C.VAL_BATCH_SIZE
    elif which == 'test':
        which_batch_size = C.TEST_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, 5), dtype=float)
    _labs = list(which_labels)

    preds = np.zeros((steps, 5), dtype=float)

    if not ordered:
        shuffle(_labs)

    ts = time.time()
    for s in range(steps):
        # HERE
        print(s)
        # HERE
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)
        labels, data, _ = D.load_data(labels_selected, which_labels, frames, which_data, resize=True, ordered=ordered)

        if C.ON_GPU:
            data = to_gpu(data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            if which == 'train':
                config = True
            else:
                config = False

            with chainer.using_config('train', config):
                if which == 'train':
                    model.cleargrads()
                prediction = model(data)

                loss = mean_absolute_error(prediction, labels)

                if which == 'train':
                    loss.backward()
                    optimizer.update()

        if record_loss:
            loss_tmp.append(float(loss.data))
            pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))
        if record_predictions and which == 'test':
            preds[s] = to_cpu(prediction.data)

    if record_loss:
        pred_diff[e] = np.mean(pd_tmp, axis=0)
        loss_tmp_mean = np.mean(loss_tmp, axis=0)
        loss_saving.append(loss_tmp_mean)
        print('E %d. %s loss: ' %(e, which), loss_tmp_mean,
              ' pred diff OCEAS: ', pred_diff[e],
              ' time: ', time.time() - ts)

        U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])

        if which == 'test' and save_all_results:
            U.record_loss_all_test(loss_tmp)
    if record_predictions and which == 'test':
        U.record_all_predictions(which, preds)


print('Enter training loop with validation')
for e in range(continuefrom, epochs):
    train_on = 'bg'
    # validate_on = 'face'
    # print('trained on: %s val on: %s' % (train_on, validate_on))
    test_on = 'bg'
    print('trained on: %s test on %s' % (train_on, test_on))
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    # run(which='train', steps=training_steps, which_labels=train_labels, frames=id_frames,
    #     model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_train,
    #     loss_saving=train_loss, which_data=train_on)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    # run(which='val', steps=val_steps, which_labels=val_labels, frames=id_frames,
    #     model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_val,
    #     loss_saving=val_loss, which_data=validate_on)
    # ----------------------------------------------------------------------------
    # test
    # ----------------------------------------------------------------------------
    times = 1
    for i in range(1):
        if times == 1:
            ordered = True
            save_all_results = True
        else:
            ordered = False
            save_all_results = False

        run(which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
            model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
            loss_saving=test_loss, which_data=test_on, ordered=ordered, save_all_results=save_all_results,
            record_loss=False, record_predictions=True )
    # best val 'all': epoch_99_32
    # best val 'bg': epoch_89_33
    # best val 'face': epoch_29_34

    # # save model
    # if ((e + 1) % 10) == 0:
    #     name = os.path.join(P.MODELS, 'epoch_%d_34' % e)
    #     chainer.serializers.save_npz(name, my_model)

