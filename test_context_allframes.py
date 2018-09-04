# Q: what is the influence of background on personality prediction?

'''
Results:
__________________________________________
|_____ test  |       |       |           |
|    |_______| face  | BG    | BG + face |
| train      |       |       |           |
|------------|-------|-------|-----------|
| face       |   x   |       |           |
|------------|-------|-------|-----------|
| BG         |       |   x   |           |
|------------|-------|-------|-----------|
| BG + face  |       |       |     x     |
------------------------------------------
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


'''
get model
for each video, load all frames
pass each frame through the network and extract 
for each frame
extract feature after h = self.b1(x)
take the mean of this feature
pass it through the rest
record this loss
'''

my_model = Deepimpression()

load_model = True
if load_model:
    p = os.path.join(P.MODELS, 'epoch_29_22')
    chainer.serializers.load_npz(p, my_model)
    print('model loaded')
    continuefrom = 0
else:
    continuefrom = 0


if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

# epochs = C.EPOCHS
epochs = 1

test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
test_loss = []
pred_diff_test = np.zeros((epochs, 5), float)

test_steps = len(test_labels) // C.TEST_BATCH_SIZE
# test_steps = 1

id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, which_data):
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
    shuffle(_labs)

    ts = time.time()
    for s in range(steps):
        print(s)
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)
        # labels, data = D.load_data(labels_selected, which_labels, frames, which_data)
        # TODO: make sure number of frames is at index 0 of data.shapes
        labels, data = D.load_video(labels_selected, which_labels, which_data)

        if C.ON_GPU:
            data = to_gpu(data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            config = False

            frames = data.shape[0]

            for f in range(frames):
                with chainer.using_config('train', config):
                    frame = np.expand_dims(frames[f], 0)
                    prediction = model(frame)
                    # TODO: extract


                    # loss = mean_absolute_error(prediction, labels)


        loss_tmp.append(float(loss.data))

        pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))

    pred_diff[e] = np.mean(pd_tmp, axis=0)
    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    loss_saving.append(loss_tmp_mean)
    print('E %d. %s loss: ' %(e, which), loss_tmp_mean,
          ' pred diff OCEAS: ', pred_diff[e],
          ' time: ', time.time() - ts)

    U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])


print('Enter training loop with validation')
for e in range(continuefrom, epochs):
    train_on = 'face'
    validate_on = 'face'
    test_on = 'bg'
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
    for i in range(1):
        run(which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
            model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
            loss_saving=test_loss, which_data=test_on)
    # best val 'all': epoch_49_20
    # best val 'bg': epoch_79_21
    # best val 'face': epoch_29_22


    # # save model
    # if ((e + 1) % 10) == 0:
    #     name = os.path.join(P.MODELS, 'epoch_%d_22' % e)
    #     chainer.serializers.save_npz(name, my_model)

