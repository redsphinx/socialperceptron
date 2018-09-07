# Q: what is the influence of background on personality prediction?

'''
Results:
__________________________________________
|_____ test  |       |       |           |
|    |_______| face  | BG    | BG + face |
| train      |       |       |           |
|------------|-------|-------|-----------|
| face       |       |       |           |
|------------|-------|-------|-----------|
| BG         |       |       |           |
|------------|-------|-------|-----------|
| BG + face  |       |       |           |
------------------------------------------
'''

import chainer
import numpy as np
from deepimpression2.model_50 import Deepimpression
from deepimpression2.model_50 import LastLayers
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


# train_23 = epoch_49_20
# train_35 = epoch_99_32

load_model = True
p = os.path.join(P.MODELS, 'epoch_99_32')
my_model = D.load_model(Deepimpression(), p, load_model)
my_other_model = D.load_last_layers(LastLayers(), my_model, load_model)

if load_model:
    continuefrom = 0
else:
    continuefrom = 0

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)
    my_other_model = my_other_model.to_gpu(device=C.DEVICE)

print('Initializing')

epochs = 1

test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
test_loss = []
pred_diff_test = np.zeros((epochs, 5), float)

test_steps = len(test_labels) // C.TEST_BATCH_SIZE
# test_steps = 1


def run(which, steps, all_labels, model, model2, pred_diff, loss_saving, which_data, save_all_results):
    print('steps: ', steps)
    assert(which == 'test')
    assert(which_data in ['all', 'bg', 'face'])

    loss_tmp = []
    pd_tmp = np.zeros((steps, 5), dtype=float)

    ts = time.time()
    for s in range(steps):
        print(s)

        video_names = list(all_labels.keys())
        video_path = os.path.join(P.CHALEARN30_ALL_DATA, video_names[s].split('.mp4')[0] + '.h5')
        video = h5.File(video_path, 'r')

        video_keys = list(video.keys())
        # frames = len(video_keys) - 1
        frames = 10
        label = all_labels[video_names[s]][:5].astype(np.float32)

        activations = np.zeros((frames, 256))

        if C.ON_GPU:
            label = to_gpu(label, device=C.DEVICE)
            activations = to_gpu(activations, device=C.DEVICE)

        # for f in range(frames):
        for f in range(10):
            data = video[str(f)][:]
            data = data.astype(np.float32)

            if C.ON_GPU:
                data = to_gpu(data, device=C.DEVICE)

            with cp.cuda.Device(C.DEVICE):
                with chainer.using_config('train', False):
                    # frame = np.expand_dims(data, 0)
                    _, activation = model(data)
                    activations[f] = activation.data

        activations = to_cpu(activations)
        activations = np.mean(activations, axis=0)
        activations = np.expand_dims(activations, 0).astype(np.float32)

        if C.ON_GPU:
            activations = to_gpu(activations, device=C.DEVICE)

        prediction = model2(activations)
        loss = mean_absolute_error(prediction[0], label)

        loss_tmp.append(float(loss.data))

        pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(label))

    pred_diff[e] = np.mean(pd_tmp, axis=0)
    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    loss_saving.append(loss_tmp_mean)
    print('E %d. %s loss: ' %(e, which), loss_tmp_mean,
          ' pred diff OCEAS: ', pred_diff[e],
          ' time: ', time.time() - ts)

    U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])

    if which == 'test' and save_all_results:
        U.record_loss_all_test(loss_tmp)


print('Enter training loop with validation')
for e in range(continuefrom, epochs):
    train_on = 'all'
    test_on = 'all'

    times = 1
    for i in range(1):
        if times == 1:
            save_all_results = True
        else:
            save_all_results = False

        run(which='test', steps=test_steps, all_labels=test_labels, model=my_model, model2=my_other_model,
            pred_diff=pred_diff_test, loss_saving=test_loss, which_data=test_on, save_all_results=save_all_results)

# train_23 = epoch_49_20
# train_35 = epoch_99_32
