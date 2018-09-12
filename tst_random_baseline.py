# Q: what is the influence of background on personality prediction?

'''
Compare with test
Generate 1676 * 5 random values
Get MAE for each video
do this 1000 times
Plot distribution
See 95th percentile on each side
See if test losses are outside or not
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


print('Initializing')

which = 'test'
repetitions = 1001
# repetitions = 1000 # seed 6

labels = D.basic_load_personality_labels(which)

mother_seed = 42
seeds = np.random.RandomState(mother_seed).randint(low=0, high=10000, size=repetitions)

all_diff = np.zeros((repetitions, labels.shape[0]), dtype=np.float32)

for i, s in enumerate(seeds):
    random_predictions = np.random.RandomState(s).uniform(0, 1, labels.shape)
    diff = np.absolute(labels - random_predictions)
    diff = np.mean(diff, axis=1)
    all_diff[i] = diff

all_diff = np.mean(all_diff, axis=0)

# save_path = os.path.join(P.FIGURES, 'train_52')
# U.safe_mkdir(save_path)
#
# plt.figure()
# n, bins, patches = plt.hist(all_diff, 50, density=True, facecolor='g', alpha=0.75)
# plt.grid(True)
# plt.title('histogram MAE random - %s repetitions %d' % (which, repetitions))
# plt.savefig('%s/%s.png' % (save_path, 'histdiff_%d' % repetitions))




# def run(which, steps, all_labels, model, model2, pred_diff, loss_saving, which_data, save_all_results):
#     print('steps: ', steps)
#     assert(which == 'test')
#     assert(which_data in ['all', 'bg', 'face'])
#
#     loss_tmp = []
#     pd_tmp = np.zeros((steps, 5), dtype=float)
#
#     ts = time.time()
#     for s in range(steps):
#         print(s)
#
#         video_names = list(all_labels.keys())
#         video_path = os.path.join(P.CHALEARN30_ALL_DATA, video_names[s].split('.mp4')[0] + '.h5')
#         video = h5.File(video_path, 'r')
#
#         video_keys = list(video.keys())
#         # frames = len(video_keys) - 1
#         frames = 10
#         label = all_labels[video_names[s]][:5].astype(np.float32)
#
#         activations = np.zeros((frames, 256))
#
#         if C.ON_GPU:
#             label = to_gpu(label, device=C.DEVICE)
#             activations = to_gpu(activations, device=C.DEVICE)
#
#         # for f in range(frames):
#         for f in range(10):
#             data = video[str(f)][:]
#             data = data.astype(np.float32)
#
#             if C.ON_GPU:
#                 data = to_gpu(data, device=C.DEVICE)
#
#             with cp.cuda.Device(C.DEVICE):
#                 with chainer.using_config('train', False):
#                     # frame = np.expand_dims(data, 0)
#                     _, activation = model(data)
#                     activations[f] = activation.data
#
#         activations = to_cpu(activations)
#         activations = np.mean(activations, axis=0)
#         activations = np.expand_dims(activations, 0).astype(np.float32)
#
#         if C.ON_GPU:
#             activations = to_gpu(activations, device=C.DEVICE)
#
#         prediction = model2(activations)
#         loss = mean_absolute_error(prediction[0], label)
#
#         loss_tmp.append(float(loss.data))
#
#         pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(label))
#
#     pred_diff[e] = np.mean(pd_tmp, axis=0)
#     loss_tmp_mean = np.mean(loss_tmp, axis=0)
#     loss_saving.append(loss_tmp_mean)
#     print('E %d. %s loss: ' %(e, which), loss_tmp_mean,
#           ' pred diff OCEAS: ', pred_diff[e],
#           ' time: ', time.time() - ts)
#
#     U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])
#
#     if which == 'test' and save_all_results:
#         U.record_loss_all_test(loss_tmp)

#
# print('Enter training loop with validation')
# for e in range(continuefrom, epochs):
#     train_on = 'all'
#     test_on = 'all'
#
#     times = 1
#     for i in range(1):
#         if times == 1:
#             save_all_results = True
#         else:
#             save_all_results = False
#
#         run(which='test', steps=test_steps, all_labels=test_labels, model=my_model, model2=my_other_model,
#             pred_diff=pred_diff_test, loss_saving=test_loss, which_data=test_on, save_all_results=save_all_results)

# train_23 = epoch_49_20
# train_35 = epoch_99_32
