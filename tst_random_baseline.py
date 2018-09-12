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
# repetitions = 1001 # seed 42
repetitions = 1000 # seed 6

labels = D.basic_load_personality_labels(which)

mother_seed = 6
seeds = np.random.RandomState(mother_seed).randint(low=0, high=10000, size=repetitions)

all_diff = np.zeros((repetitions, labels.shape[0]), dtype=np.float32)

for i, s in enumerate(seeds):
    random_predictions = np.random.RandomState(s).uniform(0, 1, labels.shape)
    diff = np.absolute(labels - random_predictions)
    diff = np.mean(diff, axis=1)
    all_diff[i] = diff

all_diff = np.mean(all_diff, axis=0)
# U.record_loss_all_test(all_diff)

# save_path = os.path.join(P.FIGURES, 'train_52')
# U.safe_mkdir(save_path)
#
# plt.figure()
# n, bins, patches = plt.hist(all_diff, 50, density=True, facecolor='g', alpha=0.75)
# plt.grid(True)
# plt.title('histogram MAE random - %s repetitions %d' % (which, repetitions))
# plt.savefig('%s/%s.png' % (save_path, 'histdiff_%d' % repetitions))

