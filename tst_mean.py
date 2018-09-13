# get mean for each of 5 traits and use this as baseline to calculate test MAE

import numpy as np
import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D
import deepimpression2.util as U
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


print('Initializing')

test_labels = D.basic_load_personality_labels('test')
train_labels = D.basic_load_personality_labels('train')

mean_label = np.mean(train_labels, axis=0)

diff = np.absolute(test_labels - mean_label)

diff = np.mean(diff, axis=1)

# U.record_loss_all_test(diff)

print('loss: %f' % np.mean(diff))

save_path = os.path.join(P.FIGURES, 'train_56')
U.safe_mkdir(save_path)
#
plt.figure()
n, bins, patches = plt.hist(diff, 50, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.title('histogram MAE avg train - test')
plt.savefig('%s/%s.png' % (save_path, 'histdiff'))