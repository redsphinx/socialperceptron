# get mean for each of 5 traits and use this as baseline to calculate test MAE

import numpy as np
import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D
import deepimpression2.util as U
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import stats



def mae():
    print('Initializing')

    trait_type = 'not_single'


    if trait_type == 'single':
        which = 'S'
        test_labels = D.basic_load_personality_labels('test')
        train_labels = D.basic_load_personality_labels('train')

        mean_label = np.mean(train_labels, axis=0)

        diff = np.absolute(test_labels - mean_label)

        all_traits = ['O','C','E','A','S']
        t = all_traits.index(which)
        diff = diff[:, t]
        U.record_loss_all_test(diff, trait=True)

        # diff = np.mean(diff, axis=0)
        # print(diff)

        print('loss: %f' % np.mean(diff, axis=0))

        save_path = os.path.join(P.FIGURES, 'train_62')
        U.safe_mkdir(save_path)
        #
        plt.figure()
        n, bins, patches = plt.hist(diff, 50, density=True, facecolor='g', alpha=0.75)
        plt.grid(True)
        plt.title('histogram MAE avg train - test trait %s' % which)
        plt.savefig('%s/%s_%s.png' % (save_path, 'histdiff', which))
    else:
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


def pearson_r(tr=None):
    print('Initializing')

    if tr is not None:
        which = tr
        test_labels = D.basic_load_personality_labels('test')
        train_labels = D.basic_load_personality_labels('train')

        mean_label = np.mean(train_labels, axis=0)
        prediction = np.tile(mean_label, (test_labels.shape[0], 1))

        all_traits = ['O', 'C', 'E', 'A', 'S']
        t = all_traits.index(which)
        # U.record_loss_all_test(prediction, trait=True)

        # TODO: idk if pearson is the good test for this
        # cor_coeff = stats.pearsonr(test_labels[:, t], prediction[:, t]*1.00000001)[0]
        cor_coeff = np.corrcoef(test_labels[:, t], prediction[:, t])

        print('pearson correlation coef trait %s: %f' % (tr, cor_coeff))
    else:
        test_labels = D.basic_load_personality_labels('test')
        train_labels = D.basic_load_personality_labels('train')

        mean_label = np.mean(train_labels, axis=0)
        prediction = np.tile(mean_label, (test_labels.shape[0], 1))
        prediction = np.array(list(np.mean(prediction, axis=1)))

        # U.record_loss_all_test(np.mean(prediction, axis=1))

        cor_coeff = stats.pearsonr(np.mean(test_labels, axis=1), np.mean(prediction, axis=1))[0]

        print('pearson correlation coef: %s' % (str(cor_coeff)))

        # pearson correlation coef: -1.684251e-07


# pearson_r('O')