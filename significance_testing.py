import numpy as np
import os
import deepimpression2.paths as P
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
from scipy import stats


# normality testing for train_xx

num_list_1 = [45, 46]
num_list_2 = [48, 49]


def plot_both():
    for i in num_list_2:
        ref = '/scratch/users/gabras/data/loss/testall_47.txt'
        path = '/scratch/users/gabras/data/loss/testall_%d.txt' % i

        ref_load = np.genfromtxt(ref, 'float')
        path_load = np.genfromtxt(path, 'float')

        save_path = os.path.join(P.FIGURES, 'train_%s' % i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # x = np.arange(0, ref_load.shape[0])
        x = np.arange(0, 100)

        plt.figure()

        y = ref_load[:100]
        plt.plot(x, y, 'b', label='all')

        if i == 45 or i == 48:
            lab = 'bg'
        else:
            lab = 'face'

        y = path_load[:100]
        plt.plot(x, y, 'r', label=lab)

        plt.legend()

        plt.title('test MAE loss: train_47 vs. train_%d' % i)
        plt.xlabel('epochs')
        plt.savefig('%s/%s.png' % (save_path, 'test_%d' % i))


def compare_diff_to_normal():
    for i in num_list_1:
        ref = '/scratch/users/gabras/data/loss/testall_44.txt'
        path = '/scratch/users/gabras/data/loss/testall_%d.txt' % i

        ref_load = np.genfromtxt(ref, 'float')
        path_load = np.genfromtxt(path, 'float')

        diff = path_load - ref_load

        alpha = 0.05
        norm = np.random.normal(0, 1, size=diff.shape[0])

        x = np.concatenate((norm, diff))

        k2, p = stats.normaltest(x)

        print('%d p=%f' % (i, p))

        if p < alpha:  # null hypothesis: x comes from a normal distribution
            print('diff does not come from a normal distribution')
        else:
            print('maybe?')


def plot_diff_norm():
    for i in num_list_2:
        ref = '/scratch/users/gabras/data/loss/testall_47.txt'
        path = '/scratch/users/gabras/data/loss/testall_%d.txt' % i

        ref_load = np.genfromtxt(ref, 'float')
        path_load = np.genfromtxt(path, 'float')

        diff = path_load - ref_load

        save_path = os.path.join(P.FIGURES, 'train_%s' % i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        plt.figure()
        n, bins, patches = plt.hist(diff, 50, density=True, facecolor='g', alpha=0.75)
        plt.grid(True)

        if i == 45 or i == 48:
            lab = 'bg'
        else:
            lab = 'face'

        plt.title('histogram dif %s' % lab)
        plt.savefig('%s/%s.png' % (save_path, 'histdiff_%d' % i))


# lets assume that these are normally distributed

def calculate_mean_var():
    ref = '/scratch/users/gabras/data/loss/testall_44.txt'
    path45 = '/scratch/users/gabras/data/loss/testall_45.txt'
    path46 = '/scratch/users/gabras/data/loss/testall_46.txt'

    ref_load = np.genfromtxt(ref, 'float')
    path45_load = np.genfromtxt(path45, 'float')
    path46_load = np.genfromtxt(path46, 'float')

    diff45 = path45_load - ref_load
    diff46 = path46_load - ref_load

    mean45 = np.mean(diff45)
    mean46 = np.mean(diff46)

    var45 = np.var(diff45)
    var46 = np.var(diff46)

    print(mean45, var45, ' and ', mean46, var46)


def significance_test():
    _all = '/scratch/users/gabras/data/loss/testall_47.txt'
    path_bg = '/scratch/users/gabras/data/loss/testall_48.txt'
    path_face = '/scratch/users/gabras/data/loss/testall_49.txt'

    ref_load = np.genfromtxt(_all, 'float')
    path_bg_load = np.genfromtxt(path_bg, 'float')
    path_face_load = np.genfromtxt(path_face, 'float')

    diff_bg = path_bg_load - ref_load
    diff_face = path_face_load - ref_load

    value, pvalue = stats.ttest_ind(diff_bg, diff_face, equal_var=False)
    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions (fail to reject H0)')
    else:
        print('Samples are likely drawn from different distributions (reject H0)')

    # non-cropped bg vs. face
    # p = 2.4492867829504012e-70
    # Samples are likely drawn from different distributions

    # cropped bg vs. face
    # p = 1.1189541618826393e-08
    # Samples are likely drawn from different distributions


# significance_test()
