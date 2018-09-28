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


def normalize_data(arr):
    mn = np.min(arr)
    arr -= mn
    mx = np.max(arr)
    arr /= mx
    return arr


def compare_diff_to_normal():
    for i in num_list_2:
        ref = '/scratch/users/gabras/data/loss/testall_47.txt'
        path = '/scratch/users/gabras/data/loss/testall_%d.txt' % i

        ref_load = np.genfromtxt(ref, 'float')
        path_load = np.genfromtxt(path, 'float')

        diff = path_load - ref_load
        diff = normalize_data(diff)

        alpha = 0.05
        norm = np.random.normal(0, 1, size=diff.shape[0])

        x = np.concatenate((norm, diff))

        k2, p = stats.normaltest(x)

        print('%d p=%f' % (i, p))

        if p < alpha:  # null hypothesis: x comes from a normal distribution
            print('diff does not come from a normal distribution')
        else:
            print('maybe?')


# compare_diff_to_normal()


def plot_diff_norm():
    for i in num_list_2:
        ref = '/scratch/users/gabras/data/loss/testall_47.txt'
        path = '/scratch/users/gabras/data/loss/testall_%d.txt' % i

        ref_load = np.genfromtxt(ref, 'float')
        path_load = np.genfromtxt(path, 'float')

        diff = path_load - ref_load
        diff = normalize_data(diff)
        print(diff.shape)

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
        plt.savefig('%s/%s.png' % (save_path, 'histdiff_%d_2' % i))


# plot_diff_norm()


# lets assume that these are normally distributed

def calculate_mean_var():
    ref = '/scratch/users/gabras/data/loss/testall_47.txt'
    path45 = '/scratch/users/gabras/data/loss/testall_48.txt'
    path46 = '/scratch/users/gabras/data/loss/testall_49.txt'

    ref_load = np.genfromtxt(ref, 'float')
    path45_load = np.genfromtxt(path45, 'float')
    path46_load = np.genfromtxt(path46, 'float')

    diff45 = normalize_data(path45_load - ref_load)
    diff46 = normalize_data(path46_load - ref_load)

    mean45 = np.mean(diff45)
    mean46 = np.mean(diff46)

    var45 = np.var(diff45)
    var46 = np.var(diff46)

    print(mean45, var45, ' and ', mean46, var46)
    # 45 46 -- not cropped
    # 0.41189869252511124 0.011381779310396646  and  0.5415425225184934 0.02147196975321762
    # 47 48 -- cropped
    # 0.4273576817938818 0.01963095162241388  and  0.47092922809647225 0.015323876850164547


# calculate_mean_var()


def significance_test():
    _all = '/scratch/users/gabras/data/loss/testall_44.txt'
    path_bg = '/scratch/users/gabras/data/loss/testall_45.txt'
    path_face = '/scratch/users/gabras/data/loss/testall_46.txt'

    ref_load = np.genfromtxt(_all, 'float')
    path_bg_load = np.genfromtxt(path_bg, 'float')
    path_face_load = np.genfromtxt(path_face, 'float')

    diff_bg = normalize_data(path_bg_load - ref_load)
    diff_face = normalize_data(path_face_load - ref_load)

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

    # normalized diff

    # cropped bg vs. face
    # p = 2.7322686025278617e-21
    # Samples are likely drawn from different distributions

    # non-cropped bg vs. face
    # p = 2.7011427057418338e-166
    # Samples are likely drawn from different distributions


# significance_test()

# Let's assume data is not from normal distribution, which is probably the case given the results of normaltest


def kolmogorov_smirnov():
    _all = '/scratch/users/gabras/data/loss/testall_47.txt'
    path_bg = '/scratch/users/gabras/data/loss/testall_48.txt'
    path_face = '/scratch/users/gabras/data/loss/testall_49.txt'

    ref_load = np.genfromtxt(_all, 'float')
    path_bg_load = np.genfromtxt(path_bg, 'float')
    path_face_load = np.genfromtxt(path_face, 'float')

    diff_bg = np.abs(path_bg_load - ref_load)
    diff_face = np.abs(path_face_load - ref_load)

    value, pvalue = stats.ks_2samp(diff_bg, diff_face)

    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions')
    else:
        print('Samples are likely drawn from different distributions')

    # normalized diff
    # 45, 46 -- not cropped
    # p = 2.864512141804348e-167
    # Samples are likely drawn from different distributions
    # 48, 49 -- cropped
    # p = 4.9692098719771285e-28
    # Samples are likely drawn from different distributions

# kolmogorov_smirnov()


def kruskal_wallis():
    _all = '/scratch/users/gabras/data/loss/testall_47.txt'
    path_bg = '/scratch/users/gabras/data/loss/testall_48.txt'
    path_face = '/scratch/users/gabras/data/loss/testall_49.txt'

    ref_load = np.genfromtxt(_all, 'float')
    path_bg_load = np.genfromtxt(path_bg, 'float')
    path_face_load = np.genfromtxt(path_face, 'float')

    diff_bg = path_bg_load - ref_load
    diff_face = path_face_load - ref_load

    # value, pvalue = stats.kruskal(diff_bg, diff_face)
    value, pvalue = stats.kruskal(path_face_load, path_bg_load)

    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions')
    else:
        print('Samples are likely drawn from different distributions')

    # 45, 46 -- not cropped
    # p = 5.53712529445153e-75
    # Samples are likely drawn from different distributions

    # 48, 49 -- cropped
    # p = 1.937016396570183e-10
    # Samples are likely drawn from different distributions


kruskal_wallis()


def kruskal_wallis_random():
    # compare with the random baseline
    # is the difference between face-baseline and bg-baseline and all-baseline significant?
    # not cropped: 44, 45, 46
    # cropped: 47, 48, 49
    # rrnd = '/scratch/users/gabras/data/loss/testall_52.txt' # chance
    rrnd = '/scratch/users/gabras/data/loss/testall_56.txt' # avg train

    _all = '/scratch/users/gabras/data/loss/testall_44.txt'
    path_bg = '/scratch/users/gabras/data/loss/testall_45.txt'
    path_face = '/scratch/users/gabras/data/loss/testall_46.txt'

    random_load = np.genfromtxt(rrnd, 'float')
    ref_load = np.genfromtxt(_all, 'float')
    path_bg_load = np.genfromtxt(path_bg, 'float')
    path_face_load = np.genfromtxt(path_face, 'float')

    value, pvalue = stats.kruskal(random_load, path_face_load)

    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions')
    else:
        print('Samples are likely drawn from different distributions')

    # not cropped, chance, avg train
    # 44: p=0.0 p=0.0157
    # 45: p=0.0 p=0.00548
    # 46: p=6.52618061570611e-271 p=3.0034918737488883e-78

    # cropped, chance, avg train
    # 47: p=0.0     p=0.106
    # 48: p=0.0     p=7.3931979121337195e-09
    # 49: p=0.0     p=8.683701146900212e-29

# kruskal_wallis_random()


def any_calculate_mean_var(p):
    load = np.genfromtxt(p, 'float')
    mean = np.mean(load)
    var = np.var(load)
    print(mean, var)


# p = '/scratch/users/gabras/data/loss/testall_52.txt'
# any_calculate_mean_var(p)
