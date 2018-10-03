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


# kruskal_wallis()


def kruskal_wallis_random(nam):
    # compare with the random baseline
    # is the difference between face-baseline and bg-baseline and all-baseline significant?
    # not cropped: 44, 45, 46
    # cropped: 47, 48, 49
    # rrnd = '/scratch/users/gabras/data/loss/testall_52.txt' # chance
    rrnd_1 = '/scratch/users/gabras/data/loss/testall_62_%s.txt' % nam # avg train
    rrnd_2 = '/scratch/users/gabras/data/loss/testall_66_%s.txt' % nam  # luminance lin reg

    # _all = '/scratch/users/gabras/data/loss/testall_61_%s.txt' % nam
    # path_bg = '/scratch/users/gabras/data/loss/testall_60_%s.txt' % nam
    # path_face = '/scratch/users/gabras/data/loss/testall_59_%s.txt' % nam

    random_load_1 = np.genfromtxt(rrnd_1, 'float')
    random_load_2 = np.genfromtxt(rrnd_2, 'float')

    # ref_load = np.genfromtxt(_all, 'float')
    # path_bg_load = np.genfromtxt(path_bg, 'float')
    # path_face_load = np.genfromtxt(path_face, 'float')


    print('all')
    value, pvalue = stats.kruskal(random_load_1, random_load_2)
    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions')
    else:
        print('Samples are likely drawn from different distributions')

    # print('face')
    # value, pvalue = stats.kruskal(random_load, path_face_load)
    # print(value, pvalue)
    # if pvalue > 0.05:
    #     print('Samples are likely drawn from the same distributions')
    # else:
    #     print('Samples are likely drawn from different distributions')
    #
    # print('bg')
    # value, pvalue = stats.kruskal(random_load, path_bg_load)
    # print(value, pvalue)
    # if pvalue > 0.05:
    #     print('Samples are likely drawn from the same distributions')
    # else:
    #     print('Samples are likely drawn from different distributions')

    # not cropped, chance, avg train
    # 44: p=0.0 p=0.0157
    # 45: p=0.0 p=0.00548
    # 46: p=6.52618061570611e-271 p=3.0034918737488883e-78

    # cropped, chance, avg train
    # 47: p=0.0     p=0.106
    # 48: p=0.0     p=7.3931979121337195e-09
    # 49: p=0.0     p=8.683701146900212e-29

    # single trait O, avg train ,           luminance
    # all:      0.017478315712775676 *      0.010746198174232571 *
    # face:     0.08856113619832962         0.05553303271880753
    # bg:       0.6984833140171955          0.8276628401245589
    # single trait C, avg train             luminance
    # all:      1.573681176472284e-05 *     6.53049634166552e-05 *
    # face:     3.9433060996943704e-07 *    1.998446861864009e-06 *
    # bg:       0.1228061601932074          0.22193827145310174
    # single trait E, avg train             luminance
    # all:      0.008003239245212037 *      0.008979161480908769 *
    # face:     0.00046253241516373324 *    0.0006126204063795387 *
    # bg:       0.5045562228670757          0.47560576204659044
    # single trait A, avg train             luminance
    # all:      0.3951223320065298          0.33208466461332287
    # face:     0.8242363130275407          0.9737319508804226
    # bg:       0.16444856065934194         0.22164797127834848
    # single trait S, avg train             luminance
    # all:      0.25445698364755176         0.25237743609781205
    # face:     0.01843581698280407 *       0.016037983837817144 *
    # bg:       0.9778875421287562          0.9803934968622353

    # linreg and avg train are not significantly different from each other

# kruskal_wallis_random('O')
# kruskal_wallis_random('C')
# kruskal_wallis_random('E')
# kruskal_wallis_random('A')
# kruskal_wallis_random('S')


def kruskal_wallis_random_between(nam):
    # compare face vs all
    # is the difference between face-baseline and bg-baseline and all-baseline significant?
    # not cropped: 44, 45, 46
    # cropped: 47, 48, 49
    # rrnd = '/scratch/users/gabras/data/loss/testall_52.txt' # chance
    rrnd = '/scratch/users/gabras/data/loss/testall_62_%s.txt' % nam # avg train

    _all = '/scratch/users/gabras/data/loss/testall_61_%s.txt' % nam
    path_bg = '/scratch/users/gabras/data/loss/testall_60_%s.txt' % nam
    path_face = '/scratch/users/gabras/data/loss/testall_59_%s.txt' % nam

    random_load = np.genfromtxt(rrnd, 'float')
    ref_load = np.genfromtxt(_all, 'float')
    path_bg_load = np.genfromtxt(path_bg, 'float')
    path_face_load = np.genfromtxt(path_face, 'float')

    diff_all = np.abs(random_load - ref_load)
    diff_face = np.abs(random_load - path_face_load)

    print('all vs face')
    value, pvalue = stats.kruskal(diff_all, diff_face)
    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions')
    else:
        print('Samples are likely drawn from different distributions')

    # O     p=0.05104840525884535
    # C     p=9.645119500327745e-09  *
    # E     p=2.5024293031449985e-13 *
    # A     p=8.361516816329688e-07 *
    # S     p=3.469432646720448e-09 *

# kruskal_wallis_random_between('S')


def any_calculate_mean_var(p):
    load = np.genfromtxt(p, 'float')
    mean = np.mean(load)
    var = np.var(load)
    print(mean, var)


# p = '/scratch/users/gabras/data/loss/testall_52.txt'
# any_calculate_mean_var(p)


def nice_print(p):
    if p > 0.05:
        print('Samples are likely drawn from the same distributions')
    else:
        print('Samples are likely drawn from different distributions')


def kruskal_wallis_mae_5_traits(t):
    targets = ['uniform', 'avg_train', 'luminance']
    assert(t in targets)

    face_data = os.path.join(P.LOG_BASE, 'testall_69.txt')  # epoch_29_34, file: testall_69.txt
    bg_data = os.path.join(P.LOG_BASE, 'testall_70.txt')  # epoch_89_33, file: testall_70.txt
    all_data = os.path.join(P.LOG_BASE, 'testall_58.txt')  # epoch_9_57, file: testall_58.txt
    uniform_data = os.path.join(P.LOG_BASE, 'testall_52.txt')  # train_52, file: testall_52.txt
    avg_train_data = os.path.join(P.LOG_BASE, 'testall_56.txt')  # train_56, testall_56.txt
    linreg_luminance_data = os.path.join(P.LOG_BASE, 'testall_65.txt')  # train_65, file: testall_65.txt

    face_load = np.genfromtxt(face_data, 'float')
    bg_load = np.genfromtxt(bg_data, 'float')
    all_load = np.genfromtxt(all_data, 'float')
    uniform_load = np.genfromtxt(uniform_data, 'float')
    avg_train_load = np.genfromtxt(avg_train_data, 'float')
    lum_load = np.genfromtxt(linreg_luminance_data, 'float')

    target_names = [uniform_load, avg_train_load, lum_load]

    which = ['face', 'bg', 'all']
    var_names = [face_load, bg_load, all_load]

    target = target_names[targets.index(t)]

    # significantly different from avg. train?
    print('--- significance testing for %s\n' % t)
    for i, w in enumerate(which):
        value, pvalue = stats.kruskal(var_names[i], target)
        print('%s: p=%f' % (w, pvalue))
        nice_print(pvalue)
        print('--------------\n')
    # --- significance testing for avg_train
    # face: p=0.012610
    # Samples are likely drawn from different distributions
    # --------------
    # bg: p=0.000594
    # Samples are likely drawn from different distributions
    # --------------
    # all: p=0.661576
    # Samples are likely drawn from the same distributions
    # --------------
    # --- significance testing for uniform
    # face: p=0.000000
    # Samples are likely drawn from different distributions
    # --------------
    # bg: p=0.000000
    # Samples are likely drawn from different distributions
    # --------------
    # all: p=0.000000
    # Samples are likely drawn from different distributions
    # --------------
    # --- significance testing for luminance
    # face: p=0.016299
    # Samples are likely drawn from different distributions
    # --------------
    # bg: p=0.000391
    # Samples are likely drawn from different distributions
    # --------------
    # all: p=0.737995
    # Samples are likely drawn from the same distributions
    # --------------


# kruskal_wallis_mae_5_traits('luminance')