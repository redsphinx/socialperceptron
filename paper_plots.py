import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import deepimpression2.paths as P
import numpy as np
import os
import seaborn as sns
import pandas as pd
from deepimpression2.chalearn30.data_utils import basic_load_personality_labels


save_path = P.PAPER_PLOTS


def basic_mae_5_traits():
    # names = ['face', 'background', 'all', 'avg. train', 'lin.reg. luminance']
    # names = ['face', 'background', 'all', 'avg. train']
    # values = [0.116, 0.127, 0.119, 0.122]

    # with decay=0.001 for 'all'
    save_path = os.path.join(P.PAPER_PLOTS, 'weight_decay_all')
    names = ['face', 'background', 'all', 'luminance']
    values = [0.1156, 0.1274, 0.1152, 0.1219]

    plt.figure()
    x_pos = np.arange(len(names))
    bar1 = plt.bar(x_pos, values)

    plt.xticks(x_pos, names, rotation='45')
    plt.ylabel('mean absolute error')
    plt.ylim((0.10, 0.14))
    plt.title('MAE averaged over all traits')
    plt.subplots_adjust(bottom=0.2)

    # Add counts above the two bar graphs

    for i, rect in enumerate(bar1):
        if i in [0, 1]:
            star = ' *'
        else:
            star = ''
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0,
                 height,
                 '%s%s' % (str(values[i]), star),
                 ha='center', va='bottom')

    plt.savefig('%s/%s.png' % (save_path, 'plt_mae_5_traits_basic'))


# basic_mae_5_traits()

def sns_basic_mae_5_traits():
    # with decay=0.001 for 'all'

    save_path = os.path.join(P.PAPER_PLOTS, 'weight_decay_all', 'sns.png')

    d = {'models': ['luminance', 'face', 'background', 'all'], 'MAE': [0.1219, 0.1156, 0.1274, 0.1152]}
    df = pd.DataFrame(data=d)

    sns.set(style='whitegrid')

    fig, axs = plt.subplots(ncols=2, figsize=(11, 5))

    axs[0].set_ylim([0.10, 0.14])
    sns.barplot(data=df, x='models', y='MAE', ax=axs[0])

    sig = np.zeros((4, 4))
    sig[1, 0] = 6.81e-4
    sig[2, 0] = 2.81e-8
    sig[2, 1] = 1.03e-12
    sig[3, 0] = 4.56e-11
    sig[3, 1] = 0.68
    sig[3, 2] = 1.48e-40

    ds = pd.DataFrame({'Luminance': sig[:, 0], 'Face': sig[:, 1], 'Background': sig[:, 2], 'All': sig[:, 3]}, index=['luminance', 'face', 'background', 'all'])
    mask = np.zeros_like(sig, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(ds, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, annot_kws={"size":8},
                square=True, linewidths=.5, ax=axs[1], cbar=False)

    def annotateBars(row, ax=axs[0]):
        for p in ax.patches:
            ax.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=8, color='black', xytext=(0, 10),
                        textcoords='offset points')

    ds.apply(annotateBars)

    for t in axs[1].yaxis.get_major_ticks():
        t.label.set_fontsize(8)
    for t in axs[1].xaxis.get_major_ticks():
        t.label.set_fontsize(8)

    plt.yticks(rotation=45)

    fig.savefig(save_path)


# sns_basic_mae_5_traits()



def boxplot_mae_5_traits():
    # paths
    face_data = os.path.join(P.LOG_BASE, 'testall_69.txt')  # epoch_29_34, file: testall_69.txt
    bg_data = os.path.join(P.LOG_BASE, 'testall_70.txt')  # epoch_89_33, file: testall_70.txt
    all_data = os.path.join(P.LOG_BASE, 'testall_58.txt')  # epoch_9_57, file: testall_58.txt
    uniform_data = os.path.join(P.LOG_BASE, 'testall_52.txt') # train_52, file: testall_52.txt
    avg_train_data = os.path.join(P.LOG_BASE, 'testall_56.txt')  # train_56, testall_56.txt
    linreg_luminance_data = os.path.join(P.LOG_BASE, 'testall_65.txt')  # train_65, file: testall_65.txt

    # load it
    face_load = np.genfromtxt(face_data, 'float')
    bg_load = np.genfromtxt(bg_data, 'float')
    all_load = np.genfromtxt(all_data, 'float')
    uniform_load = np.genfromtxt(uniform_data, 'float')
    avg_train_load = np.genfromtxt(avg_train_data, 'float')
    lum_load = np.genfromtxt(linreg_luminance_data, 'float')

    # data = [face_load, bg_load, all_load, uniform_load, avg_train_load, lum_load]
    data = [face_load, bg_load, all_load, avg_train_load, lum_load]

    plt.figure()
    plt.boxplot(data, showfliers=False)

    # names = ['', 'face', 'background', 'all', 'uniform', 'avg. train', 'lin.reg. luminance']
    names = ['', 'face', 'background', 'all', 'avg. train', 'lin.reg. luminance']
    x_pos = np.arange(len(names))
    plt.xticks(x_pos, names, rotation='vertical')
    plt.ylabel('mean absolute error')


    plt.title('MAE averaged over all traits')
    plt.subplots_adjust(bottom=0.25)

    plt.text(100, 100, 'HII', fontsize=100)

    plt.savefig('%s/%s.png' % (save_path, 'plt_mae_5_traits_boxplot'))


# boxplot_mae_5_traits()


def basic_mae_single_traits():
    # names = ['face', 'background', 'all', 'avg. train', 'lin.reg. luminance']
    names = ['face', 'background', 'all', 'avg. train']

    f = [0.112, 0.114, 0.115, 0.11, 0.12]
    b = [0.12, 0.126, 0.13, 0.113, 0.129]
    a = [0.11, 0.116, 0.117, 0.107, 0.125]
    at = [0.118, 0.129, 0.127, 0.108, 0.127]

    o_values = [0.112, 0.12, 0.11, 0.118]
    c_values = [0.114, 0.126, 0.116, 0.129]
    e_values = [0.115, 0.13, 0.117, 0.127]
    a_values = [0.11, 0.113, 0.107, 0.108]
    s_values = [0.12, 0.129, 0.125, 0.127]

    traits = ['O', 'C', 'E', 'A', 'S']

    sig_face = [1, 2, 4]
    sig_all = [0, 1, 2]

    # plt.figure()
    fig, ax = plt.subplots()
    x_pos = np.arange(len(traits))
    w = 0.19
    bar_face = ax.bar(x_pos+w, f, w)
    bar_bg = ax.bar(x_pos+2*w, b, w)
    bar_all = ax.bar(x_pos+3*w, a, w)
    bar_at = ax.bar(x_pos+4*w, at, w)
    ax.set_xticks(x_pos + 2.5*w)
    ax.set_xticklabels(('O', 'C', 'E', 'A', 'S'))
    # ax.autoscale_view()

    # plt.xticks(x_pos, traits)
    # plt.ylabel('mean absolute error')
    plt.ylabel('mean absolute error')
    plt.ylim((0.10, 0.1325))
    plt.title('MAE individual traits')
    # plt.subplots_adjust(bottom=0.2)

    all_bars = [bar_face, bar_bg, bar_all, bar_at]

    for barp in all_bars:
        for i, rect in enumerate(barp):
            if i in [0, 1]:
                star = '*'
            else:
                star = ''
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0,
                     height,
                     '%s%s' % (str(f[i]), star),
                     ha='center', va='bottom')

    plt.savefig('%s/%s.png' % (save_path, 'plt_mae_single_traits_basic'))


# basic_mae_single_traits()
# p_values_bar_plots()


def plot_mae_predictions_agreeableness():
    # make 3 plots, face, bg, all
    # x axis is predictions, y axis is labels
    # face_path = os.path.join(P.LOG_BASE, 'pred_97.txt')
    # bg_path = os.path.join(P.LOG_BASE, 'pred_99.txt')
    # all_path = os.path.join(P.LOG_BASE, 'pred_94.txt')
    face_path = os.path.join(P.LOG_BASE, 'pred_85_A.txt')
    bg_path = os.path.join(P.LOG_BASE, 'pred_86_A.txt')
    all_path = os.path.join(P.LOG_BASE, 'pred_84_A.txt')

    paths = [face_path, bg_path, all_path]
    names = ['face', 'bg', 'all']

    agreeableness = 3
    ground_truth = basic_load_personality_labels('test')[:, agreeableness]

    for m in range(3):
        plt.figure()
        x = ground_truth
        # y = np.genfromtxt(paths[m], float, delimiter=',')[:, agreeableness]
        y = np.genfromtxt(paths[m], float, delimiter=',')
        n = names[m]

        clr = np.abs(x - y) #np.abs(x - y)

        sc = plt.scatter(x, y, s=3, c=clr)
        plt.plot(x, x, 'r')

        plt.colorbar(sc)
        plt.title('single Agreeableness: %s vs. labels' % (n))
        plt.xlabel('labels')
        plt.ylabel('predictions')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')

        # plt.savefig('%s/%s.png' % (save_path, '5_traits_A_pred_vs_labels_%s' % n))
        plt.savefig('%s/%s.png' % (save_path, 'single_traits_A_pred_vs_labels_%s' % n))


# plot_mae_predictions_agreeableness()

def plot_labels():
    trait = 3 # agreeableness
    ground_truth = basic_load_personality_labels('test')[:, trait]
    # np.ndarray.sort(ground_truth)
    plt.figure()
    # x = np.arange(0, ground_truth.shape[0])
    # sc = plt.scatter(x, ground_truth, s=2, c=ground_truth)
    plt.hist(ground_truth, bins=5)
    # plt.colorbar(sc)
    plt.title('agreeableness labels')
    # plt.ylabel('label values')
    plt.savefig('%s/%s.png' % (save_path, 'agreeableness_hist'))


# plot_labels()
