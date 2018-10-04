import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import deepimpression2.paths as P
import numpy as np
import os


save_path = P.PAPER_PLOTS


def basic_mae_5_traits():
    # names = ['face', 'background', 'all', 'avg. train', 'lin.reg. luminance']
    names = ['face', 'background', 'all', 'avg. train']
    values = [0.116, 0.127, 0.119, 0.122]

    plt.figure()
    x_pos = np.arange(len(names))
    bar1 = plt.bar(x_pos, values)

    plt.xticks(x_pos, names, rotation='45')
    plt.ylabel('mean absolute error')
    plt.ylim((0.11, 0.13))
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


basic_mae_single_traits()