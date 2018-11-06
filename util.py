import numpy as np
import os
import deepimpression2.paths as P
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
from deepimpression2.savitzky_golay_filter import savitzky_golay
from scipy.signal import savgol_filter


def sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s


def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x))
    return s


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def record_loss(which, loss, cm_trait, label_stats):
    if which == 'train':
        path = P.TRAIN_LOG
    elif which == 'val':
        path = P.VAL_LOG
    elif which == 'test':
        path = P.TEST_LOG

    line = ''

    with open(path, 'a') as mf:
        cm_trait = list(cm_trait.flatten())
        # 20, 5*4, OCEAS * [tl, fl, tr, fr]
        for i in range(len(cm_trait)):
            line += str(cm_trait[i])[0:4] + ','
        # 10, 2*5, [left, right] * OCEAS
        label_stats = list(label_stats.flatten())
        for i in range(len(label_stats)):
            line += str(label_stats[i])[0:4] + ','

        line = line[0:-1]
        # tmp = '%s,%s\n' % (str(loss)[0:6], line)
        # print(tmp)
        mf.write('%s,%s\n' % (str(loss)[0:6], line))


def record_loss_sanity(which, loss, pred_diff):
    assert(which in ['train', 'val', 'test'])

    if which == 'train':
        path = P.TRAIN_LOG
    elif which == 'val':
        path = P.VAL_LOG
    elif which == 'test':
        path = P.TEST_LOG

    line = ''

    with open(path, 'a') as mf:
        for i in range(len(pred_diff)):
            line += str(pred_diff[i])[0:6] + ','

        line = line[0:-1]
        # tmp = '%s,%s\n' % (str(loss)[0:6], line)
        mf.write('%s,%s\n' % (str(loss)[0:6], line))


def record_loss_all_test(loss_tmp, trait=False):
    if trait:
        num = P.TEST_LOG.split('test_')[-1].split('.')[0]
    else:
        num = P.TEST_LOG.split('_')[-1].split('.')[0]
    path = os.path.join(P.LOG_BASE, 'testall_%s.txt' % num)
    with open(path, 'a') as my_file:
        for i in loss_tmp:
            line = '%f\n' % (i)
            my_file.write(line)


def record_all_predictions(which, preds, trait=None):
    assert (which in ['test'])
    if trait is None:
        path = P.PREDICTION_LOG
    else:
        assert(trait in ['O', 'C', 'E', 'A', 'S'])
        path = os.path.join(P.LOG_BASE, 'pred_87_%s.txt' % trait)

    if len(preds.shape) == 2:
        with open(path, 'a') as mf:
            for i in range(preds.shape[0]):
                line = ''
                for j in range(preds.shape[1]):
                    line = line + str(preds[i][j])[0:6] + ','
                line = line[0:-1] + '\n'
                mf.write(line)
    else:
        with open(path, 'a') as mf:
            for i in range(preds.shape[0]):
                line = str(preds[i])[0:6] + '\n'
                mf.write(line)


def pred_diff_trait(prediction, labels):
    # OCEAS
    diff = np.abs(prediction - labels)
    diff = np.mean(diff, axis=0)
    return diff


def binarize(arr, trait_mode='all', xe='sigmoid'):
    shapes = arr.shape
    if trait_mode == 'all':
        assert(arr.ndim == 2)

        shapes = arr.shape
        new_arr = np.zeros(shapes, dtype=int)

        for i in range(shapes[0]):
            for j in range(shapes[1]):
                if sigmoid(arr[i][j]) > 0.5:
                    new_arr[i][j] = 1
                else:
                    new_arr[i][j] = 0

    elif trait_mode == 'collapse':

        if xe == 'sigmoid':
            assert (arr.ndim == 2)
            new_arr = np.zeros(shapes, dtype=int)

            for i in range(shapes[0]):
                if sigmoid(arr[i]) > 0.5:
                    new_arr[i] = 1
                else:
                    new_arr[i] = 0
        elif xe == 'softmax':
            assert (arr.ndim == 3)
            new_arr = np.zeros((shapes[0], shapes[1]), dtype=int)

            # Truth
            # 1 = left
            # 0 = right
            #
            # gabi, don't do this with softmax
            # [0, 1] = 0 = right
            # [1, 0] = 1 = left
            #
            # softmax, do this
            # [class 0, class 1]
            # [0, 1] = 1 = left
            # [1, 0] = 0 = right

            for i in range(shapes[0]):
                if softmax(arr[i])[0] > 0.5:
                    new_arr[i] = [0, 1]
                else:
                    new_arr[i] = [1, 0]

    return new_arr


def make_confusion_matrix(prediction, labels, trait_mode='all', xe='sigmoid'):
    # shape prediction and labels (32, 5, 2)
    # (1, 0) = left    (0, 1) = right
    prediction = binarize(prediction, trait_mode, xe)
    shapes = prediction.shape
    tl, fl, tr, fr = 0, 0, 0, 0
    cm_per_trait = np.zeros((shapes[1], 4), dtype=int)  # traits: OCEAS, confusions: tl, fl, tr, fr

    if trait_mode == 'all':
        for i in range(shapes[0]):
            for j in range(shapes[1]):
                if labels[i][j] == 1:
                    if prediction[i][j] == 1:
                        tl += 1
                        cm_per_trait[j][0] += 1
                    else:
                        fl += 1
                        cm_per_trait[j][1] += 1
                elif labels[i][j] == 0:
                    if prediction[i][j] == 0:
                        tr += 1
                        cm_per_trait[j][2] += 1
                    else:
                        fr += 1
                        cm_per_trait[j][3] += 1

        return [tl, fl, tr, fr], cm_per_trait

    elif trait_mode == 'collapse':
        if xe == 'sigmoid':
            for i in range(shapes[0]):
                if int(labels[i]) == 1:
                    if int(prediction[i]) == 1:
                        tl += 1
                    else:
                        fl += 1
                elif int(labels[i]) == 0:
                    if int(prediction[i]) == 0:
                        tr += 1
                    else:
                        fr += 1
        elif xe == 'softmax':
            for i in range(shapes[0]):
                if int(labels[i]) == 1:
                    if int(prediction[i][0]) == 1:
                        tl += 1
                    else:
                        fl += 1
                elif int(labels[i]) == 0:
                    if int(prediction[i][0]) == 0:
                        tr += 1
                    else:
                        fr += 1

        return [tl, fl, tr, fr]


def mk_plots(which, num, trait=False):
    assert(which in ['train', 'val', 'test'])

    if which == 'train':
        # loss_path = P.TRAIN_LOG
        loss_path = '/scratch/users/gabras/data/loss/train_%s.txt' % num
    elif which == 'val':
        # loss_path = P.VAL_LOG
        loss_path = '/scratch/users/gabras/data/loss/val_%s.txt' % num
    elif which == 'test':
        pass
        # loss_path = P.TEST_LOG

    save_path = os.path.join(P.FIGURES, 'train_%s' % num)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # data = np.genfromtxt(loss_path, float, delimiter=',')[0:22]
    data = np.genfromtxt(loss_path, float, delimiter=',')
    x = np.arange(0, data.shape[0])

    plt.figure()
    # cross entropy loss plot
    y = data[:, 0]
    # smoothing
    smooth = False
    if smooth:
        # y_hat = savitzky_golay(y, 31, 3)  # window size, polynomial order
        y_hat = savgol_filter(y, 31, 3)
        y = y_hat
    # y = data
    plt.plot(x, y, 'r')
    if trait:
        plt.title('%s mean absolute error loss trait %s' % (which, list(num)[-1]))
    else:
        plt.title('%s mean absolute error loss' % which)
    plt.xlabel('epochs')
    plt.savefig('%s/%s.png' % (save_path, which))

    # confusion matrix
    # oceas = data[:, 1:21].reshape((data.shape[0], 5, 4))
    #
    # traits = ['o', 'c', 'e', 'a', 's']
    # for i in range(5):
    #     plt.figure()
    #     t = oceas[:, i, :]
    #     tl = t[:, 0]
    #     fl = t[:, 1]
    #     tr = t[:, 2]
    #     fr = t[:, 3]
    #     plt.plot(x, tl, label='TL')
    #     plt.plot(x, fl, label='FL')
    #     plt.plot(x, tr, label='TR')
    #     plt.plot(x, fr, label='FR')
    #
    #     plt.legend()
    #
    #     plt.title('confusion matrix trait "%s" %s' % (traits[i], which))
    #     plt.xlabel('epochs')
    #
    #     plt.savefig('%s/cm_%s_%s.png' % (save_path, traits[i], which))


# n = '64'
# mk_plots('train', n)
# mk_plots('val', n)
#
# n = '33'
# mk_plots('train', n)
# mk_plots('val', n)
#
# n = '59_C'
# mk_plots('train', n)
# mk_plots('val', n)
#
# n = '59_E'
# mk_plots('train', n)
# mk_plots('val', n)
#
# n = '59_A'
# mk_plots('train', n)
# mk_plots('val', n)

# n = '61_O'
# mk_plots('train', n)
# mk_plots('val', n)
# n = '61_C'
# mk_plots('train', n)
# mk_plots('val', n)
# n = '61_E'
# mk_plots('train', n)
# mk_plots('val', n)
# n = '61_A'
# mk_plots('train', n)
# mk_plots('val', n)
# n = '61_S'
# mk_plots('train', n)
# mk_plots('val', n)

# mk_plots('train', '95_O')
# mk_plots('val', '95_O')
# mk_plots('train', '95_C')
# mk_plots('val', '95_C')
# mk_plots('train', '95_E')
# mk_plots('val', '95_E')
# mk_plots('train', '95_A')
# mk_plots('val', '95_A')
# mk_plots('train', '95_S')
# mk_plots('val', '95_S')