import numpy as np
import os
import deepimpression2.paths as P
import matplotlib.pyplot as plt


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def record_loss(which, loss, cm_trait):
    if which == 'train':
        path = P.TRAIN_LOG
    elif which == 'val':
        path = P.VAL_LOG
    elif which == 'test':
        path = P.TEST_LOG

    line = ''

    with open(path, 'a') as mf:
        cm_trait = list(cm_trait.flatten())
        for i in range(len(cm_trait)):
            line += str(cm_trait[i])[0:4]
            line += ','

        line = line[0:-1]
        # tmp = '%s,%s\n' % (str(loss)[0:6], line)
        # print(tmp)
        mf.write('%s,%s\n' % (str(loss)[0:6], line))


def binarize(arr):
    assert(arr.ndim == 3)

    shapes = arr.shape
    new_arr = np.zeros(shapes, dtype=int)

    for i in range(shapes[0]):
        for j in range(shapes[1]):
            if arr[i][j][0] > arr[i][j][1]:
                new_arr[i][j] = [1, 0]
            else:
                new_arr[i][j] = [0, 1]

    return new_arr


def make_confusion_matrix(prediction, labels):
    # shape prediction and labels (32, 5, 2)
    # (1, 0) = left    (0, 1) = right
    prediction = binarize(prediction)
    shapes = prediction.shape
    tl, fl, tr, fr = 0, 0, 0, 0
    cm_per_trait = np.zeros((shapes[1], 4), dtype=int) # traits: OCEAS, confusions: tl, fl, tr, fr

    for i in range(shapes[0]):
        for j in range(shapes[1]):
            if labels[i][j][0] == 1:
                if prediction[i][j][0] == 1:
                    tl += 1
                    cm_per_trait[j][0] += 1
                else:
                    fl += 1
                    cm_per_trait[j][1] += 1
            elif labels[i][j][0] == 0:
                if prediction[i][j][0] == 0:
                    tr += 1
                    cm_per_trait[j][2] += 1
                else:
                    fr += 1
                    cm_per_trait[j][3] += 1

    # cm_per_trait = np.mean(cm_per_trait, axis=0)

    return [tl, fl, tr, fr], cm_per_trait


def mk_plots(which):
    assert(which in ['train', 'val', 'test'])

    if which == 'train':
        loss_path = P.TRAIN_LOG
        name = 'train'
    elif which == 'val':
        loss_path = P.VAL_LOG
        name = 'val'
    elif which == 'test':
        loss_path = P.TEST_LOG

    data = np.genfromtxt(loss_path, float, delimiter=',')
    x = np.arange(0, data.shape[0])

    # cross entropy loss plot
    y = data[:, 0]
    # plt.plot(x, y, 'r')
    # plt.title('%s cross entropy loss' % name)
    # plt.xlabel('epochs')
    # # name = os.path.join(P.FIGURES, 'train.png')
    # plt.savefig('%s.png' % name)

    # confusion matrix
    oceas = data[:, 1:len(data)].reshape((100, 5, 4))
    o = oceas[:, 0, :]
    c = oceas[:, 1, :]
    e = oceas[:, 2, :]
    a = oceas[:, 3, :]
    s = oceas[:, 4, :]





mk_plots('train')


