import numpy as np
import os
from scipy import stats
import h5py as h5
from shutil import copyfile
from tqdm import tqdm

import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D


def find_index():
    key_where = '4ZlcaXadwlo.005.mp4'
    labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    for i, k in enumerate(labels.keys()):
        if k == key_where:
            print(i)
            break

# find_index()


def fix_labels(labels):
    ignore = 173
    labels = np.delete(labels, ignore, axis=0)
    return labels


def correlations_ground_truth(trait, name):
    traits_all = ['O', 'C', 'E', 'A', 'S']
    index = traits_all.index(trait)
    target = D.basic_load_personality_labels('test')
    target = target[:, index]
    target = fix_labels(target)

    path = os.path.join(P.LOG_BASE, name)
    predictions = np.genfromtxt(path, delimiter=',', dtype=float)

    r, p = stats.pearsonr(predictions, target)

    print(name, 'r: ', r, 'p: ', p)


def corr_face_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_109_O.txt', 'pred_109_C.txt', 'pred_109_E.txt', 'pred_109_A.txt', 'pred_109_S.txt']  # frame=6
    models = ['pred_132_O.txt', 'pred_132_C.txt', 'pred_132_E.txt', 'pred_132_A.txt', 'pred_132_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])


# corr_face_single_deepimpression()


def corr_bg_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_110_O.txt', 'pred_110_C.txt', 'pred_110_E.txt', 'pred_110_A.txt', 'pred_110_S.txt']  # frame=6
    models = ['pred_133_O.txt', 'pred_133_C.txt', 'pred_133_E.txt', 'pred_133_A.txt', 'pred_133_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])

# corr_bg_single_deepimpression()

def corr_all_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_112_O.txt', 'pred_112_C.txt', 'pred_112_E.txt', 'pred_112_A.txt', 'pred_112_S.txt']  # frame=6
    # models = ['pred_118_O.txt', 'pred_118_C.txt', 'pred_118_E.txt', 'pred_118_A.txt', 'pred_118_S.txt']  # frame=30
    # models = []  # frame=10
    models = ['pred_134_O.txt', 'pred_134_C.txt', 'pred_134_E.txt', 'pred_134_A.txt', 'pred_134_S.txt']  # frame=10

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])
# 
# corr_all_single_deepimpression()

def correlations_resnet_ground_truth(trait, name):
    traits_all = ['O', 'C', 'E', 'A', 'S']
    index = traits_all.index(trait)
    target = D.basic_load_personality_labels('test')
    target = target[:, index]

    path = os.path.join(P.LOG_BASE, name)
    predictions = np.genfromtxt(path, delimiter=',', dtype=float)
    predictions = predictions[:, index]

    r, p = stats.pearsonr(predictions, target)

    print(name, 'r: ', r, 'p: ', p)


def corr_face_resnet():
    traits = ['O', 'C', 'E', 'A', 'S']
    for i in range(5):
        correlations_resnet_ground_truth(traits[i], 'pred_114.txt')


def corr_bg_resnet():
    traits = ['O', 'C', 'E', 'A', 'S']
    for i in range(5):
        correlations_resnet_ground_truth(traits[i], 'pred_115.txt')


def how_many_frames():
    total = 0
    id_frames = h5.File(P.NUM_FRAMES, 'r')
    labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    for k in labels.keys():
        k = k.split('.mp4')[0]
        total += id_frames[k][0]
    # 309973
    p = '/scratch/users/gabras/data/chalearn30/all_data'

    totsize = 0
    for k in labels.keys():
        name = k.split('.mp4')[0] + '.h5'
        path = os.path.join(p, name)
        totsize += os.path.getsize(path)

    # 252936585160/1024/1024/1024 = 236GB

    for k in tqdm(labels.keys()):
        name = k.split('.mp4')[0] + '.h5'
        d = '/home/gabras/chalearn_test_data'
        src = os.path.join(p, name)
        dst = os.path.join(d, name)
        copyfile(src, dst)


# how_many_frames()








