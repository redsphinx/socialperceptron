import numpy as np
import os
from scipy import stats

import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D


def correlations_ground_truth(trait, name):
    traits_all = ['O', 'C', 'E', 'A', 'S']
    index = traits_all.index(trait)
    target = D.basic_load_personality_labels('test')
    target = target[:, index]

    path = os.path.join(P.LOG_BASE, name)
    predictions = np.genfromtxt(path, delimiter=',', dtype=float)

    r, p = stats.pearsonr(predictions, target)

    print(name, 'r: ', r, 'p: ', p)


def corr_face_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_109_O.txt', 'pred_109_C.txt', 'pred_109_E.txt', 'pred_109_A.txt', 'pred_109_S.txt']  # frame=6
    models = ['pred_116_O.txt', 'pred_116_C.txt', 'pred_116_E.txt', 'pred_116_A.txt', 'pred_116_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])


# corr_face_single_deepimpression()


def corr_bg_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_110_O.txt', 'pred_110_C.txt', 'pred_110_E.txt', 'pred_110_A.txt', 'pred_110_S.txt']  # frame=6
    models = ['pred_117_O.txt', 'pred_117_C.txt', 'pred_117_E.txt', 'pred_117_A.txt', 'pred_117_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])


def corr_all_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_112_O.txt', 'pred_112_C.txt', 'pred_112_E.txt', 'pred_112_A.txt', 'pred_112_S.txt']  # frame=6
    models = ['pred_118_O.txt', 'pred_118_C.txt', 'pred_118_E.txt', 'pred_118_A.txt', 'pred_118_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])

corr_all_single_deepimpression()

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


# corr_bg_resnet()
